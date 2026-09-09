use std::{path::Path, process::Command};

use super::super::instructions::ParsedInstruction;
use super::super::mutations::{self, Mutation};
use super::super::runner::Harness;

const ASM_SOURCE: &str = "src/asm/pasta_mul-armv8.S";

const MUTABLE_MNEMONICS: [&str; 4] = ["adcs", "adc", "sbcs", "csel"];

pub(crate) struct AArch64Apple;

impl Harness for AArch64Apple {
    fn source_path() -> &'static str {
        ASM_SOURCE
    }

    fn mutations(source: &str) -> Result<Vec<Mutation>, String> {
        mutations::prepare(source, &MUTABLE_MNEMONICS, replacements)
    }

    fn with_mutation(cmd: &mut Command, mutated_asm: Option<&Path>) {
        cmd.arg("--features").arg("aarch64-asm");

        if let Some(asm_path) = mutated_asm {
            cmd.env("PASTA_CURVES_ARMV8_ASM_SOURCE", asm_path);
        }
    }
}

fn replacements(
    instruction: &ParsedInstruction<'_>,
) -> Result<Vec<(&'static str, String)>, String> {
    let mnemonic = instruction.mnemonic.to_ascii_lowercase();
    let expected = if mnemonic == "csel" { 4 } else { 3 };
    if instruction.operands.len() != expected {
        return Err(format!(
            "{} at line {} has {} operands, expected {expected}",
            instruction.mnemonic,
            instruction.line,
            instruction.operands.len()
        ));
    }
    let emit = |mnemonic: &str, operands: &[&str]| format!("{mnemonic} {}", operands.join(","));
    let pair = |first: String, second: String| format!("{first}\n{}{second}", instruction.indent);
    let width = register_width(instruction.operands[0]).ok_or_else(|| {
        format!(
            "unsupported destination register {:?} for {} at line {}",
            instruction.operands[0], instruction.mnemonic, instruction.line
        )
    })?;
    if instruction.operands[..3]
        .iter()
        .any(|operand| register_width(operand) != Some(width))
    {
        return Err(format!(
            "mixed or unsupported registers for {} at line {}",
            instruction.mnemonic, instruction.line
        ));
    }
    let zero = if width == 64 { "xzr" } else { "wzr" };

    let replacements = match mnemonic.as_str() {
        "adcs" => vec![
            ("carry-clear", emit("adds", &instruction.operands)),
            (
                "carry-set",
                pair(
                    emit("subs", &[zero, zero, zero]),
                    emit("adcs", &instruction.operands),
                ),
            ),
        ],
        "sbcs" => vec![
            ("carry-set", emit("subs", &instruction.operands)),
            (
                "carry-clear",
                pair(
                    emit("adds", &[zero, zero, zero]),
                    emit("sbcs", &instruction.operands),
                ),
            ),
        ],
        "adc" => {
            let destination = instruction.operands[0];
            if destination.eq_ignore_ascii_case("xzr") {
                return Err("ADC mutation with xzr destination is unsupported".to_owned());
            }
            vec![
                ("carry-clear", emit("add", &instruction.operands)),
                (
                    "carry-set",
                    pair(
                        emit("add", &instruction.operands),
                        emit("add", &[destination, destination, "#1"]),
                    ),
                ),
            ]
        }
        "csel" => {
            let destination = instruction.operands[0];
            let when_true = instruction.operands[1];
            let when_false = instruction.operands[2];
            vec![
                ("select-first", emit("mov", &[destination, when_true])),
                ("select-second", emit("mov", &[destination, when_false])),
            ]
        }
        mnemonic => return Err(format!("unsupported mutation mnemonic {mnemonic}")),
    };

    Ok(replacements)
}

fn register_width(register: &str) -> Option<u8> {
    let register = register.to_ascii_lowercase();
    match register.as_str() {
        "xzr" => Some(64),
        "wzr" => Some(32),
        _ if register
            .strip_prefix('x')
            .is_some_and(|number| number.parse::<u8>().is_ok_and(|number| number <= 30)) =>
        {
            Some(64)
        }
        _ if register
            .strip_prefix('w')
            .is_some_and(|number| number.parse::<u8>().is_ok_and(|number| number <= 30)) =>
        {
            Some(32)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = concat!(
        "_func:\n",
        "    adcs x1,x2,x3 // carry\n",
        "    adc x4,x5,x6\n",
        "    sbcs xzr,x7,x8\n",
        "    csel x9,x10,x11,lo // choose\n",
    );

    #[test]
    fn parses_supported_instructions_and_builds_stable_ids() {
        let mutations = AArch64Apple::mutations(SOURCE).unwrap();
        assert_eq!(mutations.len(), 8);
        assert_eq!(mutations[0].id, "adcs/1/carry-clear");
        assert_eq!(mutations[1].id, "adcs/1/carry-set");
        assert_eq!(mutations[6].id, "csel/1/select-first");
        assert_eq!(mutations[7].id, "csel/1/select-second");
    }

    #[test]
    fn applies_exactly_one_mutation_and_preserves_comments() {
        let mutations = AArch64Apple::mutations(SOURCE).unwrap();
        let mutated = mutations[1].apply(SOURCE).unwrap();
        assert!(
            mutated.contains("    subs xzr,xzr,xzr\n    adcs"),
            "{mutated:?}"
        );
        assert!(mutated.contains("    adcs x1,x2,x3 // carry\n"));
        assert_eq!(mutated.matches("subs xzr,xzr,xzr").count(), 1);
    }

    #[test]
    fn rejects_changed_source() {
        let mutation = AArch64Apple::mutations(SOURCE).unwrap().remove(0);
        assert!(mutation
            .apply(&SOURCE.replace("x1,x2,x3", "x1,x2,x4"))
            .is_err());
    }

    #[test]
    fn enumerates_every_supported_instruction_in_the_backend() {
        let source = include_str!("../../../src/asm/pasta_mul-armv8.S");
        let mutations = AArch64Apple::mutations(source).unwrap();
        // 77 ADCS + 29 ADC + 11 SBCS + 12 CSEL, with two mutations each.
        assert_eq!(mutations.len(), 258);
        assert_eq!(mutations.first().unwrap().line, 59);
        assert_eq!(mutations.last().unwrap().line, 472);
    }
}
