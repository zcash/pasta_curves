use std::{path::Path, process::Command};

use super::super::instructions::ParsedInstruction;
use super::super::mutations::{self, Mutation};
use super::super::runner::Harness;

const ASM_SOURCE: &str = "src/asm/pasta_mul-armv8.S";
const SURVIVORS: &str = "xtask/asm-mutants-aarch64-apple.txt";

/// Every mnemonic the mutations read or write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mnemonic {
    Adc,
    Adcs,
    Add,
    Adds,
    Csel,
    Mov,
    Sbcs,
    Subs,
}

impl Mnemonic {
    /// The mnemonics whose behaviour depends on the carry flag.
    const MUTABLE: [Self; 4] = [Self::Adcs, Self::Adc, Self::Sbcs, Self::Csel];

    fn parse(text: &str) -> Option<Self> {
        Some(match text.to_ascii_lowercase().as_str() {
            "adc" => Self::Adc,
            "adcs" => Self::Adcs,
            "add" => Self::Add,
            "adds" => Self::Adds,
            "csel" => Self::Csel,
            "mov" => Self::Mov,
            "sbcs" => Self::Sbcs,
            "subs" => Self::Subs,
            _ => return None,
        })
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Adc => "adc",
            Self::Adcs => "adcs",
            Self::Add => "add",
            Self::Adds => "adds",
            Self::Csel => "csel",
            Self::Mov => "mov",
            Self::Sbcs => "sbcs",
            Self::Subs => "subs",
        }
    }
}

pub(crate) struct AArch64Apple;

impl Harness for AArch64Apple {
    fn source_path() -> &'static str {
        ASM_SOURCE
    }

    fn survivors_path() -> &'static str {
        SURVIVORS
    }

    fn mutations(source: &str) -> Result<Vec<Mutation>, String> {
        let mutable: Vec<&str> = Mnemonic::MUTABLE.iter().map(|m| m.as_str()).collect();
        mutations::prepare(source, &mutable, replacements)
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
    let mnemonic = Mnemonic::parse(instruction.mnemonic).ok_or_else(|| {
        format!(
            "unsupported mutation mnemonic {} at line {}",
            instruction.mnemonic, instruction.line
        )
    })?;
    let expected = if mnemonic == Mnemonic::Csel { 4 } else { 3 };
    if instruction.operands.len() != expected {
        return Err(format!(
            "{} at line {} has {} operands, expected {expected}",
            instruction.mnemonic,
            instruction.line,
            instruction.operands.len()
        ));
    }
    let emit = |mnemonic: Mnemonic, operands: &[&str]| {
        format!("{} {}", mnemonic.as_str(), operands.join(","))
    };
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

    let replacements = match mnemonic {
        Mnemonic::Adcs => vec![
            ("carry-clear", emit(Mnemonic::Adds, &instruction.operands)),
            (
                "carry-set",
                pair(
                    emit(Mnemonic::Subs, &[zero, zero, zero]),
                    emit(Mnemonic::Adcs, &instruction.operands),
                ),
            ),
        ],
        Mnemonic::Sbcs => vec![
            ("carry-set", emit(Mnemonic::Subs, &instruction.operands)),
            (
                "carry-clear",
                pair(
                    emit(Mnemonic::Adds, &[zero, zero, zero]),
                    emit(Mnemonic::Sbcs, &instruction.operands),
                ),
            ),
        ],
        Mnemonic::Adc => {
            let destination = instruction.operands[0];
            if destination.eq_ignore_ascii_case("xzr") {
                return Err("ADC mutation with xzr destination is unsupported".to_owned());
            }
            vec![
                ("carry-clear", emit(Mnemonic::Add, &instruction.operands)),
                (
                    "carry-set",
                    pair(
                        emit(Mnemonic::Add, &instruction.operands),
                        emit(Mnemonic::Add, &[destination, destination, "#1"]),
                    ),
                ),
            ]
        }
        Mnemonic::Csel => {
            let destination = instruction.operands[0];
            let when_true = instruction.operands[1];
            let when_false = instruction.operands[2];
            vec![
                (
                    "select-first",
                    emit(Mnemonic::Mov, &[destination, when_true]),
                ),
                (
                    "select-second",
                    emit(Mnemonic::Mov, &[destination, when_false]),
                ),
            ]
        }
        other => {
            return Err(format!("unsupported mutation mnemonic {}", other.as_str()));
        }
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
    fn mnemonic_names_round_trip() {
        for mnemonic in [
            Mnemonic::Adc,
            Mnemonic::Adcs,
            Mnemonic::Add,
            Mnemonic::Adds,
            Mnemonic::Csel,
            Mnemonic::Mov,
            Mnemonic::Sbcs,
            Mnemonic::Subs,
        ] {
            assert_eq!(Mnemonic::parse(mnemonic.as_str()), Some(mnemonic));
        }
        assert_eq!(Mnemonic::parse("ADCS"), Some(Mnemonic::Adcs));
        assert_eq!(Mnemonic::parse("umulh"), None);
    }

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
        assert!(
            mutation
                .apply(&SOURCE.replace("x1,x2,x3", "x1,x2,x4"))
                .is_err()
        );
    }

    #[test]
    fn enumerates_every_supported_instruction_in_the_backend() {
        let source = include_str!("../../../../src/asm/pasta_mul-armv8.S");
        let mutations = AArch64Apple::mutations(source).unwrap();
        // 77 ADCS + 29 ADC + 11 SBCS + 12 CSEL, with two mutations each.
        assert_eq!(mutations.len(), 258);
        assert_eq!(mutations.first().unwrap().line, 59);
        assert_eq!(mutations.last().unwrap().line, 472);
    }
}
