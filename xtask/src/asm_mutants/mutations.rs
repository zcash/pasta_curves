use std::{collections::BTreeMap, fmt, ops::Range};

use super::instructions::{self, ParsedInstruction};

/// Generates all individual mutations from the given assembly.
pub(super) fn prepare(
    source: &str,
    mutable_mnemonics: &[&str],
    replacements: impl Fn(&ParsedInstruction) -> Result<Vec<(&'static str, String)>, String>,
) -> Result<Vec<Mutation>, String> {
    let mut ordinals = BTreeMap::<String, usize>::new();
    let mut mutations = Vec::new();

    for instruction in instructions::parse(source)? {
        let mnemonic = instruction.mnemonic.to_ascii_lowercase();
        if !mutable_mnemonics.contains(&mnemonic.as_str()) {
            continue;
        }
        let ordinal = ordinals.entry(mnemonic.clone()).or_default();
        *ordinal += 1;
        for (kind, replacement) in replacements(&instruction)? {
            mutations.push(Mutation {
                id: format!("{mnemonic}/{ordinal}/{kind}"),
                line: instruction.line,
                range: instruction.range.clone(),
                original: source[instruction.range.clone()].to_owned(),
                replacement,
            });
        }
    }

    if mutations.is_empty() {
        return Err("assembly contained no supported AArch64 mutation sites".to_owned());
    }
    Ok(mutations)
}

/// A mutation of an assembly instruction.
///
/// From [Go Assembly Mutation Testing]:
///
/// > Mutations turn an instruction that behaves differently based on a flag, into an
/// > equivalent instruction that behaves as if the flag was always or never set. They
/// > need not to change anything else, to avoid accidentally breaking the test run and
/// > causing a mutation testing false negative. In particular, we can’t use any register
/// > and we need to leave the final flags untouched.
///
/// [Go Assembly Mutation Testing]: https://words.filippo.io/assembly-mutation/
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Mutation {
    pub id: String,
    pub line: usize,
    range: Range<usize>,
    original: String,
    replacement: String,
}

impl Mutation {
    /// Applies this mutation to the assembly source.
    pub fn apply(&self, source: &str) -> Result<String, String> {
        let actual = source
            .get(self.range.clone())
            .ok_or_else(|| format!("{} has an invalid source span", self.id))?;
        if actual != self.original {
            return Err(format!(
                "{} expected {:?}, found {:?}",
                self.id, self.original, actual
            ));
        }
        Ok(instructions::replace(
            source,
            self.range.clone(),
            &self.replacement,
        ))
    }
}

impl fmt::Display for Mutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (line {})", self.id, self.line)
    }
}

pub(super) fn format(mutations: &[&Mutation]) -> String {
    mutations
        .iter()
        .map(|mutation| format!("  {mutation}"))
        .collect::<Vec<_>>()
        .join("\n")
}
