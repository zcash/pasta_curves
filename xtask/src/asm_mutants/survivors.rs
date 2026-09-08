//! The tracked list of mutants that survive the test suite.

use std::collections::BTreeSet;

use super::mutations::Mutation;

/// An expected survivor: the mutation's id and the instruction it mutates.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct Expected {
    pub id: String,
    pub instruction: String,
}

/// Parses the tracked list, ignoring blank lines and `#` comments.
pub(super) fn parse(contents: &str) -> Result<Vec<Expected>, String> {
    let mut expected = Vec::new();
    for (index, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (id, instruction) = line
            .split_once('|')
            .ok_or_else(|| format!("line {}: expected `<id> | <instruction>`", index + 1))?;
        expected.push(Expected {
            id: id.trim().to_owned(),
            instruction: instruction.trim().to_owned(),
        });
    }

    let mut seen = BTreeSet::new();
    for entry in &expected {
        if !seen.insert(&entry.id) {
            return Err(format!("{} is listed more than once", entry.id));
        }
    }
    Ok(expected)
}

/// Checks that every listed mutant still names a real mutation site.
pub(super) fn verify_sites(expected: &[Expected], mutations: &[Mutation]) -> Result<(), String> {
    let mut problems = Vec::new();
    for entry in expected {
        match mutations.iter().find(|mutation| mutation.id == entry.id) {
            None => problems.push(format!("  {} no longer names a mutation site", entry.id)),
            Some(mutation) if mutation.instruction() != entry.instruction => {
                problems.push(format!(
                    "  {} now mutates {:?}, but is recorded against {:?}",
                    entry.id,
                    mutation.instruction(),
                    entry.instruction,
                ))
            }
            Some(_) => (),
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "the tracked survivors no longer match the assembly:\n{}",
            problems.join("\n"),
        ))
    }
}

/// Compares the mutants that survived this run against the tracked list.
pub(super) fn compare(expected: &[Expected], survived: &[&Mutation]) -> Result<(), String> {
    let listed: BTreeSet<&str> = expected.iter().map(|entry| entry.id.as_str()).collect();
    let observed: BTreeSet<&str> = survived
        .iter()
        .map(|mutation| mutation.id.as_str())
        .collect();

    let added: Vec<&&str> = observed.difference(&listed).collect();
    let removed: Vec<&&str> = listed.difference(&observed).collect();

    if added.is_empty() && removed.is_empty() {
        return Ok(());
    }

    let mut message = String::new();
    if !added.is_empty() {
        message.push_str(&format!(
            "{} mutant(s) survived that are not tracked, so the assembly has \
             gained untested carry decisions:\n",
            added.len(),
        ));
        for id in &added {
            let mutation = survived
                .iter()
                .find(|mutation| &mutation.id.as_str() == *id)
                .expect("added ids come from the survivors");
            message.push_str(&format!("  {mutation}\n"));
        }
        message.push_str(
            "Add a test that kills each one. If a mutant is equivalent, and so \
             cannot be killed by any input, record it in the tracked list with \
             the argument for why the path is unreachable.\n",
        );
    }
    if !removed.is_empty() {
        message.push_str(&format!(
            "{} tracked mutant(s) were killed, so their recorded justification \
             no longer holds:\n",
            removed.len(),
        ));
        for id in &removed {
            message.push_str(&format!("  {id}\n"));
        }
        message.push_str("This is an improvement. Drop those entries from the tracked list.\n");
    }
    Err(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIST: &str = concat!(
        "# a comment, and the justification for the entry below it\n",
        "\n",
        "adc/15/carry-clear | adc x23,xzr,xzr\n",
        "csel/9/select-first | csel x10,x10,x14,lo\n",
    );

    #[test]
    fn parses_entries_and_ignores_comments() {
        let expected = parse(LIST).unwrap();
        assert_eq!(expected.len(), 2);
        assert_eq!(expected[0].id, "adc/15/carry-clear");
        assert_eq!(expected[0].instruction, "adc x23,xzr,xzr");
    }

    #[test]
    fn rejects_a_malformed_line() {
        assert!(parse("adc/15/carry-clear\n").is_err());
    }

    #[test]
    fn rejects_a_duplicate_entry() {
        assert!(parse(&format!("{LIST}{LIST}")).is_err());
    }

    fn mutation(id: &str, instruction: &str) -> Mutation {
        Mutation::for_test(id, instruction)
    }

    #[test]
    fn accepts_a_list_that_matches_the_assembly() {
        let mutations = vec![
            mutation("adc/15/carry-clear", "adc x23,xzr,xzr"),
            mutation("csel/9/select-first", "csel x10,x10,x14,lo"),
        ];
        assert!(verify_sites(&parse(LIST).unwrap(), &mutations).is_ok());
    }

    #[test]
    fn rejects_an_entry_whose_site_is_gone() {
        let mutations = vec![mutation("adc/15/carry-clear", "adc x23,xzr,xzr")];
        let error = verify_sites(&parse(LIST).unwrap(), &mutations).unwrap_err();
        assert!(error.contains("csel/9/select-first"), "{error}");
        assert!(error.contains("no longer names a mutation site"), "{error}");
    }

    #[test]
    fn rejects_an_entry_that_drifted_onto_another_instruction() {
        let mutations = vec![
            mutation("adc/15/carry-clear", "adc x23,xzr,x17"),
            mutation("csel/9/select-first", "csel x10,x10,x14,lo"),
        ];
        let error = verify_sites(&parse(LIST).unwrap(), &mutations).unwrap_err();
        assert!(error.contains("now mutates"), "{error}");
    }

    #[test]
    fn accepts_survivors_that_match_the_list() {
        let expected = parse(LIST).unwrap();
        let survived = [
            mutation("adc/15/carry-clear", "adc x23,xzr,xzr"),
            mutation("csel/9/select-first", "csel x10,x10,x14,lo"),
        ];
        let survived: Vec<&Mutation> = survived.iter().collect();
        assert!(compare(&expected, &survived).is_ok());
    }

    #[test]
    fn reports_an_untracked_survivor_as_missing_coverage() {
        let expected = parse(LIST).unwrap();
        let survived = vec![
            mutation("adc/15/carry-clear", "adc x23,xzr,xzr"),
            mutation("csel/9/select-first", "csel x10,x10,x14,lo"),
            mutation("adcs/5/carry-clear", "adcs x22,x22,x17"),
        ];
        let survived: Vec<&Mutation> = survived.iter().collect();
        let error = compare(&expected, &survived).unwrap_err();
        assert!(error.contains("adcs/5/carry-clear"), "{error}");
        assert!(error.contains("untested carry decisions"), "{error}");
    }

    #[test]
    fn reports_a_tracked_mutant_that_is_now_killed() {
        let expected = parse(LIST).unwrap();
        let survived = [mutation("adc/15/carry-clear", "adc x23,xzr,xzr")];
        let survived: Vec<&Mutation> = survived.iter().collect();
        let error = compare(&expected, &survived).unwrap_err();
        assert!(error.contains("csel/9/select-first"), "{error}");
        assert!(error.contains("no longer holds"), "{error}");
    }
}
