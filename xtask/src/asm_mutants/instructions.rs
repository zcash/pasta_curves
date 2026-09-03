#[derive(Debug)]
pub(super) struct ParsedInstruction<'a> {
    pub range: std::ops::Range<usize>,
    pub line: usize,
    pub indent: &'a str,
    pub mnemonic: &'a str,
    pub operands: Vec<&'a str>,
}

/// Parses generic assembly structure with Tree-sitter and retains exact source
/// spans. Architecture-specific code decides which instruction shapes it knows
/// how to mutate.
pub(super) fn parse(source: &str) -> Result<Vec<ParsedInstruction<'_>>, String> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_asm::LANGUAGE.into())
        .map_err(|error| format!("failed to load assembly grammar: {error}"))?;
    let tree = parser
        .parse(source, None)
        .ok_or("Tree-sitter failed to parse assembly")?;

    let mut instructions = Vec::new();
    collect_instructions(source, tree.root_node(), &mut instructions)?;
    Ok(instructions)
}

fn collect_instructions<'a>(
    source: &'a str,
    node: tree_sitter::Node<'_>,
    instructions: &mut Vec<ParsedInstruction<'a>>,
) -> Result<(), String> {
    if node.kind() == "instruction" {
        let mnemonic_node = node
            .child_by_field_name("kind")
            .ok_or_else(|| format!("instruction at {:?} has no mnemonic", node.range()))?;
        let mnemonic = &source[mnemonic_node.byte_range()];
        let mut operands = Vec::new();
        let mut children = node.walk();
        for child in node.named_children(&mut children) {
            if child.id() != mnemonic_node.id() {
                operands.push(&source[child.byte_range()]);
            }
        }
        // This grammar includes leading horizontal whitespace in the
        // instruction node's span. Exclude it so replacing an instruction
        // preserves the original indentation.
        let node_text = &source[node.byte_range()];
        let instruction_start =
            node.start_byte() + (node_text.len() - node_text.trim_start().len());
        let line_start = source[..instruction_start]
            .rfind('\n')
            .map_or(0, |index| index + 1);
        instructions.push(ParsedInstruction {
            range: instruction_start..node.end_byte(),
            line: mnemonic_node.start_position().row + 1,
            indent: &source[line_start..instruction_start],
            mnemonic,
            operands,
        });
        return Ok(());
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_instructions(source, child, instructions)?;
    }
    Ok(())
}

pub(super) fn replace(source: &str, range: std::ops::Range<usize>, replacement: &str) -> String {
    let mut output =
        String::with_capacity(source.len() - (range.end - range.start) + replacement.len());
    output.push_str(&source[..range.start]);
    output.push_str(replacement);
    output.push_str(&source[range.end..]);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_sitter_parses_generic_operand_shapes() {
        let parsed = parse(
            ".text\nfoo:\n  ldp x1, x2, [x3, #16] // pair\n  tbl v0.16b, {v1.16b, v2.16b}, v3.16b\n",
        )
        .unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].mnemonic, "ldp");
        assert_eq!(parsed[0].operands, ["x1", "x2", "[x3, #16]"]);
        assert_eq!(parsed[1].mnemonic, "tbl");
        // The generic grammar still identifies the instruction and its source
        // span when architecture-specific operands require error recovery.
        assert_eq!(
            &".text\nfoo:\n  ldp x1, x2, [x3, #16] // pair\n  tbl v0.16b, {v1.16b, v2.16b}, v3.16b\n"
                [parsed[1].range.clone()],
            "tbl v0.16b, {v1.16b, v2.16b}, v3.16b"
        );
    }
}
