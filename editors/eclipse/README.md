# Boring Language — Eclipse Syntax Highlighting

Uses the [TM4E](https://github.com/eclipse/tm4e) plugin (TextMate for Eclipse),
which ships with Eclipse IDE for Java / Spring / etc. since 2022.

## Install

1. **Check TM4E is present**: *Help → About → Installation Details* — look for "TM4E".
   If missing: *Help → Eclipse Marketplace* → search "TM4E" → install.

2. **Register the grammar**:
   *Window → Preferences → TextMate → Grammars → Add…*
   Browse to `boring.tmLanguage.json` in this directory → OK.

3. **Map the file extension**:
   *Window → Preferences → TextMate → File Associations → Add…*
   Pattern: `*.br` — select the "Boring Language" grammar → OK.

4. Reopen any `.br` file — it should now be highlighted.

## Theme

Any TextMate-compatible theme works. The grammar uses standard scopes so
`keyword.control`, `string.quoted.double`, `comment.line`, etc. pick up
whatever colours your current Eclipse theme defines for those roles.
