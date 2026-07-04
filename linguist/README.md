# Linguist submission for the Boring language

This directory contains everything needed to submit Boring to
[github-linguist/linguist](https://github.com/github-linguist/linguist).

## Files

| File | Purpose |
|------|---------|
| `Boring.tmLanguage.json` | TextMate grammar — copy to `grammars/` in the linguist repo |
| `samples/hello.br` | Basics: bindings, functions, structs, enums, collections |
| `samples/traits.br` | Traits, generics, closures, pipe operator, error handling |
| `samples/async.br` | Async tasks, channels, ownership qualifiers, streams, modules |
| `samples/gpu.br` | GPU kernel structs, CUDA/Metal targets, GPU memory qualifiers |
| `languages.yml` | Block to insert into `lib/linguist/languages.yml` |

## Submission steps

1. **Fork** https://github.com/github-linguist/linguist
2. **Copy** `Boring.tmLanguage.json` → `grammars/Boring.tmLanguage.json`
3. **Copy** `samples/` → `samples/Boring/`
4. **Edit** `lib/linguist/languages.yml` — insert the block from `languages.yml`
   in alphabetical order (between "Boo" and "BrightScript")
5. **Run** `bundle exec rake samples` to generate the heuristic data
6. **Open PR** with title: `Add Boring language`

## Grammar coverage

- Comments: `# line comment`
- Strings: `"..."` with `{interpolation}`, triple-quoted `"""..."""`
- Numbers: decimal, hex `0xFF`, binary `0b1010`, octal `0o755`, float
- Keywords: all 40+ language keywords
- Types: primitive aliases (`int`, `float`, `bool`, `string`, `str`)
- Type names: PascalCase identifiers
- Function definitions: `def`, `req`, `stream`, `task` forms + shorthand
- Function calls: highlighted with a distinct scope
- Attributes: `@derive`, `@test`, `@error`, etc.
- Ownership qualifiers: `'task`, `'actor`, `'shared`, `'heap`, `'weak`, etc.
- Operators: `|>`, `..`, `..<`, `->`, `?=`, `?.`, compound assignments
