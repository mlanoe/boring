// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of Boring.
// Boring is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// See the LICENSE file at the project root for the full text.

use thiserror::Error;

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let preprocessed = preprocess_triple_strings(source)?;
    Lexer::new(&preprocessed).tokenize()
}

/// Expand `"""..."""` triple-quoted strings into regular `"..."` single-line strings
/// before the line-by-line tokeniser runs.
///
/// Transformation rules:
///   - The newline immediately after the opening `"""` is stripped.
///   - The newline immediately before the closing `"""` is stripped.
///   - The leading indentation of the closing `"""` is stripped from every
///     content line (common-indent removal, like Python `textwrap.dedent`).
///   - Internal newlines are encoded as `\n` escape sequences so that the
///     existing single-line `lex_string` function can handle them.
///   - `"` characters inside the content are escaped as `\"`.
///   - `{expr}` interpolation holes are left intact — `lex_string` processes them.
///   - The same number of physical newlines are re-emitted after the closing `"`
///     so that tokens following the triple-quoted string keep their correct line
///     numbers.
///
/// Regular `"..."` strings found while scanning are copied verbatim (so a `"""`
/// that happens to appear inside a regular string literal is never misidentified).
fn preprocess_triple_strings(source: &str) -> Result<String, LexError> {
    let chars: Vec<char> = source.chars().collect();
    let n = chars.len();
    let mut result = String::with_capacity(source.len());
    let mut i = 0;
    let mut line = 1usize;

    while i < n {
        let ch = chars[i];

        // Track newlines for error reporting, copy them through unchanged.
        if ch == '\n' {
            line += 1;
            result.push('\n');
            i += 1;
            continue;
        }

        // Skip comments (everything from `#` to end-of-line) verbatim — no
        // string processing inside comments.
        if ch == '#' {
            while i < n && chars[i] != '\n' {
                result.push(chars[i]);
                i += 1;
            }
            continue;
        }

        if ch == '"' {
            // ── Triple-quoted string ────────────────────────────────────────
            if chars.get(i + 1) == Some(&'"') && chars.get(i + 2) == Some(&'"') {
                let start_line = line;
                i += 3; // consume opening """

                let mut raw = String::new();
                let mut newlines_consumed = 0usize;
                let mut closed = false;

                while i < n {
                    // Detect closing """
                    if chars[i] == '"'
                        && chars.get(i + 1) == Some(&'"')
                        && chars.get(i + 2) == Some(&'"')
                    {
                        i += 3;
                        closed = true;
                        break;
                    }
                    if chars[i] == '\n' {
                        newlines_consumed += 1;
                        line += 1;
                    }
                    raw.push(chars[i]);
                    i += 1;
                }

                if !closed {
                    return Err(LexError::UnterminatedString { line: start_line });
                }

                // Dedent and encode as a single-line string token.
                let content = dedent_triple(&raw);
                result.push('"');
                for c in content.chars() {
                    match c {
                        '"'  => { result.push('\\'); result.push('"'); }
                        '\n' => { result.push('\\'); result.push('n'); }
                        '\r' => { result.push('\\'); result.push('r'); }
                        // Backslashes and `{}`/`{{`/`}}` are left intact so that
                        // lex_string handles escape sequences and interpolation holes
                        // exactly as it would for a regular string.
                        other => result.push(other),
                    }
                }
                result.push('"');

                // Re-emit the consumed newlines to keep subsequent token line
                // numbers correct.
                for _ in 0..newlines_consumed {
                    result.push('\n');
                }
                continue;
            }

            // ── Regular single-line string — copy verbatim ─────────────────
            // We must copy the full string (including escape sequences and
            // `{...}` interpolation holes) without touching it, so that a `"""`
            // sequence inside a regular string is never misidentified.
            result.push('"');
            i += 1;
            while i < n {
                let c = chars[i];
                result.push(c);
                i += 1;
                if c == '\\' {
                    // Escape: copy the next character unconditionally.
                    if i < n {
                        result.push(chars[i]);
                        i += 1;
                    }
                } else if c == '{' {
                    // Interpolation hole: copy until the matching `}` at depth 0.
                    let mut depth = 1usize;
                    while i < n && depth > 0 {
                        let ic = chars[i];
                        result.push(ic);
                        i += 1;
                        if ic == '{' { depth += 1; }
                        else if ic == '}' { depth -= 1; }
                    }
                } else if c == '"' || c == '\n' {
                    // End of string (or unterminated — lex_string will report the error).
                    if c == '\n' { line += 1; }
                    break;
                }
            }
            continue;
        }

        result.push(ch);
        i += 1;
    }
    Ok(result)
}

/// Strip common leading indentation from a triple-quoted string's raw content.
///
/// Steps:
///   1. Strip the optional newline immediately after the opening `"""`.
///   2. Strip the optional newline immediately before the closing `"""`.
///   3. Determine the minimum indentation of all non-empty lines.
///   4. Remove that many leading bytes from every line.
fn dedent_triple(raw: &str) -> String {
    // Strip optional leading newline (the one right after """).
    let s = raw.strip_prefix('\n').unwrap_or(raw);
    // Strip optional trailing newline (the one right before """).
    let s = s.strip_suffix('\n').unwrap_or(s);

    if s.is_empty() {
        return String::new();
    }

    // Minimum indentation among non-empty lines (byte count — safe because
    // indentation is always ASCII spaces or tabs).
    let indent = s
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.bytes().take_while(|&b| b == b' ' || b == b'\t').count())
        .min()
        .unwrap_or(0);

    if indent == 0 {
        return s.to_string();
    }

    s.lines()
        .map(|l| {
            if l.trim().is_empty() {
                ""
            } else if l.len() >= indent {
                &l[indent..]
            } else {
                l
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Error)]
pub enum LexError {
    #[error("line {line}: unexpected character '{ch}'")]
    UnexpectedChar { line: usize, ch: char },
    #[error("line {line}: unterminated string literal")]
    UnterminatedString { line: usize },
    #[error("line {line}: inconsistent indentation (mixed tabs and spaces)")]
    MixedIndentation { line: usize },
    #[error("line {line}: dedent does not match any outer indentation level")]
    InvalidDedent { line: usize },
}

impl LexError {
    pub fn line(&self) -> usize {
        match self {
            LexError::UnexpectedChar { line, .. } => *line,
            LexError::UnterminatedString { line } => *line,
            LexError::MixedIndentation { line } => *line,
            LexError::InvalidDedent { line } => *line,
        }
    }

    pub fn msg(&self) -> String {
        match self {
            LexError::UnexpectedChar { ch, .. } => format!("unexpected character '{}'", ch),
            LexError::UnterminatedString { .. } => "unterminated string literal".to_string(),
            LexError::MixedIndentation { .. } => "inconsistent indentation (mixed tabs and spaces)".to_string(),
            LexError::InvalidDedent { .. } => "dedent does not match any outer indentation level".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RawInterpPart {
    Lit(String),
    Hole(String),
    HoleFormatted(String, String), // (expr_src, fmt_spec)
}

/// Split a hole's raw content on the first `:` at depth 0 (not inside parens/brackets/braces).
/// `{expr:spec}` → `("expr", Some("spec"))`, `{expr}` → `("expr", None)`.
fn split_hole_fmt(s: &str) -> (&str, Option<&str>) {
    let mut depth: i32 = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ':' if depth == 0 => {
                let fmt = s[i + 1..].trim_start();
                return if fmt.is_empty() { (s, None) } else { (&s[..i], Some(fmt)) };
            }
            _ => {}
        }
    }
    (s, None)
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Int(i64),
    Float(f64),
    Str(String),
    StringInterp(Vec<RawInterpPart>),
    Bool(bool),
    Nil,

    // Identifier
    Ident(String),

    // Keywords
    Let, Var, Def, Return,
    If, Elif, Else,
    Match,
    While, Do, Loop, Wait,
    For, In,
    Break, Continue,
    Struct, Enum, Trait, Use, Ext,
    As,
    And, Or, Not,
    Is,
    SelfKw,
    Throw, Throws, Try, Catch,
    Defer,
    Void,
    Pub,
    Guard,
    Task,
    Join,
    Stream,
    Yield,
    Static,
    Type,
    Req,
    Transient,
    With,
    Get,
    Set,
    Init,
    Pass,
    Native,
    Mod,

    // Operators
    Plus, Minus, Star, Slash, Percent,
    Eq,
    EqEq, EqEqEq, BangEq,
    Lt, Gt, LtEq, GtEq,
    Dot, DotDot, DotDotEq, DotDotDot,
    Question, QuestionDot, Bang,
    Tick,
    // Bitwise operators
    Ampersand, Pipe, Caret, Tilde,
    // Compound assignment
    PlusEq, MinusEq, StarEq, SlashEq, PercentEq,
    AmpersandEq, PipeEq, CaretEq,
    QuestionEq,  // `?=` — assign if nil
    PipeArrow,   // `|>` — pipe operator

    // Delimiters
    LParen, RParen,
    LBracket, RBracket,
    LBrace, RBrace,
    Comma, Colon,
    At,

    // Layout
    Newline, Semicolon, Indent, Dedent, Eof,

    // Comment (full-line `# text`) — preserved for the transpiler
    Comment(String),
}

struct Lexer<'a> {
    source: &'a str,
    line: usize,
    indent_stack: Vec<usize>,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self { source, line: 0, indent_stack: vec![0] }
    }

    fn tokenize(mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens: Vec<Token> = Vec::new();
        let lines: Vec<&str> = self.source.lines().collect();
        let mut i = 0;
        // Track open paren/bracket/brace depth to suppress indent/dedent inside them.
        // Exception: colon-blocks inside parens (e.g. lambdas) still need indent/dedent tracking.
        let mut paren_depth: i32 = 0;
        // Number of active colon-blocks opened while paren_depth > 0.
        // When > 0, we still apply indent/dedent logic even though paren_depth > 0.
        let mut inner_colon_blocks: i32 = 0;
        // The indent level at the time each inner colon block was opened, so we know when to pop.
        let mut inner_block_base_indents: Vec<usize> = Vec::new();
        while i < lines.len() {
            self.line = i + 1;
            let raw = lines[i];
            let trimmed = raw.trim_start();
            if trimmed.is_empty() {
                if paren_depth == 0 {
                    tokens.push(Token { kind: TokenKind::Newline, line: self.line });
                }
                i += 1;
                continue;
            }
            let apply_indent = paren_depth == 0 || inner_colon_blocks > 0;
            if trimmed.starts_with('#') && apply_indent {
                // Emit any pending DEDENTs based on the comment line's indentation,
                // so a top-level comment after an indented block closes that block correctly.
                let comment_indent = measure_indent(raw, self.line)?;
                let cur_indent = *self.indent_stack.last().unwrap();
                if comment_indent < cur_indent {
                    loop {
                        let top = *self.indent_stack.last().unwrap();
                        if top <= comment_indent { break; }
                        self.indent_stack.pop();
                        tokens.push(Token { kind: TokenKind::Dedent, line: self.line });
                    }
                }
                let text = trimmed[1..].trim().to_string();
                tokens.push(Token { kind: TokenKind::Comment(text), line: self.line });
                tokens.push(Token { kind: TokenKind::Newline, line: self.line });
                i += 1;
                continue;
            }
            let line_indent = if apply_indent { measure_indent(raw, self.line)? } else { 0 };
            if apply_indent {
                // Before emitting this line's tokens, check if we should close inner colon blocks
                // because indentation dropped back to or below their base level.
                while inner_colon_blocks > 0 {
                    let base = *inner_block_base_indents.last().unwrap();
                    if line_indent <= base {
                        // We've exited this inner block; pop blocks until we reach the right level.
                        // The DEDENT tokens will be emitted below in the normal dedent logic.
                        inner_colon_blocks -= 1;
                        inner_block_base_indents.pop();
                    } else {
                        break;
                    }
                }
                let cur_indent = *self.indent_stack.last().unwrap();
                if line_indent > cur_indent {
                    self.indent_stack.push(line_indent);
                    tokens.push(Token { kind: TokenKind::Indent, line: self.line });
                } else if line_indent < cur_indent {
                    loop {
                        let top = *self.indent_stack.last().unwrap();
                        if top == line_indent { break; }
                        if top < line_indent {
                            return Err(LexError::InvalidDedent { line: self.line });
                        }
                        self.indent_stack.pop();
                        tokens.push(Token { kind: TokenKind::Dedent, line: self.line });
                    }
                }
            }
            let content = raw.trim_start();
            let content = &content[..comment_start(content)];
            let line_tokens = lex_line(content.trim_end(), self.line)?;
            // Update paren depth and track colon-blocks inside parens.
            let mut last_meaningful_is_colon = false;
            for t in &line_tokens {
                match &t.kind {
                    TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace => {
                        paren_depth += 1;
                        last_meaningful_is_colon = false;
                    }
                    TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace => {
                        paren_depth -= 1;
                        last_meaningful_is_colon = false;
                    }
                    TokenKind::Colon => { last_meaningful_is_colon = true; }
                    _ => { last_meaningful_is_colon = false; }
                }
            }
            // If this line ended with ':' while inside parens, the next lines form a block body.
            if last_meaningful_is_colon && paren_depth > 0 {
                inner_colon_blocks += 1;
                inner_block_base_indents.push(line_indent);
            }
            tokens.extend(line_tokens);
            // Emit newline when at top level or when ending a colon-block line inside parens.
            if paren_depth == 0 || last_meaningful_is_colon || inner_colon_blocks > 0 {
                tokens.push(Token { kind: TokenKind::Newline, line: self.line });
            }
            i += 1;
        }
        // Close all open indents
        while self.indent_stack.len() > 1 {
            self.indent_stack.pop();
            tokens.push(Token { kind: TokenKind::Dedent, line: self.line });
        }
        tokens.push(Token { kind: TokenKind::Eof, line: self.line });
        Ok(tokens)
    }
}

/// Returns the byte index of the first `#` that is not inside a string literal,
/// or `s.len()` if there is no such `#`.
/// Handles `\"` escapes and `{...}` interpolation holes inside strings.
fn comment_start(s: &str) -> usize {
    let mut chars = s.char_indices();
    while let Some((idx, ch)) = chars.next() {
        match ch {
            '#' => return idx,
            '"' => {
                // skip over the string literal, respecting escapes and {..} holes
                loop {
                    match chars.next() {
                        None | Some((_, '\n')) => break,
                        Some((_, '"')) => break,
                        Some((_, '{')) => {
                            // `{{` → literal `{`; anything else → interpolation hole
                            match chars.next() {
                                Some((_, '{')) => {}  // {{ escaped
                                Some((_, '}')) => {}  // {} empty hole
                                _ => {
                                    // Skip hole content until matching `}`
                                    let mut depth = 1usize;
                                    loop {
                                        match chars.next() {
                                            None => break,
                                            Some((_, '{')) => depth += 1,
                                            Some((_, '}')) => {
                                                depth -= 1;
                                                if depth == 0 { break; }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                        Some((_, '\\')) => { chars.next(); } // any escape: consume next char
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    s.len()
}

fn measure_indent(line: &str, line_no: usize) -> Result<usize, LexError> {
    let mut spaces = 0usize;
    let mut tabs = 0usize;
    for ch in line.chars() {
        match ch {
            ' '  => spaces += 1,
            '\t' => tabs += 1,
            _    => break,
        }
    }
    if spaces > 0 && tabs > 0 {
        return Err(LexError::MixedIndentation { line: line_no });
    }
    Ok(spaces + tabs * 4)
}

type CharIter<'a> = std::iter::Peekable<std::str::CharIndices<'a>>;

fn lex_line(content: &str, line: usize) -> Result<Vec<Token>, LexError> {
    let mut chars = content.char_indices().peekable();
    let mut tokens = Vec::new();
    while let Some(&(_, ch)) = chars.peek() {
        if ch.is_whitespace() { chars.next(); continue; }
        let tok = lex_token(&mut chars, line)?;
        tokens.push(tok);
    }
    Ok(tokens)
}

fn lex_token(chars: &mut CharIter<'_>, line: usize) -> Result<Token, LexError> {
    let (_, ch) = chars.next().unwrap();
    let kind = match ch {
        '(' => TokenKind::LParen,
        ')' => TokenKind::RParen,
        '[' => TokenKind::LBracket,
        ']' => TokenKind::RBracket,
        '{' => TokenKind::LBrace,
        '}' => TokenKind::RBrace,
        ',' => TokenKind::Comma,
        ';' => TokenKind::Semicolon,
        ':' => TokenKind::Colon,
        '+' => {
            if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next(); TokenKind::PlusEq
            } else { TokenKind::Plus }
        }
        '-' => {
            if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next(); TokenKind::MinusEq
            } else { TokenKind::Minus }
        }
        '*' => {
            if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next(); TokenKind::StarEq
            } else { TokenKind::Star }
        }
        '/' => {
            if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next(); TokenKind::SlashEq
            } else { TokenKind::Slash }
        }
        '%' => {
            if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next(); TokenKind::PercentEq
            } else { TokenKind::Percent }
        }
        '&' => {
            if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next(); TokenKind::AmpersandEq
            } else { TokenKind::Ampersand }
        }
        '|' => {
            if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next(); TokenKind::PipeEq
            } else if chars.peek().map(|(_, c)| *c == '>').unwrap_or(false) {
                chars.next(); TokenKind::PipeArrow
            } else { TokenKind::Pipe }
        }
        '^' => {
            if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next(); TokenKind::CaretEq
            } else { TokenKind::Caret }
        }
        '~' => TokenKind::Tilde,
        '@' => TokenKind::At,
        '\'' => TokenKind::Tick,
        '?' => {
            if chars.peek().map(|(_, c)| *c == '.').unwrap_or(false) {
                chars.next();
                TokenKind::QuestionDot
            } else if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next();
                TokenKind::QuestionEq
            } else {
                TokenKind::Question
            }
        }
        '!' => {
            if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next();
                TokenKind::BangEq
            } else {
                TokenKind::Bang
            }
        }
        '=' => {
            if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next();
                if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                    chars.next();
                    TokenKind::EqEqEq
                } else {
                    TokenKind::EqEq
                }
            } else {
                TokenKind::Eq
            }
        }
        '<' => {
            if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next();
                TokenKind::LtEq
            } else {
                TokenKind::Lt
            }
        }
        '>' => {
            if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                chars.next();
                TokenKind::GtEq
            } else {
                TokenKind::Gt
            }
        }
        '.' => {
            if chars.peek().map(|(_, c)| *c == '.').unwrap_or(false) {
                chars.next();
                if chars.peek().map(|(_, c)| *c == '.').unwrap_or(false) {
                    chars.next();
                    TokenKind::DotDotDot
                } else if chars.peek().map(|(_, c)| *c == '=').unwrap_or(false) {
                    chars.next();
                    TokenKind::DotDotEq
                } else {
                    TokenKind::DotDot
                }
            } else {
                TokenKind::Dot
            }
        }
        '"' => lex_string(chars, line)?,
        c if c.is_ascii_digit() => lex_number(c, chars)?,
        c if c.is_alphabetic() || c == '_' => {
            let s = lex_ident(c, chars);
            keyword_or_ident(s)
        }
        other => return Err(LexError::UnexpectedChar { line, ch: other }),
    };
    Ok(Token { kind, line })
}

fn lex_string(chars: &mut CharIter<'_>, line: usize) -> Result<TokenKind, LexError> {
    let mut parts: Vec<RawInterpPart> = Vec::new();
    let mut current_lit = String::new();
    loop {
        match chars.next() {
            None | Some((_, '\n')) => return Err(LexError::UnterminatedString { line }),
            Some((_, '"')) => break,
            // `{{` → literal `{`
            Some((_, '{')) => {
                if chars.peek().map(|(_, c)| *c) == Some('{') {
                    chars.next();
                    current_lit.push('{');
                } else {
                    // Interpolation hole: collect until matching `}`
                    if !current_lit.is_empty() {
                        parts.push(RawInterpPart::Lit(std::mem::take(&mut current_lit)));
                    }
                    let mut inner = String::new();
                    let mut depth = 0usize;
                    loop {
                        match chars.next() {
                            None | Some((_, '\n')) => return Err(LexError::UnterminatedString { line }),
                            Some((_, '{')) => { depth += 1; inner.push('{'); }
                            Some((_, '}')) => {
                                if depth == 0 { break; }
                                depth -= 1;
                                inner.push('}');
                            }
                            Some((_, c)) => inner.push(c),
                        }
                    }
                    let part = match split_hole_fmt(&inner) {
                        (expr, Some(fmt)) => RawInterpPart::HoleFormatted(expr.trim().to_string(), fmt.to_string()),
                        (expr, None)      => RawInterpPart::Hole(expr.trim().to_string()),
                    };
                    parts.push(part);
                }
            }
            // `}}` → literal `}`
            Some((_, '}')) => {
                if chars.peek().map(|(_, c)| *c) == Some('}') {
                    chars.next();
                }
                current_lit.push('}');
            }
            Some((_, '\\')) => {
                match chars.peek() {
                    Some((_, 'n')) => { chars.next(); current_lit.push('\n'); }
                    Some((_, 't')) => { chars.next(); current_lit.push('\t'); }
                    Some((_, 'r')) => { chars.next(); current_lit.push('\r'); }
                    Some((_, '0')) => { chars.next(); current_lit.push('\0'); }
                    Some((_, '\\')) => { chars.next(); current_lit.push('\\'); }
                    Some((_, '"')) => { chars.next(); current_lit.push('"'); }
                    Some((_, 'u')) => {
                        chars.next(); // consume 'u'
                        if chars.peek().map(|(_, c)| *c) == Some('{') {
                            chars.next(); // consume '{'
                            let mut hex = String::new();
                            loop {
                                match chars.peek() {
                                    Some((_, '}')) => { chars.next(); break; }
                                    Some((_, c)) if c.is_ascii_hexdigit() => {
                                        hex.push(*c);
                                        chars.next();
                                    }
                                    _ => break,
                                }
                            }
                            if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                if let Some(ch) = char::from_u32(code) {
                                    current_lit.push(ch);
                                }
                            }
                        } else {
                            current_lit.push('u');
                        }
                    }
                    _ => { current_lit.push('\\'); }
                }
            }
            Some((_, c)) => current_lit.push(c),
        }
    }
    if parts.is_empty() {
        Ok(TokenKind::Str(current_lit))
    } else {
        if !current_lit.is_empty() {
            parts.push(RawInterpPart::Lit(current_lit));
        }
        Ok(TokenKind::StringInterp(parts))
    }
}

fn lex_number(first: char, chars: &mut CharIter<'_>) -> Result<TokenKind, LexError> {
    // Hex / binary / octal prefix: 0x, 0b, 0o
    if first == '0' {
        let prefix = chars.peek().map(|(_, c)| *c);
        match prefix {
            Some('x') | Some('X') => {
                chars.next();
                let digits = lex_digits(chars, |c| c.is_ascii_hexdigit());
                let val = u64::from_str_radix(&digits, 16).unwrap_or(0) as i64;
                return Ok(TokenKind::Int(val));
            }
            Some('b') | Some('B') => {
                chars.next();
                let digits = lex_digits(chars, |c| matches!(c, '0' | '1'));
                let val = u64::from_str_radix(&digits, 2).unwrap_or(0) as i64;
                return Ok(TokenKind::Int(val));
            }
            Some('o') | Some('O') => {
                chars.next();
                let digits = lex_digits(chars, |c| matches!(c, '0'..='7'));
                let val = u64::from_str_radix(&digits, 8).unwrap_or(0) as i64;
                return Ok(TokenKind::Int(val));
            }
            _ => {}
        }
    }

    let mut s = String::from(first);
    while chars.peek().map(|(_, c)| c.is_ascii_digit() || *c == '_').unwrap_or(false) {
        let c = chars.next().unwrap().1;
        if c != '_' { s.push(c); }
    }
    // Check for float: digit '.' digit (not '..' range)
    if chars.peek().map(|(_, c)| *c == '.').unwrap_or(false) {
        let mut peek2 = chars.clone();
        peek2.next();
        if peek2.peek().map(|(_, c)| c.is_ascii_digit()).unwrap_or(false) {
            s.push(chars.next().unwrap().1); // consume '.'
            while chars.peek().map(|(_, c)| c.is_ascii_digit() || *c == '_').unwrap_or(false) {
                let c = chars.next().unwrap().1;
                if c != '_' { s.push(c); }
            }
            // Optional exponent
            if chars.peek().map(|(_, c)| *c == 'e' || *c == 'E').unwrap_or(false) {
                s.push(chars.next().unwrap().1);
                if chars.peek().map(|(_, c)| *c == '+' || *c == '-').unwrap_or(false) {
                    s.push(chars.next().unwrap().1);
                }
                while chars.peek().map(|(_, c)| c.is_ascii_digit()).unwrap_or(false) {
                    s.push(chars.next().unwrap().1);
                }
            }
            return Ok(TokenKind::Float(s.parse().unwrap()));
        }
    }
    Ok(TokenKind::Int(s.parse().unwrap()))
}

/// Collect digits (skipping `_` separators) while `pred` matches.
fn lex_digits(chars: &mut CharIter<'_>, pred: impl Fn(char) -> bool) -> String {
    let mut s = String::new();
    while let Some(&(_, c)) = chars.peek() {
        if c == '_' { chars.next(); continue; }
        if pred(c) { s.push(c); chars.next(); } else { break; }
    }
    s
}

fn lex_ident(first: char, chars: &mut CharIter<'_>) -> String {
    let mut s = String::from(first);
    while chars.peek().map(|(_, c)| c.is_alphanumeric() || *c == '_').unwrap_or(false) {
        s.push(chars.next().unwrap().1);
    }
    s
}

fn keyword_or_ident(s: String) -> TokenKind {
    match s.as_str() {
        "let"      => TokenKind::Let,
        "var"      => TokenKind::Var,
        "def"      => TokenKind::Def,
        "return"   => TokenKind::Return,
        "if"       => TokenKind::If,
        "elif"     => TokenKind::Elif,
        "else"     => TokenKind::Else,
        "match"    => TokenKind::Match,
        "while"    => TokenKind::While,
        "do"       => TokenKind::Do,
        "loop"     => TokenKind::Loop,
        "wait"     => TokenKind::Wait,
        "for"      => TokenKind::For,
        "in"       => TokenKind::In,
        "break"    => TokenKind::Break,
        "continue" => TokenKind::Continue,
        "struct"   => TokenKind::Struct,
        "enum"     => TokenKind::Enum,
        "trait"    => TokenKind::Trait,
        "use"      => TokenKind::Use,
        "ext"      => TokenKind::Ext,
        "as"       => TokenKind::As,
        "and"      => TokenKind::And,
        "or"       => TokenKind::Or,
        "not"      => TokenKind::Not,
        "is"       => TokenKind::Is,
        "self"     => TokenKind::SelfKw,
        "throw"    => TokenKind::Throw,
        "throws"   => TokenKind::Throws,
        "try"      => TokenKind::Try,
        "catch"    => TokenKind::Catch,
        "defer"    => TokenKind::Defer,
        "void"     => TokenKind::Void,
        "pub"      => TokenKind::Pub,
        "guard"    => TokenKind::Guard,
        "task"     => TokenKind::Task,
        "join"     => TokenKind::Join,
        "stream"   => TokenKind::Stream,
        "yield"    => TokenKind::Yield,
        "static"   => TokenKind::Static,
        "type"     => TokenKind::Type,
        "req"      => TokenKind::Req,
        "transient" => TokenKind::Transient,
        "with"      => TokenKind::With,
        "set"      => TokenKind::Set,
        "init"     => TokenKind::Init,
        "pass"     => TokenKind::Pass,
        "native"   => TokenKind::Native,
        "mod"      => TokenKind::Mod,
        "true"     => TokenKind::Bool(true),
        "false"    => TokenKind::Bool(false),
        "nil"      => TokenKind::Nil,
        _          => TokenKind::Ident(s),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        lex(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_keywords() {
        let k = kinds("let var def req transient return if elif else match while for in break continue struct enum trait use as and or not true false nil try catch defer throw throws");
        assert!(k.contains(&TokenKind::Let));
        assert!(k.contains(&TokenKind::Try));
        assert!(k.contains(&TokenKind::Catch));
        assert!(k.contains(&TokenKind::Defer));
        assert!(k.contains(&TokenKind::Throw));
        assert!(k.contains(&TokenKind::Throws));
        assert!(k.contains(&TokenKind::Bool(true)));
        assert!(k.contains(&TokenKind::Bool(false)));
        assert!(k.contains(&TokenKind::Nil));
        assert!(k.contains(&TokenKind::Req));
        assert!(k.contains(&TokenKind::Transient));
    }

    #[test]
    fn test_integer() {
        assert_eq!(kinds("42"), vec![TokenKind::Int(42), TokenKind::Newline, TokenKind::Eof]);
    }

    #[test]
    fn test_float() {
        assert_eq!(kinds("3.14"), vec![TokenKind::Float(3.14), TokenKind::Newline, TokenKind::Eof]);
    }

    #[test]
    fn test_float_not_confused_with_range() {
        let k = kinds("1..3");
        assert!(k.contains(&TokenKind::Int(1)));
        assert!(k.contains(&TokenKind::DotDot));
        assert!(k.contains(&TokenKind::Int(3)));
    }

    #[test]
    fn test_string() {
        assert_eq!(kinds("\"hello\""), vec![TokenKind::Str("hello".into()), TokenKind::Newline, TokenKind::Eof]);
    }

    #[test]
    fn test_string_interp_lex() {
        let k = kinds("\"hi {name}!\"");
        assert!(matches!(&k[0], TokenKind::StringInterp(_)));
    }

    #[test]
    fn test_string_no_interp() {
        assert_eq!(kinds("\"plain\""), vec![TokenKind::Str("plain".into()), TokenKind::Newline, TokenKind::Eof]);
    }

    #[test]
    fn test_indent_dedent() {
        let src = "if true:\n    x\n";
        let k = kinds(src);
        assert!(k.contains(&TokenKind::Indent));
        assert!(k.contains(&TokenKind::Dedent));
    }

    #[test]
    fn test_operators() {
        let k = kinds("+ - * / % == != < > <= >= =");
        assert!(k.contains(&TokenKind::Plus));
        assert!(k.contains(&TokenKind::EqEq));
        assert!(k.contains(&TokenKind::BangEq));
        assert!(k.contains(&TokenKind::LtEq));
        assert!(k.contains(&TokenKind::GtEq));
    }

    #[test]
    fn test_range_tokens() {
        let k = kinds("0..3");
        assert!(k.contains(&TokenKind::DotDot));
    }

    #[test]
    fn test_range_inclusive_tokens() {
        let k = kinds("0..=3");
        assert!(k.contains(&TokenKind::DotDotEq));
    }

    #[test]
    fn test_braces() {
        let k = kinds("{ }");
        assert!(k.contains(&TokenKind::LBrace));
        assert!(k.contains(&TokenKind::RBrace));
    }

    #[test]
    fn test_bang() {
        let k = kinds("!");
        assert!(k.contains(&TokenKind::Bang));
    }

    #[test]
    fn test_comment_preserved() {
        // Comments are kept as Comment tokens (the transpiler uses them).
        let k = kinds("# this is a comment");
        assert_eq!(k, vec![TokenKind::Comment("this is a comment".into()), TokenKind::Newline, TokenKind::Eof]);
    }

    #[test]
    fn test_inline_comment_stripped() {
        // tokens before '#' survive; rest is dropped
        let k = kinds("let x = 42 # inline comment");
        assert!(k.contains(&TokenKind::Let));
        assert!(k.contains(&TokenKind::Int(42)));
        // "inline", "comment" must NOT appear as identifiers
        assert!(!k.iter().any(|t| matches!(t, TokenKind::Ident(s) if s == "comment")));
    }

    #[test]
    fn test_hash_inside_string_not_comment() {
        // '#' inside a string literal must not be treated as a comment start
        let k = kinds("\"foo#bar\"");
        assert_eq!(k, vec![TokenKind::Str("foo#bar".into()), TokenKind::Newline, TokenKind::Eof]);
    }

    #[test]
    fn test_hash_inside_interp_string_not_comment() {
        // '#' inside an interpolated string must not start a comment
        let k = kinds("\"{x}#suffix\"");
        assert!(matches!(&k[0], TokenKind::StringInterp(parts) if {
            parts.iter().any(|p| matches!(p, RawInterpPart::Lit(s) if s == "#suffix"))
        }));
    }

    #[test]
    fn test_unicode_escape_emoji() {
        // \u{1F600} → 😀
        let k = kinds("\"\\u{1F600}\"");
        assert_eq!(k[0], TokenKind::Str("😀".into()));
    }

    #[test]
    fn test_unicode_escape_ascii() {
        // \u{41} → 'A'
        let k = kinds("\"\\u{41}\"");
        assert_eq!(k[0], TokenKind::Str("A".into()));
    }

    #[test]
    fn test_unicode_escape_in_interp() {
        // "\u{1F600} {name}!" — unicode escape followed by interpolation hole
        let k = kinds("\"\\u{1F600} {name}!\"");
        assert!(matches!(&k[0], TokenKind::StringInterp(parts) if {
            matches!(&parts[0], RawInterpPart::Lit(s) if s == "😀 ")
        }));
    }
}
