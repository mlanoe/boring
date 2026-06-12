#!/usr/bin/env python3
"""Generate docs/book.html and README.html — no external dependencies."""

import re, sys, html
from pathlib import Path

DOCS = Path(__file__).parent
ROOT = DOCS.parent

# ── Syntax highlighter ────────────────────────────────────────────────────────

BORING_KW = {
    "let","mut","var","def","req","set","type","struct","enum","trait","ext","mod","use","pub",
    "return","if","elif","else","match","while","for","in","loop","do",
    "break","continue","guard","try","catch","throw","throws","defer","task",
    "stream","yield","channel",
    "nil","true","false","as","and","or","not","is","self","void","pass","init",
}
BORING_TYPES = {"int","uint","float","bool","string","str","Self"}


def highlight_boring(code: str) -> str:
    result = []
    i = 0
    n = len(code)
    while i < n:
        # Comment
        if code[i] == '#':
            end = code.find('\n', i)
            end = end if end != -1 else n
            result.append(f'<span class="co">{html.escape(code[i:end])}</span>')
            i = end
        # String literal
        elif code[i] == '"':
            j = i + 1
            s_chars = ['"']
            while j < n and code[j] != '"':
                if code[j] == '\\' and j + 1 < n:
                    s_chars.append(html.escape(code[j:j+2]))
                    j += 2
                elif code[j] == '{':
                    # interpolation hole
                    depth = 1
                    s_chars.append('<span class="si">{')
                    j += 1
                    while j < n and depth > 0:
                        if code[j] == '{': depth += 1
                        elif code[j] == '}': depth -= 1
                        if depth > 0:
                            s_chars.append(html.escape(code[j]))
                        j += 1
                    s_chars.append('}</span>')
                else:
                    s_chars.append(html.escape(code[j]))
                    j += 1
            s_chars.append('"')
            if j < n: j += 1
            result.append(f'<span class="s">{"".join(s_chars)}</span>')
            i = j
        # Number
        elif code[i].isdigit() or (code[i] == '-' and i + 1 < n and code[i+1].isdigit()
                                    and (i == 0 or not code[i-1].isalnum())):
            j = i
            if code[j] == '-': j += 1
            while j < n and (code[j].isdigit() or code[j] == '.'):
                j += 1
            result.append(f'<span class="nb">{html.escape(code[i:j])}</span>')
            i = j
        # Identifier / keyword / type
        elif code[i].isalpha() or code[i] == '_':
            j = i
            while j < n and (code[j].isalnum() or code[j] == '_'):
                j += 1
            word = code[i:j]
            # macro call: word followed by !
            if j < n and code[j] == '!':
                result.append(f'<span class="fm">{html.escape(word)}!</span>')
                i = j + 1
            elif word in BORING_KW:
                result.append(f'<span class="kw">{html.escape(word)}</span>')
                i = j
            elif word in BORING_TYPES:
                result.append(f'<span class="kt">{html.escape(word)}</span>')
                i = j
            elif j < n and code[j] == '(':
                result.append(f'<span class="fn">{html.escape(word)}</span>')
                i = j
            else:
                result.append(html.escape(word))
                i = j
        # Attribute @derive(...)
        elif code[i] == '@':
            j = i + 1
            while j < n and (code[j].isalnum() or code[j] == '_'):
                j += 1
            result.append(f'<span class="at">{html.escape(code[i:j])}</span>')
            i = j
        # Ownership qualifier tick
        elif code[i] == "'" and i + 1 < n and (code[i+1].isalpha() or code[i+1] in ("'", "=")):
            j = i + 1
            while j < n and (code[j].isalnum() or code[j] == '_'):
                j += 1
            result.append(f'<span class="lf">{html.escape(code[i:j])}</span>')
            i = j
        else:
            result.append(html.escape(code[i]))
            i += 1
    return "".join(result)


def highlight_rust(code: str) -> str:
    RUST_KW = {
        "fn","let","mut","pub","struct","enum","impl","trait","use","mod","type",
        "match","if","else","while","for","loop","return","break","continue",
        "in","as","where","self","Self","super","crate","async","await","move",
        "ref","const","static","unsafe","extern","dyn","Box","Vec","Option",
        "Some","None","Ok","Err","Result","Arc","Rc","Weak","HashMap","HashSet",
        "true","false",
    }
    result = []
    i = 0
    n = len(code)
    while i < n:
        # Line comment
        if i + 1 < n and code[i:i+2] == '//':
            end = code.find('\n', i)
            end = end if end != -1 else n
            result.append(f'<span class="co">{html.escape(code[i:end])}</span>')
            i = end
        # String
        elif code[i] == '"':
            j = i + 1
            while j < n and code[j] != '"':
                if code[j] == '\\': j += 1
                j += 1
            j += 1
            result.append(f'<span class="s">{html.escape(code[i:j])}</span>')
            i = j
        # Lifetime 'a
        elif code[i] == "'" and i + 1 < n and code[i+1].isalpha():
            j = i + 1
            while j < n and code[j].isalnum(): j += 1
            result.append(f'<span class="lf">{html.escape(code[i:j])}</span>')
            i = j
        # Number
        elif code[i].isdigit():
            j = i
            while j < n and (code[j].isdigit() or code[j] in '.iu_f'): j += 1
            result.append(f'<span class="nb">{html.escape(code[i:j])}</span>')
            i = j
        # Attribute #[...]
        elif code[i] == '#' and i + 1 < n and code[i+1] == '[':
            j = code.find(']', i)
            j = j + 1 if j != -1 else n
            result.append(f'<span class="at">{html.escape(code[i:j])}</span>')
            i = j
        # Identifier
        elif code[i].isalpha() or code[i] == '_':
            j = i
            while j < n and (code[j].isalnum() or code[j] == '_'): j += 1
            word = code[i:j]
            if word in RUST_KW:
                result.append(f'<span class="kw">{html.escape(word)}</span>')
            elif j < n and code[j] == '!':
                result.append(f'<span class="fm">{html.escape(word)}!</span>')
                j += 1
            elif j < n and code[j] == '(':
                result.append(f'<span class="fn">{html.escape(word)}</span>')
            elif word[0].isupper():
                result.append(f'<span class="kt">{html.escape(word)}</span>')
            else:
                result.append(html.escape(word))
            i = j
        else:
            result.append(html.escape(code[i]))
            i += 1
    return "".join(result)


def highlight_sh(code: str) -> str:
    result = []
    for line in code.splitlines():
        if line.startswith('#'):
            result.append(f'<span class="co">{html.escape(line)}</span>')
        else:
            result.append(html.escape(line))
    return "\n".join(result)


def highlight(code: str, lang: str) -> str:
    if lang == "boring": return highlight_boring(code)
    if lang == "rust":   return highlight_rust(code)
    if lang in ("sh", "bash", "shell"): return highlight_sh(code)
    return html.escape(code)


# ── Markdown → HTML ───────────────────────────────────────────────────────────

def inline(text: str) -> str:
    """Convert inline markdown to HTML — single-pass so bold can wrap code spans."""
    result = []
    i = 0
    n = len(text)
    while i < n:
        # Bold: **...** — inner content may contain `code` spans
        if text[i:i+2] == '**':
            end = text.find('**', i + 2)
            if end != -1:
                inner = text[i+2:end]
                result.append(f'<strong>{inline(inner)}</strong>')
                i = end + 2
                continue
        # Code span: `...`
        if text[i] == '`':
            end = text.find('`', i + 1)
            if end != -1:
                result.append(f'<code>{html.escape(text[i+1:end])}</code>')
                i = end + 1
                continue
        # Italic: *...*
        if text[i] == '*':
            end = text.find('*', i + 1)
            if end != -1:
                result.append(f'<em>{inline(text[i+1:end])}</em>')
                i = end + 1
                continue
        # Link: [text](url)
        if text[i] == '[':
            m = re.match(r'\[([^\]]+)\]\(([^)]+)\)', text[i:])
            if m:
                result.append(f'<a href="{html.escape(m.group(2))}">{html.escape(m.group(1))}</a>')
                i += len(m.group(0))
                continue
        # Regular character
        result.append(html.escape(text[i]))
        i += 1
    return ''.join(result)


def parse_table(lines: list) -> str:
    ESCAPED_PIPE = '\x00PIPE\x00'
    rows = []
    for i, line in enumerate(lines):
        if re.match(r'^\s*\|[-:| ]+\|\s*$', line):
            continue
        # Protect escaped pipes \| and | inside backtick code spans before splitting
        safe = line.replace(r'\|', ESCAPED_PIPE)
        safe = re.sub(r'`[^`]*`', lambda m: m.group(0).replace('|', ESCAPED_PIPE), safe)
        cells = [c.strip().replace(ESCAPED_PIPE, '|') for c in safe.strip().strip('|').split('|')]
        tag = 'th' if i == 0 else 'td'
        row = "".join(f'<{tag}>{inline(c)}</{tag}>' for c in cells)
        rows.append(f'<tr>{row}</tr>')
    return '<table><thead>' + rows[0] + '</thead><tbody>' + "".join(rows[1:]) + '</tbody></table>'


def convert(md: str) -> str:
    lines = md.splitlines()
    html_parts = []
    i = 0
    toc_ids: dict = {}

    def slug(text: str) -> str:
        # GFM-compatible anchor: remove non-(word|space|hyphen) chars,
        # then replace each space individually with '-' (no collapsing).
        # This preserves double-dashes produced by em-dash + surrounding spaces:
        #   "Advanced — Strings" → "advanced  strings" → "advanced--strings"
        s = re.sub(r'[^\w\s-]', '', text.lower())
        s = s.strip().replace(' ', '-')
        toc_ids[s] = toc_ids.get(s, 0) + 1
        return s if toc_ids[s] == 1 else f"{s}-{toc_ids[s]-1}"

    while i < len(lines):
        line = lines[i]

        # Fenced code block
        m = re.match(r'^```(\w*)$', line)
        if m:
            lang = m.group(1).lower()
            code_lines = []
            i += 1
            while i < len(lines) and lines[i] != '```':
                code_lines.append(lines[i])
                i += 1
            code = "\n".join(code_lines)
            highlighted = highlight(code, lang)
            lang_label = f'<span class="lang-label">{lang}</span>' if lang else ''
            data = f' data-lang="{lang}"' if lang else ''
            html_parts.append(f'<div class="code-block"{data}>{lang_label}<pre><code>{highlighted}</code></pre></div>')
            i += 1
            continue

        # Horizontal rule
        if re.match(r'^---+$', line.strip()):
            html_parts.append('<hr>')
            i += 1
            continue

        # Blockquote
        if line.startswith('>'):
            bq_lines = []
            while i < len(lines) and lines[i].startswith('>'):
                bq_lines.append(lines[i][1:].lstrip())
                i += 1
            html_parts.append(f'<blockquote>{inline(" ".join(bq_lines))}</blockquote>')
            continue

        # Headings
        m = re.match(r'^(#{1,4})\s+(.+)$', line)
        if m:
            level = len(m.group(1))
            text  = m.group(2)
            sid   = slug(re.sub(r'[`*]', '', text))
            html_parts.append(f'<h{level} id="{sid}">{inline(text)}</h{level}>')
            i += 1
            continue

        # Table
        if '|' in line and i + 1 < len(lines) and re.match(r'^\s*\|[-:| ]+\|\s*$', lines[i+1]):
            table_lines = [line]
            i += 1
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            html_parts.append(parse_table(table_lines))
            continue

        # Ordered list — also handles indented `    - sub` lines as nested <ul>
        if re.match(r'^\d+\. ', line):
            items = []
            while i < len(lines):
                cur = lines[i]
                if re.match(r'^\d+\. ', cur):
                    text = re.sub(r'^\d+\. ', '', cur)
                    items.append(f'<li>{inline(text)}')
                    i += 1
                    # collect indented sub-items belonging to this <li>
                    sub_items = []
                    while i < len(lines) and re.match(r'^    [-*] ', lines[i]):
                        sub_text = re.sub(r'^    [-*] ', '', lines[i])
                        sub_items.append(f'<li>{inline(sub_text)}</li>')
                        i += 1
                    if sub_items:
                        items.append('<ul>' + ''.join(sub_items) + '</ul>')
                    items.append('</li>')
                else:
                    break
            html_parts.append('<ol>' + "".join(items) + '</ol>')
            continue

        # Unordered list
        if re.match(r'^[-*] ', line):
            items = []
            while i < len(lines) and re.match(r'^[-*] ', lines[i]):
                items.append(f'<li>{inline(lines[i][2:])}</li>')
                i += 1
            html_parts.append('<ul>' + "".join(items) + '</ul>')
            continue

        # Blank line
        if not line.strip():
            i += 1
            continue

        # Paragraph
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith('#') \
              and not lines[i].startswith('>') and not lines[i].startswith('-') \
              and not lines[i].startswith('```') and '|' not in lines[i]:
            para.append(lines[i])
            i += 1
        html_parts.append(f'<p>{inline(" ".join(para))}</p>')

    return "\n".join(html_parts)


# ── CSS & template ────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 16px; line-height: 1.7; color: #1a1a2e;
  background: #f8f9fa;
}
.sidebar {
  position: fixed; top: 0; left: 0; width: 270px; height: 100vh;
  background: #1a1a2e; color: #c9d1d9; overflow-y: auto;
  padding: 0 0 2rem 0; z-index: 100;
}
.sidebar-header {
  background: #0f3460; padding: 1.2rem 1.5rem; margin-bottom: 0.5rem;
}
.sidebar-header h2 { font-size: 1rem; color: #e94560; font-weight: 700; letter-spacing: 0.03em; }
.sidebar-header p  { font-size: 0.75rem; color: #8b949e; margin-top: 0.2rem; }
.sidebar nav a {
  display: block; padding: 0.3rem 1.5rem; color: #8b949e;
  text-decoration: none; font-size: 0.875rem; transition: all 0.15s;
}
.sidebar nav a:hover { color: #e6edf3; background: rgba(255,255,255,0.05); }
.sidebar nav a.h2 { font-weight: 600; color: #c9d1d9; margin-top: 0.4rem; }
.sidebar nav a.h3 { padding-left: 2.5rem; font-size: 0.825rem; }
.main { margin-left: 270px; padding: 3rem 4rem; max-width: 980px; }
h1 { font-size: 2.2rem; font-weight: 800; color: #0f3460; margin: 2rem 0 1rem;
     border-bottom: 3px solid #e94560; padding-bottom: 0.5rem; }
h2 { font-size: 1.6rem; font-weight: 700; color: #16213e; margin: 2.5rem 0 1rem;
     border-bottom: 1px solid #dee2e6; padding-bottom: 0.4rem; }
h3 { font-size: 1.15rem; font-weight: 600; color: #0f3460; margin: 1.8rem 0 0.6rem; }
h4 { font-size: 1rem;    font-weight: 600; color: #495057; margin: 1.2rem 0 0.4rem; }
p  { margin-bottom: 1rem; }
a  { color: #e94560; text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: none; border-top: 1px solid #dee2e6; margin: 2rem 0; }
strong { color: #16213e; }
code {
  font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", Consolas, monospace;
  font-size: 0.875em; background: #e8ecf0; color: #c7254e;
  padding: 0.15em 0.4em; border-radius: 4px;
}
.code-block { position: relative; margin: 1.2rem 0; }
.code-block pre {
  background: #1e2030; border-radius: 8px; padding: 1.2rem 1.5rem;
  overflow-x: auto; line-height: 1.6;
  border-left: 3px solid transparent;
}
.code-block[data-lang="boring"] pre {
  background: #1e2030;
  border-left: 3px solid #89b4fa;
}
.code-block[data-lang="rust"] pre {
  background: #d4d8e0;
  border-left: 3px solid #b7410e;
}
.code-block pre code {
  font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
  font-size: 0.875rem; background: none; color: #cdd6f4; padding: 0;
  border-radius: 0;
}
.code-block[data-lang="rust"] pre code { color: #2d2d3a; }
.lang-label {
  position: absolute; top: 0; right: 0;
  background: #313244; color: #7f849c; font-size: 0.7rem;
  padding: 0.2rem 0.6rem; border-radius: 0 8px 0 6px;
  font-family: "JetBrains Mono", monospace; text-transform: uppercase;
  letter-spacing: 0.06em;
}
.code-block[data-lang="boring"] .lang-label { background: #1e3a5f; color: #89b4fa; }
.code-block[data-lang="rust"]   .lang-label { background: #e8e0f0; color: #7c3aed; }
/* Rust light-theme syntax overrides */
.code-block[data-lang="rust"] .kw { color: #7c3aed; }
.code-block[data-lang="rust"] .kt { color: #0e7490; }
.code-block[data-lang="rust"] .fn { color: #1d4ed8; }
.code-block[data-lang="rust"] .fm { color: #c2410c; }
.code-block[data-lang="rust"] .s  { color: #16a34a; }
.code-block[data-lang="rust"] .si { color: #92400e; }
.code-block[data-lang="rust"] .nb { color: #c2410c; }
.code-block[data-lang="rust"] .co { color: #9ca3af; }
.code-block[data-lang="rust"] .at { color: #dc2626; }
.code-block[data-lang="rust"] .lf { color: #9d174d; }
/* Syntax colors (Catppuccin-inspired) */
.kw { color: #cba6f7; font-weight: 600; }  /* keywords */
.kt { color: #89dceb; }                     /* types */
.fn { color: #89b4fa; }                     /* function names */
.fm { color: #fab387; }                     /* macros */
.s  { color: #a6e3a1; }                     /* strings */
.si { color: #f9e2af; font-weight: 600; }   /* string interpolation */
.nb { color: #fab387; }                     /* numbers */
.co { color: #6c7086; font-style: italic; } /* comments */
.at { color: #f38ba8; }                     /* attributes */
.lf { color: #f2cdcd; }                     /* lifetime/qualifier */
/* Tables */
table { border-collapse: collapse; width: 100%; margin: 1.2rem 0; font-size: 0.9rem; }
th { background: #0f3460; color: #fff; padding: 0.6rem 1rem; text-align: left; }
td { padding: 0.5rem 1rem; border-bottom: 1px solid #dee2e6; }
tr:nth-child(even) td { background: #f1f3f5; }
tr:hover td { background: #e8f4fd; }
/* Blockquote (preview banner) */
blockquote {
  background: #fff3cd; border-left: 4px solid #e94560;
  padding: 1rem 1.5rem; border-radius: 0 8px 8px 0;
  margin: 1.5rem 0; color: #664d03; font-size: 0.95rem;
}
ul, ol { padding-left: 1.5rem; margin-bottom: 1rem; }
li { margin-bottom: 0.3rem; }
@media (max-width: 900px) {
  .sidebar { display: none; }
  .main { margin-left: 0; padding: 1.5rem; }
}
"""

TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<aside class="sidebar">
  <div class="sidebar-header">
    <h2>{sidebar_title}</h2>
    <p>{sidebar_sub}</p>
  </div>
  <nav id="toc"></nav>
</aside>
<main class="main" id="content">
{body}
</main>
<script>
// Build TOC from headings
const toc = document.getElementById('toc');
document.querySelectorAll('h2,h3').forEach(h => {{
  const a = document.createElement('a');
  a.href = '#' + h.id;
  a.textContent = h.textContent;
  a.className = h.tagName.toLowerCase();
  toc.appendChild(a);
}});
// Highlight active section on scroll
const headings = [...document.querySelectorAll('h2,h3')];
const links = [...toc.querySelectorAll('a')];
window.addEventListener('scroll', () => {{
  let current = headings[0];
  headings.forEach(h => {{ if (window.scrollY >= h.offsetTop - 80) current = h; }});
  links.forEach(a => a.style.color = a.getAttribute('href') === '#' + current?.id ? '#e94560' : '');
}});
</script>
</body>
</html>
"""

# ── Main ──────────────────────────────────────────────────────────────────────

def build_file(src: Path, dest: Path, title: str,
               sidebar_title: str = "Boring Language",
               sidebar_sub: str = "Preview · Beta") -> None:
    md   = src.read_text(encoding="utf-8")
    body = convert(md)
    out  = TEMPLATE.format(css=CSS, body=body,
                           title=title,
                           sidebar_title=sidebar_title,
                           sidebar_sub=sidebar_sub)
    dest.write_text(out, encoding="utf-8")
    print(f"Generated {dest}  ({len(out)//1024} KB)")


def main():
    build_file(
        src=DOCS / "book.md",
        dest=DOCS / "book.html",
        title="The Boring Programming Language — Language Book",
        sidebar_title="Boring Language",
        sidebar_sub="Language Book · Beta",
    )
    build_file(
        src=ROOT / "README.md",
        dest=ROOT / "README.html",
        title="The new programming language is Boring",
        sidebar_title="Boring",
        sidebar_sub="Overview",
    )


if __name__ == "__main__":
    main()
