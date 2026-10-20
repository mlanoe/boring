#!/usr/bin/env python3
"""Merge several tests/cases/<name>.br fixtures into one scenario file.

Usage: merge_cases.py <out_name> <case> [<case> ...]     (run from tests/cases)

Each case's top-level `def`/`struct`/`enum`/`trait`/`view` names are prefixed with the case
name (CamelCase for types -- the parser tells types from values by capitalization), because
the transpiler flattens `mod`, so a module gives no namespace isolation. Each `main` becomes
`run_<case>`, and a generated `main` prints a `== <case> ==` header before calling each one.
`use` lines are hoisted and de-duplicated. The `.expected` file is the cases' expected
outputs, concatenated with the same headers.

Returns (via `merge()`) None for a case it cannot merge safely (unknown top-level shape);
callers verify the merged result by actually building and running it.
"""
import re
import sys

TYPE_TOKEN = r'[\w\[\]?&<>{}=,.\' ]+?'
DECL_KW = re.compile(r'^(?:pub\s+)?(?:struct|enum|trait|view)\s+(\w+)')
DECL_DEF = re.compile(r'^(?:pub\s+)?def\s+(?:' + TYPE_TOKEN + r'\s+)??(\w+)\s*[(<]')
DECL_TYPED = re.compile(r'^(?:pub\s+)?(?!def\b|let\b|var\b|use\b|mut\b)' + TYPE_TOKEN + r'\s+(\w+)\s*\(.*\)\s*(?:throws(?:\s+\w+)?\s*)?:\s*$')
MAIN = re.compile(r'^(pub\s+)?(def\s+)?(?:' + TYPE_TOKEN + r'\s+)??main\s*\(\s*\)(\s*throws)?\s*:')


def camel(n):
    return ''.join(w.capitalize() for w in n.split('_'))


def _blocks(lines):
    """Split into top-level blocks: (kind, name, lines) where kind is 'decl', 'main', 'use'
    or 'stmt'. A block is a top-level line plus the indented/blank lines that follow it;
    decorator and comment lines attach to the next block."""
    blocks, pending, i = [], [], 0
    while i < len(lines):
        l = lines[i]
        if not l.strip() or l.lstrip().startswith('#') or l.startswith('@'):
            pending.append(l); i += 1; continue
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or lines[j][0].isspace()):
            j += 1
        body = pending + lines[i:j]; pending = []
        if l.startswith('use '):
            kind, name = 'use', None
        elif MAIN.match(l):
            kind, name = 'main', None
        elif l.startswith('ext '):
            kind, name = 'decl', None
        else:
            m = DECL_KW.match(l) or DECL_DEF.match(l) or DECL_TYPED.match(l)
            kind, name = ('decl', m.group(1)) if m else ('stmt', None)
        blocks.append((kind, name, body)); i = j
    if pending:
        blocks.append(('tail', None, pending))
    return blocks


GLOBAL_DECL = re.compile(r'^(?:pub\s+)?(?:let|var|mut|static|const)\b')


def top_level_names(lines):
    """Names declared at top level. Raises ValueError for a shape that cannot be merged
    without changing what the case tests (globals used by functions, `main` mixed with
    top-level statements, no entry point at all)."""
    blocks = _blocks(lines)
    kinds = {k for k, _, _ in blocks}
    stmts = [b for k, _, b in blocks if k == 'stmt']
    if 'main' in kinds and stmts:
        raise ValueError('main mixed with top-level statements')
    if 'main' not in kinds and not stmts:
        raise ValueError('no entry point')
    gnames = set()
    for blk in stmts:
        first = next(x for x in blk if x.strip() and not x.lstrip().startswith('#') and not x.startswith('@'))
        m = GLOBAL_DECL.match(first) and re.match(
            r'^(?:pub\s+)?(?:let|var|mut|static|const)\s+(?:' + TYPE_TOKEN + r'\s+)??(\w+)\s*(?:=|$)', first)
        if m:
            gnames.add(m.group(1))
    decl_text = '\n'.join('\n'.join(b) for k, _, b in blocks if k == 'decl')
    for g in gnames:
        if re.search(r'(?<![\w.])' + re.escape(g) + r'\b', decl_text):
            raise ValueError('global used by a declaration: ' + g)
    return {n for k, n, _ in blocks if k == 'decl' and n}


def rewrite_case(n, text):
    lines = text.rstrip('\n').split('\n')
    tops = top_level_names(lines)
    blocks = _blocks(lines)
    uses = [next(x for x in b if x.startswith('use ')) for k, _, b in blocks if k == 'use']
    subs = []
    for x in sorted(tops, key=len, reverse=True):
        new = camel(n) + x if x[0].isupper() else f'{n}_{x}'
        subs.append((re.compile(r'(?<![\w.])' + re.escape(x) + r'\b'), new))

    def rename(ls):
        out = []
        for l in ls:
            if not l.lstrip().startswith('#'):
                for rx, rep in subs:
                    l = rx.sub(rep, l)
            out.append(l)
        return out

    out, stmt_lines = [], []
    for k, _, b in blocks:
        if k == 'use':
            continue
        if k == 'main':
            ls = list(b)
            idx = next(i for i, l in enumerate(ls) if MAIN.match(l))
            mm = MAIN.match(ls[idx])
            ls[idx] = f'def run_{n}(){mm.group(3) or ""}:' + ls[idx][mm.end():]
            out += rename(ls[:idx]) + [ls[idx]] + rename(ls[idx + 1:])
        elif k == 'decl':
            out += rename(b)
        elif k == 'stmt':
            stmt_lines += rename(b)
        else:
            out += b
    if stmt_lines:
        out += [f'def run_{n}() throws:'] + [('    ' + l if l.strip() else l) for l in stmt_lines]
    return uses, f'# ---- {n} ----\n' + '\n'.join(out)


def merge(out_name, names, read=lambda p: open(p).read()):
    uses, body, calls, exp = [], [], [], []
    throws = False
    for n in names:
        src = read(f'{n}.br')
        u, b = rewrite_case(n, src)
        for x in u:
            if x not in uses:
                uses.append(x)
        body.append(b)
        calls.append(f'    print "== {n} =="\n    run_{n}()')
        e = read(f'{n}.expected').rstrip('\n')
        exp.append(f'== {n} ==\n' + (e + '\n' if e else ''))
    head = ('\n'.join(uses) + '\n\n') if uses else ''
    br = head + '\n\n'.join(body) + '\n\ndef main() throws:\n' + '\n'.join(calls) + '\n'
    return br, ''.join(exp)


if __name__ == '__main__':
    out, cases = sys.argv[1], sys.argv[2:]
    br, exp = merge(out, cases)
    open(out + '.br', 'w').write(br)
    open(out + '.expected', 'w').write(exp)
