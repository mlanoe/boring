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
DECL_DEF = re.compile(r'^(?:pub\s+)?def\s+(?:' + TYPE_TOKEN + r'\s+)??(\w+)\s*\(')
DECL_TYPED = re.compile(r'^(?:pub\s+)?(?!def\b|let\b|var\b|use\b|mut\b)' + TYPE_TOKEN + r'\s+(\w+)\s*\(.*\)\s*(?:throws\s*)?:\s*$')
MAIN = re.compile(r'^(pub\s+)?(def\s+)?(?:' + TYPE_TOKEN + r'\s+)??main\s*\(\s*\)(\s*throws)?\s*:')


def camel(n):
    return ''.join(w.capitalize() for w in n.split('_'))


def top_level_names(lines):
    """Names declared at top level; raises ValueError on an unrecognised top-level line."""
    names, has_main = set(), False
    for l in lines:
        if not l.strip() or l[0].isspace() or l.lstrip().startswith('#') or l.startswith('@'):
            continue
        if l.startswith('use '):
            continue
        if MAIN.match(l):
            has_main = True
            continue
        m = DECL_KW.match(l) or DECL_DEF.match(l) or DECL_TYPED.match(l)
        if not m:
            raise ValueError('unrecognised top-level line: ' + l[:40])
        names.add(m.group(1))
    if not has_main:
        raise ValueError('no main')
    return names


def rewrite_case(n, text):
    lines = text.rstrip('\n').split('\n')
    uses = [l for l in lines if l.startswith('use ')]
    lines = [l for l in lines if not l.startswith('use ')]
    tops = top_level_names(lines)
    subs = []
    for x in sorted(tops, key=len, reverse=True):
        new = camel(n) + x if x[0].isupper() else f'{n}_{x}'
        subs.append((re.compile(r'(?<![\w.])' + re.escape(x) + r'\b'), new))
    out = []
    for l in lines:
        mm = MAIN.match(l)
        if mm:
            rest = l[mm.end():]
            l = f'def run_{n}(){mm.group(3) or ""}:' + rest
        elif not l.lstrip().startswith('#'):
            for rx, rep in subs:
                l = rx.sub(rep, l)
        out.append(l)
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
