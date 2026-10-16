#!/usr/bin/env python3
"""Resident Metal A/B for the native Q8_0 decode schedules.

Builds isolated `rows` and `split-k` projects from the checked-in benchmark,
then runs them in alternating order. The repository's boring.toml is never
modified. By default the script refuses to time a busy machine.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "tensor_q8_decode_bench.br"
LINE = re.compile(
    r"q8_0 n=(?P<n>\d+) k=(?P<k>\d+): (?P<us>[0-9.]+) us/call, "
    r"(?P<gbs>[0-9.]+) GB/s, sum=(?P<sum>[-0-9.]+) "
    r"weighted=(?P<weighted>[-0-9.]+) first=(?P<first>[-+0-9.eE]+) "
    r"last=(?P<last>[-+0-9.eE]+)"
)
SCHEDULES = ("rows", "split-k")


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return completed.stdout


def build(schedule: str, compiler: Path, workspace: Path) -> Path:
    project = workspace / schedule
    project.mkdir()
    shutil.copy2(SOURCE, project / "main.br")
    (project / "boring.toml").write_text(
        "[project]\n"
        f'name = "q8-decode-{schedule}"\n'
        'version = "0.1.0"\n\n'
        "[tensor.linear.decode]\n"
        f'q8_0 = "{schedule}"\n\n'
        "[tensor.linear.prefill]\n"
        'q8_0 = "native"\n',
        encoding="utf-8",
    )
    print(run([str(compiler), "build", "--target", "metal", "main.br"], project), end="")
    generated = project / "main_metal"
    print(run(["cargo", "build", "--release"], generated), end="")
    return generated / "target" / "release" / "main_metal"


def measure(binary: Path, calls: int) -> dict[tuple[int, int], dict[str, float]]:
    env = os.environ.copy()
    env["BORING_METAL_BATCH"] = "1"
    output = run([str(binary), str(calls)], binary.parent, env)
    print(output, end="")
    result: dict[tuple[int, int], dict[str, float]] = {}
    for match in LINE.finditer(output):
        values = {key: float(match.group(key)) for key in ("us", "gbs", "sum", "weighted", "first", "last")}
        result[(int(match.group("n")), int(match.group("k")))] = values
    if len(result) != 6:
        raise RuntimeError(f"expected six benchmark rows, parsed {len(result)}")
    return result


def close_enough(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=2e-4, abs_tol=2e-4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=ROOT / "target" / "release" / "boring")
    parser.add_argument("--calls", type=int, default=500)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-load", type=float, default=2.0, help="maximum accepted one-minute load average")
    parser.add_argument("--force", action="store_true", help="run even when the load check fails")
    args = parser.parse_args()

    if args.calls < 1 or args.repetitions < 1:
        parser.error("--calls and --repetitions must be positive")
    load = os.getloadavg()[0]
    if load > args.max_load and not args.force:
        parser.error(f"one-minute load is {load:.2f}, above {args.max_load:.2f}; retry on a quiet machine or pass --force")
    compiler = args.compiler.resolve()
    if not compiler.is_file():
        parser.error(f"compiler not found: {compiler}; run `cargo build --release` first")

    samples: dict[str, dict[tuple[int, int], list[dict[str, float]]]] = {
        schedule: {} for schedule in SCHEDULES
    }
    with tempfile.TemporaryDirectory(prefix="boring-q8-ab-") as temporary:
        workspace = Path(temporary)
        binaries = {schedule: build(schedule, compiler, workspace) for schedule in SCHEDULES}
        for repetition in range(args.repetitions):
            order = SCHEDULES if repetition % 2 == 0 else tuple(reversed(SCHEDULES))
            print(f"\nrepetition {repetition + 1}/{args.repetitions}: {' then '.join(order)}")
            for schedule in order:
                measured = measure(binaries[schedule], args.calls)
                for shape, values in measured.items():
                    samples[schedule].setdefault(shape, []).append(values)

    print("\nshape          rows us (min-max)     split-k us (min-max)  split/rows")
    for shape in sorted(samples["rows"]):
        row_times = [sample["us"] for sample in samples["rows"][shape]]
        split_times = [sample["us"] for sample in samples["split-k"][shape]]
        row_median = statistics.median(row_times)
        split_median = statistics.median(split_times)
        print(
            f"{shape[0]:6d}x{shape[1]:-6d}  "
            f"{row_median:7.2f} ({min(row_times):.2f}-{max(row_times):.2f})  "
            f"{split_median:9.2f} ({min(split_times):.2f}-{max(split_times):.2f})  "
            f"{split_median / row_median:8.3f}x"
        )
        reference = samples["rows"][shape][0]
        for sample in samples["split-k"][shape]:
            for field in ("sum", "weighted", "first", "last"):
                if not close_enough(reference[field], sample[field]):
                    raise RuntimeError(
                        f"numeric mismatch for {shape} {field}: rows={reference[field]} split-k={sample[field]}"
                    )


if __name__ == "__main__":
    main()
