#!/usr/bin/env bash
# Runs the transpiler integration suite, then purges the regenerable test artifacts
# (tools/clean-build.sh; the shared target/transpile-cases alone is ~10GB) even when the run
# fails or is interrupted. Set BORING_TEST_KEEP=1 to keep the cache for faster re-runs.
# `--test-threads=1` is mandatory, see the header comment in tests/transpile.rs.
exec "$(dirname "$0")/test.sh" --test transpile "$@" -- --test-threads=1
