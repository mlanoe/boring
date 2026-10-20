#!/usr/bin/env bash
# Runs the transpiler integration suite and, when BORING_TEST_CLEAN=1, purges the shared
# build dir (target/transpile-cases, ~10GB) afterwards. Leave it unset to keep the cache
# (faster re-runs); CI relies on that cache.
set -u
cd "$(dirname "$0")/.."
cargo test --test transpile "$@" -- --test-threads=1
status=$?
if [ "${BORING_TEST_CLEAN:-0}" = "1" ]; then
  rm -rf target/transpile-cases
fi
exit $status
