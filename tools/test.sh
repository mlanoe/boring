#!/usr/bin/env bash
# Runs `cargo test` with the given arguments, then purges the regenerable test artifacts
# (tools/clean-build.sh) even when the run fails or is interrupted.
# Set BORING_TEST_KEEP=1 to keep them (faster re-runs; CI calls cargo directly and keeps its cache).
# Usage: tools/test.sh [cargo test args...]   e.g. tools/test.sh --test metal_codegen
set -u
cd "$(dirname "$0")/.."
cleanup() {
  [ "${BORING_TEST_KEEP:-0}" = "1" ] || tools/clean-build.sh || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
cargo test "$@"
