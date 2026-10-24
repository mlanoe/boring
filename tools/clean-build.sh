#!/bin/sh
# Remove regenerable build artifacts (test scratch dirs, incremental cache, generated
# projects' target/ dirs). Safe to run any time no cargo process is running.
# Usage: tools/clean-build.sh
set -e
cd "$(dirname "$0")/.."
if pgrep -x cargo >/dev/null; then echo "cargo is running, aborting" >&2; exit 1; fi
rm -rf target/tmp target/transpile-cases target/debug/incremental \
       boring/interpreter/*_rust*/target examples/*/target
find tests -type d -name target -path '*_rust*' -prune -exec rm -rf {} + 2>/dev/null || true
du -sh target
