#!/usr/bin/env bash
# Benchmark: native Rust interpreter vs boring-written interpreter (multi-thread release)

NATIVE="./target/release/boring"
BORING_INTERP="./boring/interpreter/main_rust/target/release/main"
CASES_DIR="./tests/cases"

if [[ ! -x "$NATIVE" ]]; then
  echo "Building native interpreter (release)..." >&2
  cargo build --release -q
fi

if [[ ! -x "$BORING_INTERP" ]]; then
  echo "Building boring-written interpreter (release)..." >&2
  (cd boring/interpreter/main_rust && cargo build --release -q)
fi

native_total=0
boring_total=0
pass=0
fail=0
skipped=0

ms() { python3 -c "import time; print(int(time.time()*1000))"; }

printf "%-40s  %8s  %8s  %6s\n" "test" "native(ms)" "boring(ms)" "ratio"
printf "%-40s  %8s  %8s  %6s\n" "----" "----------" "----------" "-----"

for br in $(find "$CASES_DIR" -name "*.br" | sort); do
  name=$(basename "$br" .br)
  expected="$CASES_DIR/$name.expected"

  if [[ ! -f "$expected" ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  t0=$(ms)
  native_out=$("$NATIVE" "$br" 2>/dev/null)
  native_rc=$?
  t1=$(ms)
  native_ms=$((t1 - t0))

  if [[ $native_rc -ne 0 ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  t0=$(ms)
  boring_out=$(cat "$br" | "$BORING_INTERP" 2>/dev/null)
  boring_rc=$?
  t1=$(ms)
  boring_ms=$((t1 - t0))

  if [[ $boring_rc -ne 0 ]]; then
    printf "%-40s  %8d  BORING CRASHED\n" "$name" "$native_ms"
    fail=$((fail + 1))
    continue
  fi

  exp=$(cat "$expected")
  native_norm=$(printf '%s' "$native_out" | tr -d '\r')
  boring_norm=$(printf '%s' "$boring_out" | tr -d '\r')
  exp_norm=$(printf '%s' "$exp" | tr -d '\r')

  if [[ "$native_norm" != "$exp_norm" ]]; then
    printf "%-40s  NATIVE MISMATCH\n" "$name"
    fail=$((fail + 1))
    continue
  fi

  if [[ "$boring_norm" != "$exp_norm" ]]; then
    printf "%-40s  %8d  BORING MISMATCH\n" "$name" "$native_ms"
    fail=$((fail + 1))
    continue
  fi

  ratio=$(python3 -c "n=$native_ms; b=$boring_ms; print(f'{b/n:.2f}' if n>0 else '?')" 2>/dev/null || echo "?")

  printf "%-40s  %8d  %8d  %6sx\n" "$name" "$native_ms" "$boring_ms" "$ratio"

  native_total=$((native_total + native_ms))
  boring_total=$((boring_total + boring_ms))
  pass=$((pass + 1))
done

echo ""
echo "────────────────────────────────────────────────────────────────"
printf "%-40s  %8d  %8d\n" "TOTAL ($pass tests)" "$native_total" "$boring_total"
overall=$(python3 -c "print(f'{$boring_total/$native_total:.2f}x' if $native_total>0 else '?')" 2>/dev/null || echo "?")
echo "Overall ratio: $overall  (boring-written / native)"
echo "Skipped: $skipped  |  Output mismatches / crashes: $fail"
