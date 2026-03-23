#!/usr/bin/env bash
set -euo pipefail

bench_path="./build/bench/bench_tsqr"
start_m=131072
end_m=196608
step_m=4096
refine_step_m=1024
iters=100
warmup=5

usage() {
  cat <<'EOF'
Usage: bench/find_static192_switch.sh [options]

Find the first m where Static192 TSQR becomes faster than Current TSQR.

Options:
  --bench PATH          Path to bench_tsqr executable (default: ./build/bench/bench_tsqr)
  --start M             Sweep start m (default: 131072)
  --end M               Sweep end m (default: 196608)
  --step M              Coarse sweep step (default: 4096)
  --refine-step M       Refinement step after first crossing (default: 1024)
  --iters N             Benchmark iterations (default: 100)
  --warmup N            Warmup iterations (default: 5)
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench)
      bench_path="$2"
      shift 2
      ;;
    --start)
      start_m="$2"
      shift 2
      ;;
    --end)
      end_m="$2"
      shift 2
      ;;
    --step)
      step_m="$2"
      shift 2
      ;;
    --refine-step)
      refine_step_m="$2"
      shift 2
      ;;
    --iters)
      iters="$2"
      shift 2
      ;;
    --warmup)
      warmup="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$bench_path" ]]; then
  echo "bench executable not found or not executable: $bench_path" >&2
  exit 1
fi

if (( start_m <= 0 || end_m < start_m || step_m <= 0 || refine_step_m <= 0 )); then
  echo "Invalid range/step arguments." >&2
  exit 1
fi

run_case() {
  local m="$1"
  local output
  output="$("$bench_path" --m "$m" --iters "$iters" --warmup "$warmup" --type double --double-variant both)"
  local current_ms
  local static_ms
  current_ms="$(printf '%s\n' "$output" | awk '/Current TSQR avg:/ {print $4; exit}')"
  static_ms="$(printf '%s\n' "$output" | awk '/Static192 TSQR avg:/ {print $4; exit}')"
  if [[ -z "$current_ms" || -z "$static_ms" ]]; then
    echo "Failed to parse benchmark output for m=$m" >&2
    printf '%s\n' "$output" >&2
    exit 1
  fi
  local delta_ms
  delta_ms="$(awk -v current="$current_ms" -v statik="$static_ms" 'BEGIN { printf "%.6f", statik - current }')"
  printf "%d %s %s %s\n" "$m" "$current_ms" "$static_ms" "$delta_ms"
}

echo "Searching switch point for Static192 vs Current"
echo "bench=$bench_path start=$start_m end=$end_m step=$step_m refine_step=$refine_step_m iters=$iters warmup=$warmup"
printf "%-10s %-12s %-12s %-12s %-10s\n" "m" "current_ms" "static_ms" "static-current" "winner"

prev_m=""
prev_delta=""
cross_lo=""
cross_hi=""

for (( m=start_m; m<=end_m; m+=step_m )); do
  read -r case_m current_ms static_ms delta_ms < <(run_case "$m")
  winner="$(awk -v delta="$delta_ms" 'BEGIN { print (delta < 0.0) ? "static192" : "current" }')"
  printf "%-10s %-12s %-12s %-12s %-10s\n" "$case_m" "$current_ms" "$static_ms" "$delta_ms" "$winner"
  if [[ -n "$prev_m" ]] && awk -v prev="$prev_delta" -v curr="$delta_ms" 'BEGIN { exit !((prev >= 0.0) && (curr < 0.0)) }'; then
    cross_lo="$prev_m"
    cross_hi="$case_m"
    break
  fi
  prev_m="$case_m"
  prev_delta="$delta_ms"
done

if [[ -z "$cross_lo" ]]; then
  echo
  echo "No crossing found in [$start_m, $end_m] with coarse step $step_m."
  exit 0
fi

echo
echo "Coarse crossing bracket: [$cross_lo, $cross_hi]"

best_m=""
best_current=""
best_static=""

for (( m=cross_lo; m<=cross_hi; m+=refine_step_m )); do
  read -r case_m current_ms static_ms delta_ms < <(run_case "$m")
  winner="$(awk -v delta="$delta_ms" 'BEGIN { print (delta < 0.0) ? "static192" : "current" }')"
  printf "%-10s %-12s %-12s %-12s %-10s\n" "$case_m" "$current_ms" "$static_ms" "$delta_ms" "$winner"
  if awk -v delta="$delta_ms" 'BEGIN { exit !(delta < 0.0) }'; then
    best_m="$case_m"
    best_current="$current_ms"
    best_static="$static_ms"
    break
  fi
done

echo
if [[ -n "$best_m" ]]; then
  echo "First refined crossing: m=$best_m current_ms=$best_current static192_ms=$best_static"
else
  echo "No refined crossing found in [$cross_lo, $cross_hi] with refine step $refine_step_m."
fi
