#!/bin/bash
# === Configuration ===
SRC="problem1.cpp"          # your source file name
EXE="problem1.out"
THREADS=(1 2 4 8 12)           # adjust for your machine
REPS=5                         # number of repetitions per config
CSV="stencil_results.csv"

# === Build ===
echo "Compiling..."
g++ -O3 -march=native -fopenmp -std=c++17 -DNDEBUG $SRC -o $EXE || exit 1
echo "Running on $(hostname)"
echo "Threads: ${THREADS[*]}  Reps: $REPS"
echo

# === Extract kernel names ===
KERNELS=($(grep -E 'test_kernel\(stencil_' $SRC | sed -E 's/.*"([^"]+)".*/\1/'))
NK=${#KERNELS[@]}
echo "Detected ${NK} kernels"
echo "Kernel names: ${KERNELS[*]}"
echo

# === CSV header ===
echo -n "Kernel" > $CSV
for th in "${THREADS[@]}"; do echo -n ",${th}T" >> $CSV; done
echo >> $CSV

# === Main loop ===
for kname in "${KERNELS[@]}"; do
  echo -n "$kname"
  printf "\n=== %-30s ===\n" "$kname"
  for th in "${THREADS[@]}"; do
    echo "  Threads=$th"
    total=0
    for ((r=1;r<=REPS;r++)); do
      out=$(OMP_NUM_THREADS=$th ./$EXE 2>/dev/null | grep "$kname time")
      time=$(echo "$out" | grep -oE '[0-9]+' | tail -1)
      echo "    run $r: ${time} ms"
      total=$((total + time))
    done
    avg=$(awk "BEGIN {printf \"%.2f\", $total / $REPS}")
    echo -n ",$avg" >> $CSV
    echo "  -> avg $avg ms"
  done
  echo >> $CSV
done

echo
echo "Results saved to $CSV"