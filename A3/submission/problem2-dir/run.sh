#!/bin/bash

SRC="problem2.cpp"          # Source file
EXE="problem2.out"         # Output binary
RUNS=7                        # Number of runs
OUTCSV="timings.csv"          # CSV output file

echo "Compiling with AVX2 + SSE4 + OMP + fast math flags..."
g++ "$SRC" -o "$EXE" \
    -std=c++17 \
    -masm=att \
    -msse4 \
    -mavx2 \
    -march=native \
    -fopenmp \
    -fverbose-asm \
    -fno-asynchronous-unwind-tables \
    -fno-exceptions \
    -fno-rtti \
    -fcf-protection=none \
    -O2

if [[ $? -ne 0 ]]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Run,Serial(us),OMP(us),SSE4(us),AVX2(us)" > "$OUTCSV"

for ((r=1; r<=RUNS; r++)); do
    echo "Run $r/$RUNS ..."
    OUTPUT=$(./"$EXE")

    # Extract timing results (microseconds) from output lines
    SERIAL=$(echo "$OUTPUT" | grep "Serial version:" | awk '{print $5}')
    OMP=$(echo "$OUTPUT" | grep "OMP version:" | awk '{print $5}')
    SSE=$(echo "$OUTPUT" | grep "SSE version:" | awk '{print $5}')
    AVX2=$(echo "$OUTPUT" | grep "AVX2 version:" | awk '{print $5}')

    # Handle potential missing values gracefully
    if [[ -z "$SERIAL" || -z "$OMP" || -z "$SSE" || -z "$AVX2" ]]; then
        echo "Missing timing data in run $r, skipping..."
        continue
    fi

    # Append row to CSV
    echo "$r,$SERIAL,$OMP,$SSE,$AVX2" >> "$OUTCSV"
done

echo "Benchmark complete. Results saved in $OUTCSV"