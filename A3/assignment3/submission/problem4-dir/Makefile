# Makefile for problem4-v*.c

CC = gcc
CFLAGS = -std=c17 -masm=att -msse4 -mavx2 -march=native -fopenmp -fverbose-asm \
         -fno-asynchronous-unwind-tables -fno-exceptions -fcf-protection=none -O3

TARGETS = problem4-v0.out problem4-v1.out problem4-v2.out
SOURCES = problem4-v0.c problem4-v1.c problem4-v2.c

all: $(TARGETS)

problem4-v0.out: problem4-v0.c
	$(CC) $(CFLAGS) $< -o $@

problem4-v1.out: problem4-v1.c
	$(CC) $(CFLAGS) $< -o $@

problem4-v2.out: problem4-v2.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
