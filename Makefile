# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c
	$(CC) -O3 -o run run.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: run.c
	$(CC) -g -o run run.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
# (no -Ofast here: fast-math may change the int8 rounding that run.c must match)
.PHONY: runomp
runomp: run.c
	$(CC) -O3 -fopenmp -march=native run.c -lm -o run

# run all tests: run.c against model.py
.PHONY: test
test:
	pytest

.PHONY: clean
clean:
	rm -f run test
