OPTIMIZE ?= -O0
COMPILE = clang -std=c++20 -lstdc++ -Wall

main: skyjo.o skyjo.hpp main.cpp
	$(COMPILE) main.cpp skyjo.o -o main

test: skyjo.o skyjo.hpp test.cpp
	$(COMPILE) test.cpp skyjo.o -o test

skyjo.o: skyjo.cpp skyjo.hpp
	$(COMPILE) -c skyjo.cpp -o skyjo.o

clean:
	rm -f skyjo.o main test
