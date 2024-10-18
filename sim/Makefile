OPTIMIZE ?= -O0
COMPILE = clang -std=c++20 -lstdc++ -Wall

main: skyjo.o skyjo.hpp
	$(COMPILE) main.cpp skyjo.o -o main

_test: skyjo.o skyjo.hpp
	$(COMPILE) test.cpp skyjo.o -o test
	
test: _test
	./test

skyjo.o: skyjo.cpp skyjo.hpp
	$(COMPILE) -c skyjo.cpp -o skyjo.o

clean:
	rm -f skyjo.o main test
