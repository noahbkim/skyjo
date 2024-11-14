COMPILE = clang -Wall -Iinclude

build:
	mkdir build

cards.o: src/cards.c build
	$(COMPILE) -c src/cards.c -o build/cards.o

hand.o: src/hand.c build
	$(COMPILE) -c src/hand.c -o build/hand.o

test.o: cards.o hand.o test/test_skyjo.c build
	$(COMPILE) test/test_skyjo.c build/cards.o build/hand.o -o build/test.o

test: test.o
	./build/test.o
