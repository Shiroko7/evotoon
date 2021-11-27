##################################################################
# Makefie for the generator of NK landscapes
#
# Author:      Martin Pelikan
# Last update: March 03, 2008
##################################################################

CC = gcc
CFLAGS = -Wall -ansi -g -ggdb
LIBS = -lm
SRC = main.c ga.c random.c nk.c
OBJ = main.o ga.o random.o nk.o


instance=nk_38_3.22
output=output.txt
cr=0.8
mr=0.1
ps=100
me=100000
seed=20
co=1
#1-uniform_crossover
#2-


ga-nk-wols: $(OBJ)
	$(CC) $(CFLAGS) -o ga-nk $(OBJ) $(LIBS)

main.o: main.c ga.h random.h
	$(CC) -c $(CFLAGS) main.c

nk.o: nk.c nk.h random.h
	$(CC) -c $(CFLAGS) nk.c

ga.o: ga.c ga.h random.h
	$(CC) -c $(CFLAGS) ga.c

random.o: random.c random.h
	$(CC) -c $(CFLAGS) random.c

clean:
	rm -f $(OBJ) ga-nk ga-nk.exe *~

exe:
	./ga-nk-wols $(instance) $(output) $(cr) $(mr) $(ps) $(me) $(seed) $(co)
