PROJECT = AntKnapsack
CFLAGS = -Wall -g -O3
OBJECTS = Main.o ACO.o Ant.o Initialize.o

all: $(PROJECT)

.cpp.o:
	g++ -c $(CFLAGS) $<

$(PROJECT): $(OBJECTS)
	g++ $(CFLAGS) -o $(PROJECT) $(OBJECTS)
	
Main.o: Main.cpp
	g++ $(CFLAGS) -c Main.cpp
	
ACO.o: ACO.cpp ACO.h
	g++ $(CFLAGS) -c ACO.cpp
	
Ant.o: Ant.cpp Ant.h
	g++ $(CFLAGS) -c Ant.cpp

Initialize.o: Initialize.cpp Initialize.h
	g++ $(CFLAGS) -c Initialize.cpp

clean:
	rm $(OBJECTS) $(PROJECT)
