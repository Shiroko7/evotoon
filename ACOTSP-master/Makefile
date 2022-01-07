# Makefile for ACOTSP
VERSION=1.03

OPTIM_FLAGS=-O
WARN_FLAGS=-Wall -ansi -pedantic -Wno-overlength-strings -Wno-unused-result
CFLAGS=$(WARN_FLAGS) $(OPTIM_FLAGS)

# You can use this autodetection of the platform ...
ifeq ($(OS),Windows_NT)
    detected_OS=Windows
else
    detected_OS=$(shell uname -s)
endif
ifeq ($(detected_OS),Windows)
    TIMER=dos
else
    TIMER=unix
endif
# ... or force the timer implementation by uncommenting uncomment the line below ...
#TIMER=dos
#TIMER=unix
# ... or call 'make TIMER=unix'

LDLIBS=-lm

acotsp: acotsp.o TSP.o utilities.o ants.o InOut.o $(TIMER)_timer.o ls.o parse.o

all: clean acotsp

clean:
	@$(RM) *.o acotsp

acotsp: acotsp.o TSP.o utilities.o ants.o InOut.o $(TIMER)_timer.o ls.o parse.o

acotsp.o: acotsp.c

TSP.o: TSP.c TSP.h

ants.o: ants.c ants.h

InOut.o: InOut.c InOut.h

utilities.o: utilities.c utilities.h

ls.o: ls.c ls.h

parse.o: parse.c parse.h

$(TIMER)_timer.o: $(TIMER)_timer.c timer.h

dist : DIST_SRC_FILES=*.c *.h README *.tsp Makefile gpl.txt
dist : all
	@(mkdir -p ../ACOTSP-$(VERSION)			\
	&& rsync -rlpC --exclude=.svn $(DIST_SRC_FILES) ../ACOTSP-$(VERSION)/ \
        && cd .. 	\
	&& tar cf - ACOTSP-$(VERSION) | gzip -f9 > ACOTSP-$(VERSION).tar.gz \
	&& rm -rf ./ACOTSP-$(VERSION)					\
	&& echo "ACOTSP-$(VERSION).tar.gz created." && cd $(CWD) )
