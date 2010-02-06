# Solaris 2.X, HPUX 9.X and 10.X cc options
#CFLAGS = -Aa -O

# Linux or MacOSX 10.2 or most gcc compilers
CC = gcc
CFLAGS = -Wall -O2
#CFLAGS = -ansi -Wall -pedantic -O
#CFLAGS = -ansi -Wall -pedantic -g


BIN = ..

CLUSTER_OBJECTS = clust.o alloc_util.o clust_invert.o clust_io.o clust_util.o subcluster.o 

CLASSIFY_OBJECTS = classify.o classify_util.o clust_io.o clust_util.o alloc_util.o eigen.o invert.o

SPLITCLASSES_OBJECTS = SplitClasses.o alloc_util.o clust_invert.o clust_io.o clust_util.o subcluster.o 


HEADER_FILES = clust_io.h clust_util.h clust_defs.h alloc_util.h subcluster.h

default : all

all : clust classify SplitClasses

clust : clust.c $(CLUSTER_OBJECTS) $(HEADER_FILES)
	$(CC) $(CFLAGS) -o $(BIN)/clust $(CLUSTER_OBJECTS) -lm

classify: classify.c $(CLASSIFY_OBJECTS) $(HEADER_FILES)
	$(CC) $(CFLAGS) -o $(BIN)/classify $(CLASSIFY_OBJECTS) -lm

SplitClasses: SplitClasses.c $(SPLITCLASSES_OBJECTS) $(HEADER_FILES)
	$(CC) $(CFLAGS) -o $(BIN)/SplitClasses $(SPLITCLASSES_OBJECTS) -lm


clean:
	rm -f *.o $(BIN)/clust $(BIN)/classify $(BIN)/SplitClasses

