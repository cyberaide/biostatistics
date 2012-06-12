################################################################################
# Build script for Pthreads implementation of cmeans using multi-core on Delta
# lihui@indiana.edu   last update 6/12/2012
################################################################################

all:pthread_cmeans

CC	= g++

.c.o:
	$(CC) -c $<
OBJ		= Volume.o cmeansPthreadMPICPU.o MDL.o
INCLUDE = -I. -I/N/soft/intel/impi/4.0.0.028/include64/
LIBPATH = -L/N/soft/intel/impi/4.0.0.028/lib64/

MDL.o: MDL.cpp
	$(CC) -c MDL.cpp
Volume.o: Volume.cpp
	$(CC) -c Volume.cpp 
cmeansPthreadMPICPU.o: cmeansPthreadMPICPU.cpp
	$(CC) -c cmeansPthreadMPICPU.cpp -o cmeansPthreadMPICPU.o $(INCLUDE)

pthread_cmeans: $(OBJ)
	$(CC) -o pthread_cmeans $(OBJ) -lpthread -lmpi $(LIBPATH)
clean:
	rm *.o pthread_cmeans
