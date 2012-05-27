################################################################################
# Build script for OpenMP implementation of cmeans using multi-core on Delta
# lihui@indiana.edu   last update 5/23/2012
################################################################################

all:openmp_cmeans

CC	= g++

.c.o:
	$(CC) -c $<
OBJ		= Volume.o cmeansMultiCPU.o MDL.o
INCLUDE   = -I. -I/N/soft/intel/impi/4.0.0.028/include64/
LIBPATH = -L/N/soft/intel/impi/4.0.0.028/lib64/

MDL.o: MDL.cpp
	$(CC) -c MDL.cpp
Volume.o: Volume.cpp
	$(CC) -c Volume.cpp 
cmeansMultiCPU.o: cmeansMultiCPU.cpp
	$(CC) -c cmeansMultiCPU.cpp -o cmeansMultiCPU.o -fopenmp $(LIBPATH) $(INCLUDE)

openmp_cmeans: $(OBJ)
	$(CC) -o openmp_cmeans $(OBJ) -fopenmp -lmpi $(LIBPATH)
clean:
	rm *.o openmp_cmeans
