################################################################################
# Build script for OpenACC implementation of cmeans using GPUs on Delta
# lihui@indiana.edu   last update 6/24/2012
################################################################################

all:openacc_cmeans
NVDEBUG=1
CC	= pgcc
CPP	= pgcpp
GCC	= g++
.c.o:
	$(CC) -c $<

OBJ    = cmeans_openacc.o
INCLUDE  = -I.
#LIBPATH  = -L/N/u/lihui/CUDA/OpenACC/linux86-64/12.5/lib/
#ACCFLAGS = -fast -Mipa=fast,inline,reshape -acc -ta=nvidia -Minfo=acc 
ACCFLAGS = -fast -ta=nvidia,time -Minfo -Minline=levels:3 -acc


cmeans_openacc.o: cmeans_openacc.c
	$(CC) $(ACCFLAGS) -c cmeans_openacc.c -o cmeans_openacc.o $(INCLUDE)
openacc_cmeans: $(OBJ)
	$(CC) $(ACCFLAGS) -o openacc_cmeans $(OBJ) -lrt $(LIBPATH)
clean:
	rm *.o *.oo openacc_cmeans
