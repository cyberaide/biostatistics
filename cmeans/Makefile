################################################################################
# Build script for MPI/CUDA implementation of cmeans on Delta
# lihui@indiana.edu   5/9/2012
################################################################################

all:mpi_cuda

CC	= icc
MPICC	= mpiicc

NVCCFLAGS += --ptxas-options=-v
NVCCFLAGS += -Xcompiler -fopenmp
INCLUDE   = -I. -I/N/soft/intel/impi/4.0.0.028/include64/
INCLUDE   += -I/opt/cuda/include/ -I/opt/NVIDIA_GPU_Computing_SDK/C/common/inc/
LIBPATH = -L/N/soft/intel/impi/4.0.0.028/lib64/ -L/opt/cuda/lib64/ -L/opt/NVIDIA_GPU_Computing_SDK/C/lib/

.c.o:
	$(CC) -c $<
OBJ		= Volume.o cmeans.o MDL.o

MDL.o: MDL.cpp
	$(CC) -c MDL.cpp $(INCLUDE) -lcudart
Volume.o: Volume.cpp
	$(CC) -c Volume.cpp $(INCLUDE) -lcudart $(INCLUDE)
#cmeans_kernel.o: cmeans_kernel.cu
#	nvcc -c cmeans_kernel.cu -o cmeans_kernel.o $(INCLUDE)
cmeans.o: cmeans.cu
	nvcc -arch=sm_21 -c cmeans.cu -o cmeans.o $(INCLUDE) -lcudart -lmpi -lgomp

mpi_cuda: $(OBJ)
	nvcc -arch=sm_21 -o cuda_cmeans $(OBJ) $(INCLUDE) $(LIBPATH) -lcudart -lmpi -lgomp -lcutil_x86_64
clean:
	rm *.o cuda_cmeans
