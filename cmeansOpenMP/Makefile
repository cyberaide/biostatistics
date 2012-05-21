################################################################################
# Build script for OpenMP/CUDA implementation of cmeans on Delta
# lihui@indiana.edu   5/9/2012
################################################################################

all:openmp_cuda

CC	= g++

NVCCFLAGS += --ptxas-options=-v
NVCCFLAGS += -Xcompiler -fopenmp
INCLUDE   = -I. -I/N/soft/intel/impi/4.0.0.028/include64/
INCLUDE   += -I/opt/cuda/include/ -I/opt/NVIDIA_GPU_Computing_SDK/C/common/inc/
LIBPATH = -L/N/soft/intel/impi/4.0.0.028/lib64/ -L/opt/cuda/lib64/ -L/opt/NVIDIA_GPU_Computing_SDK/C/lib/

.c.o:
	$(CC) -c $<
OBJ		= Volume.o cmeansMultiGPU.o MDL.o

MDL.o: MDL.cpp
	$(CC) -c MDL.cpp $(INCLUDE) -lcudart
Volume.o: Volume.cpp
	$(CC) -c Volume.cpp $(INCLUDE) -lcudart $(INCLUDE)

#cmeansMPI_kernel.o: cmeansMPI_kernel.cu
#	nvcc -c cmeansMPI_kernel.cu -o cmeansMPI_kernel.o $(INCLUDE)

cmeansMultiGPU.o: cmeansMultiGPU.cu
	nvcc -c cmeansMultiGPU.cu -o cmeansMultiGPU.o $(INCLUDE) -lcudart -lmpi -lgomp $(NVCCFLAGS)

openmp_cuda: $(OBJ)
	$(CC) -o openmp_cuda_cmeans $(OBJ) $(INCLUDE) $(LIBPATH) -lcudart -lmpi -fopenmp -lcutil_x86_64
#	tcc -o openmp_cuda_cmeans $(OBJ) $(INCLUDE) $(LIBPATH) -lcudart -openmp -lcutil_x86_64
clean:
	rm *.o openmp_cuda_cmeans
