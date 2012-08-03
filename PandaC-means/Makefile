################################################################################
# Makefile for Panda source code
# lihui@indiana.edu   7/01/2012
################################################################################

all:panda_application

CC	= g++
NVCC	= nvcc

NVCCFLAGS += --ptxas-options=-v
NVCCFLAGS += -Xcompiler -fopenmp
INCLUDE   = -I. -I/opt/cuda/include/ -I/opt/NVIDIA_GPU_Computing_SDK/C/common/inc/
LIBPATH =  -L./ -L/opt/cuda/lib64/ -L/opt/NVIDIA_GPU_Computing_SDK/C/lib/

.c.o:
	$(CC) -c $<

OBJ		= main.o PandaLib.o PandaUtils.o PandaScan.o PandaSort.o

PandaLib.o: PandaLib.cu
	$(NVCC) -arch=sm_20 -c PandaLib.cu $(INCLUDE) -lcudart
PandaUtils.o: PandaUtils.cu
	$(NVCC) -arch=sm_20 -c PandaUtils.cu $(INCLUDE) -lcudart 
PandaScan.o: PandaScan.cu
	nvcc -arch=sm_20 -c PandaScan.cu -o PandaScan.o $(INCLUDE) -lcudart
PandaSort.o: PandaSort.cu
	nvcc -arch=sm_20 -c PandaSort.cu -o PandaSort.o $(INCLUDE) -lcudart 

main.o: main.cu
	nvcc -arch=sm_20 -c main.cu -o main.o $(INCLUDE) -lcudart
panda_application: $(OBJ)
	nvcc -arch=sm_20 -o panda_wc $(OBJ) $(INCLUDE) $(LIBPATH) -lcudart -lcutil_x86_64
clean:
	rm *.o panda_wc
