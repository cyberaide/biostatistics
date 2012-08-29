################################################################################
# Makefile for Panda source code
# lihui@indiana.edu   7/01/2012
################################################################################

all:panda_application

CC	= g++
NVCC	= nvcc

#NVCCFLAGS += --ptxas-options=-v
NVCCFLAGS += -Xcompiler -O

INCLUDE   = -I. -I/sw/keeneland/cuda/4.1/linux_binary/include/ -I/opt/NVIDIA_GPU_Computing_SDK/4.1.28/C/common/inc/
LIBPATH =  -L./ -L/sw/keeneland/cuda/4.1/linux_binary/lib64/ -L/opt/NVIDIA_GPU_Computing_SDK/4.1.28/C/lib/

.c.o:
	$(CC) -c $<

OBJ		= main.o PandaLib.o PandaUtils.o PandaScan.o PandaSort.o matrixutil.o

PandaLib.o: PandaLib.cu
	$(NVCC) -O -arch=sm_20 -c PandaLib.cu $(INCLUDE) -lcudart
PandaUtils.o: PandaUtils.cu
	$(NVCC) -O -arch=sm_20 -c PandaUtils.cu $(INCLUDE) -lcudart 
PandaScan.o: PandaScan.cu
	nvcc -arch=sm_20 -c PandaScan.cu -o PandaScan.o $(INCLUDE) -lcudart
PandaSort.o: PandaSort.cu
	nvcc -arch=sm_20 -c PandaSort.cu -o PandaSort.o $(INCLUDE) -lcudart 
matrixutil.o: matrixutil.cpp
	g++ -O -o matrixutil.o -c matrixutil.cpp
main.o: main.cu
	nvcc -O -arch=sm_20 -c main.cu -o main.o $(INCLUDE) -lcudart
panda_application: $(OBJ)
	nvcc -O -arch=sm_20 -o panda_matrixMul_1d $(OBJ) $(INCLUDE) $(LIBPATH) -lcudart 
clean:
	rm *.o panda_matrixMul
