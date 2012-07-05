# 
#        Copyright 2012 The Trustees of Indiana University.  All rights reserved.^M
#        CGL MapReduce Framework on GPUs and CPUs^M
#        Code Name: Panda 0.1^M
#        File: Global.h ^M
#        Time: 2012-07-01 ^M
#        Developer: Hui Li (lihui@indiana.edu)^M
#        This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.^M
# 


all:word_count

CC	= g++
NVCC	= nvcc

NVCCFLAGS += --ptxas-options=-v
NVCCFLAGS += -Xcompiler -fopenmp
INCLUDE   = -I. -I/opt/cuda-4.2.9/cuda/include/ -I/opt/NVIDIA_GPU_Computing_SDK/C/common/inc/
LIBPATH =  -L. -L/opt/cuda-4.2.9/cuda/lib64/ -L/opt/NVIDIA_GPU_Computing_SDK/C/lib/

.c.o:
	$(CC) -c $<
OBJ		= main.o PandaLib.o PandaUtils.o PandaScan.o PandaSort.o

PandaLib.o: PandaLib.cu
	$(NVCC) -arch=sm_20 -c PandaLib.cu $(INCLUDE)
PandaUtils.o: PandaUtils.cpp
	$(CC) -c PandaUtils.cpp $(INCLUDE)
PandaScan.o: PandaScan.cu
	nvcc -arch=sm_20 -c PandaScan.cu $(INCLUDE)
PandaSort.o: PandaSort.cu
	nvcc -arch=sm_20 -c PandaSort.cu $(INCLUDE)
main.o: main.cu
	nvcc -arch=sm_20 -c main.cu -o main.o $(INCLUDE)
word_count: $(OBJ)
	nvcc -arch=sm_20 -o word_count $(OBJ) $(INCLUDE) $(LIBPATH) -lcuda -lcudart -lcutil_x86_64
clean:
	rm *.o word_count
