
CC          = gcc
MPICC       = mpicc
NVCC        = nvcc

.c.o:
	$(CC) $(CFLAGS) -c $<

H_FILES     = kmeans.h

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

CUDA_C_SRC = cuda_main.cu cuda_io.cu cuda_wtime.cu
CUDA_CU_SRC = cuda_kmeans.cu

CUDA_C_OBJ = $(CUDA_C_SRC:%.cu=%.o)
CUDA_CU_OBJ = $(CUDA_CU_SRC:%.cu=%.o)

cuda: cuda_main
cuda_main: $(CUDA_C_OBJ) $(CUDA_CU_OBJ)
	$(NVCC) $(LDFLAGS) -o $@ $(CUDA_C_OBJ) $(CUDA_CU_OBJ)

clean:
	rm -rf *.o omp_main seq_main mpi_main cuda_main \
