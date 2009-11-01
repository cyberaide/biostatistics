#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Build script for project
#
################################################################################


# Add source files here
EXECUTABLE	:= cmeans
# CUDA source files (compiled with cudacc)
CUFILES		:= cmeans.cu cmeans_kernel.cu
# CUDA dependency files
CU_DEPS		:= \
	cmeans_kernel.cu \
	cmeans.h 	 \
	cmeanscu.h 	 \
	kmeans_kernel.cu \
	kmeans.cu \
	kmeans.h

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= \
	MDL.cpp    \
	Volume.cpp \


################################################################################
# Rules and targets
include ../../common/common.mk

kmeans: $(OBJDIR)/kmeans.cu.o
	$(LINK) -o $(TARGETDIR)/kmeans $(OBJDIR)/kmeans.cu.o $(LIB)

cubin:
	nvcc -cubin cmeans_kernel.cu -I../../common/inc/

