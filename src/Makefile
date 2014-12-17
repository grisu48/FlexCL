
# This is the path where your AMD OpenCL SDK is located
AMDSDK_PATH=/opt/AMDAPP
#AMDSDK_PATH=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64

# ==== COMPILER FLAGS ==== #
CXX=g++
CXX_FLAGS=-Wall -pedantic -std=c++11


# ==== INCLUDE AND LIB ==== #
INCLUDE=-I$(AMDSDK_PATH)/include -L$(AMDSDK_PATH)/lib/x86_64
LIBS=-lOpenCL
OBJS=FlexCL.o

# Add you own programs here
BINS=example flexcl_info


default:	all
all:	$(OBJS) $(BINS)
clean:
	rm -f *.o
FlexCL.o:	FlexCL.cpp FlexCL.hpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $< $(INCLUDE)


# ==== PROGRAM SECTION ===== #
# Insert your program instructuions here

example:	example.cpp $(OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $< $(OBJS) $(INCLUDE) $(LIBS) 

flexcl_info:	flexcl_info.cpp $(OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $< $(OBJS) $(INCLUDE) $(LIBS) 
