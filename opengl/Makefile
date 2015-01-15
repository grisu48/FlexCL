
# This is the path where your AMD OpenCL SDK is located
AMDSDK_PATH=/opt/AMDAPP
#AMDSDK_PATH=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64

# ==== COMPILER FLAGS ==== #
O_FLAGS=-Wall -pedantic
CXX=g++
CC=gcc
CXX_FLAGS=$(O_FLAGS) -std=c++11
CC_FLAGS=$(O_FLAGS) -std=c99


# ==== INCLUDE AND LIB ==== #
INCLUDE=-I$(AMDSDK_PATH)/include -L$(AMDSDK_PATH)/lib/x86_64
OBJS=
GLUT_LIBS=-lm -lGLU -lglut -lGLU -lGL -lGLEW
OCL_LIBS=-lOpenCL

# Add you own programs here
BINS=devices


default:	all
all:	$(OBJS) $(BINS)
clean:
	rm -f *.o


# ==== PROGRAM SECTION ===== #
# Insert your program instructuions here

devices:	devices.cpp $(OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $< $(OBJS) $(INCLUDE) $(GLUT_LIBS) $(OCL_LIBS) 

