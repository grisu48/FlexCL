
# This is the path where your AMD OpenCL SDK is located
AMDSDK_PATH=/opt/AMDAPP
#AMDSDK_PATH=/scratch/c703/c703432/amd_ocl/AMD-APP-SDK-v2.6-RC3-lnx64

# ==== COMPILER FLAGS ==== #
O_FLAGS=-Wall -pedantic -O2
CXX=g++
CXX_FLAGS=$(O_FLAGS) -std=c++11


# ==== INCLUDE AND LIB ==== #
INCLUDE=-I$(AMDSDK_PATH)/include -L$(AMDSDK_PATH)/lib/x86_64
OCL_LIBS=-lOpenCL
# GLUT libs, needed if the library is compiled with OpenGL support
GLUT_LIBS=-lm -lGLU -lglut -lGLU -lGL -lGLEW
OBJS=FlexCL.o
LIBS=$(OCL_LIBS) $(GLUT_LIBS)

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
	$(CXX) $(CXX_FLAGS) -o $@ $< $(OBJS) $(INCLUDE) $(LIBS) $(GLUT_LIBS) 

flexcl_info:	flexcl_info.cpp $(OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $< $(OBJS) $(INCLUDE) $(LIBS) 
