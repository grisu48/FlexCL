
/* =============================================================================
 * 
 * Title:         Query devices for OpenGL/OpenCL interoperability
 * Author:        Felix Niederwanger
 * =============================================================================
 */



#include <iostream>
#include <vector>
#include <string>
#include <exception>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <GL/glx.h>
#include <CL/cl_gl.h>

using namespace std;




static cl_context createOpenGlContext() {
	/* ==== Set up OpenGL. This is needed otherwise the current OpenGL device is not found by OpenGL ==== */
	int argc = 0;
	glutInit(&argc, NULL);
	glutCreateWindow("");
	
	GLenum res = glewInit();
    if (res != GLEW_OK)
    {
        string error = "Glew error";
        char* glewError = (char*)glewGetErrorString(res);
        error = error + string( glewError );
        throw error;
    }
	
	/* ==== Setting up OpenCL and query OpenGL shared devices==== */
	const int max_devices = 100;
	cl_int state;
	cl_platform_id platform[max_devices];
	cl_context context;
	cl_device_id devices[max_devices];

	//Get platform
	cl_uint numberOfPlatforms = 0;
	if(clGetPlatformIDs(100, platform, &numberOfPlatforms) != CL_SUCCESS)
		throw "Platform not found";
	if(numberOfPlatforms <= 0)
		throw "No platforms found";

	//Parameters needed to bind OpenGL's context to OpenCL's.
	cl_context_properties properties[] = {	CL_GL_CONTEXT_KHR, (cl_context_properties) glXGetCurrentContext(),
						CL_GLX_DISPLAY_KHR, (cl_context_properties) glXGetCurrentDisplay(),
						CL_CONTEXT_PLATFORM, (cl_context_properties) platform[0],
						0};

	//Find openGL devices.
	typedef CL_API_ENTRY cl_int (CL_API_CALL *CLpointer)(const cl_context_properties *properties, cl_gl_context_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
	CL_API_ENTRY cl_int (CL_API_CALL *myCLGetGLContextInfoKHR)(const cl_context_properties *properties, cl_gl_context_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) = (CLpointer)clGetExtensionFunctionAddressForPlatform(platform[0], "clGetGLContextInfoKHR");

	size_t size;
	state = myCLGetGLContextInfoKHR(properties, CL_DEVICES_FOR_GL_CONTEXT_KHR, max_devices*sizeof(cl_device_id), devices, &size);

	if(state != CL_SUCCESS)
		throw "Devices resolution failed (clGetGLContextInfoKHR)";
	
	const int numberSharedDevices = (int)(size/sizeof(cl_device_id));
	if(numberSharedDevices <= 0) throw "No devices found";
	
	context = clCreateContext(properties, 1, &devices[0], NULL, NULL, &state);
	if(state != CL_SUCCESS)
		throw "Creating context failed";
		
	return context;
}






int main(int argc, char** argv) {
    cout << "OpenCL/OpenGL interoperability | Query devices" << endl;

	try {
		cl_context context = createOpenGlContext();
		if(context == NULL) {
			cerr << "Context initialisation failed" << endl;
			return EXIT_FAILURE;
		} else {
			cout << "  OpenGL/OpenCL shared context created successfully" << endl;
		}
	} catch (const char *msg) {
		cerr << "Creating OpenGL/OpenCL shared context failed: " << msg << endl;
		return EXIT_FAILURE;
	} catch (string &msg) {
		cerr << "Creating OpenGL/OpenCL shared context failed: " << msg << endl;
		return EXIT_FAILURE;
	}

    return EXIT_SUCCESS;
}
