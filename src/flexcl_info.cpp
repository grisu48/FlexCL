
/* =============================================================================
 * 
 * Title:       
 * Author:      
 * License:     
 * Description: 
 * 
 * 
 * =============================================================================
 */
 
 
 
#include <iostream>
#include "FlexCL.hpp"

#ifndef EXIT_SUCCESS
    #define EXIT_SUCCESS 0
#endif
using namespace std;
using namespace flexCL;

int main(int argc, char** argv) {
    cout << "FlexCL info program | 2014, Felix Niederwanger" << endl << endl;
    cout << "  FlexCL Version    " << OpenCL::VERSION() << endl;
    cout << "  FlexCL Build      " << OpenCL::BUILD() << endl;
    cout.flush();
    
    OpenCL opencl;
    
    /* Query platforms */
    {
	unsigned int platform_count = opencl.plattform_count();
	cout << "Available platforms on this computer: " << platform_count << endl;
		vector<PlatformInfo> platforms = opencl.get_platforms();
		int i = 1;
		for(vector<PlatformInfo>::iterator it=platforms.begin(); it!=platforms.end(); it++) {
			PlatformInfo platform = *it;
			cout << "  " << i++ << ":\t" << platform.name() << endl;
			cout << "    " << platform.vendor() << endl;
			cout << "    Profile        : " << platform.profile() << endl;
			cout << "    Extensions     : " << platform.extensions() << endl;
			cout << "    OpenCL Version : " << platform.version() << endl;
			
			
			cout << endl;
		}
	}
    
    
    
    return EXIT_SUCCESS;
}
