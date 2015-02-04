
// Swap two buffers
#define SWAP(a,b) { __local int* tmp = a; a = b; b = tmp; }



__kernel void prefix_sum(__global int* dst, __global int* src, __local int* b, __local int* c) {
	/*int previous = src[0];
	dst[0] = 0;
	
	for(size_t i=1;i<len;i++) {
		int next = src[i];
		dst[i] = dst[i-1] + previous;
		previous = next;
	} */
	
	
	const int g_id = get_global_id(0);
	const int len  = get_global_size(0);
	
	// printf("gid = %d, size = %d\n",g_id,len);
		
	// coalesced copy
	int value = src[g_id]; //(g_id>0)?src[g_id]:0;
	b[g_id] = value;
	c[g_id] = value;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Do prefix sum
	for(int offset = 1; offset < len; offset <<= 1) {
		if(g_id >= offset) {
			// Do addition
			c[g_id] = b[g_id] + b[g_id - offset];
		} else {
			// Just copy the values
			c[g_id] = b[g_id];
		}
			
		// Memory barrier, so that all actions complete
		barrier(CLK_LOCAL_MEM_FENCE);
		SWAP(b,c);
	}
	
	//   copy to global memory
	dst[g_id] = b[g_id];
	// Barrier to wait for global memory
	barrier(CLK_GLOBAL_MEM_FENCE);
}
