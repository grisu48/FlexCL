
#include "people.h.cl"

#define VALUE float
#define SWAP(a,b) { __local int* tmp = a; a = b; b = tmp; }

// Uncomment this, if you need the serial functions (debug configuration)
//#define SERIAL

__kernel void init_histogram(__global int* histogram, int size) {
	const int g_id = get_global_id(0);
	const int g_size  = get_global_size(0);
	
	if(g_id < size)
		histogram[g_id] = 0;
	
	// Fillup, if we have too less worker kernels
	if(g_size < size) {
		for(int i=g_size;i<size;i++) {
			const int index = g_id + i;
			if(index < size)
				histogram[index] = 0;
		}
	}
}

__kernel void histogram(__global int* values, __global int* histogram, long n_entries) {
	const int g_id    = get_global_id  (0);
	const long g_size = get_global_size(0);
	
	if(g_id > n_entries) return;
	
	const int key = values[g_id];
	const int prev = atomic_inc(histogram + key);
}


#ifdef SERIAL
/* Prefix sum using Hillis-Steele approach */
__kernel void prefix_sum(__global int* dst, __global int* src, __local int* b, __local int* c) {
	const int g_id = get_global_id(0);
	const int len  = get_global_size(0);

// Serial version for testing	
	int previous = src[0];
	dst[0] = 0;
	
//	if(g_id != 0) return;
	
	for(size_t i=1;i<MAX_AGE;i++) {
		int next = src[i];
		dst[i] = dst[i-1] + previous;
		previous = next;
	}
}

#else

/* Prefix sum using Hillis-Steele approach
 * The result is either a INCLUSIVE or EXCLUSIVE prefix scan, depending on the
 * last line of code of the kernel.
 * */
__kernel void prefix_sum(__global int* dst, __global int* src, __local int* b, __local int* c) {
	const int g_id = get_global_id(0);
	const int len  = get_global_size(0);
		
	
	// Coalesced copy
	{
		const int value = src[g_id]; //(g_id>0 && g_id <len)?src[g_id]:0;
		b[g_id] = value;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	
	// Do actual prefix sum
	for(int offset = 1; offset < len; offset <<= 1) {
		if(g_id >= offset) {
			// Do addition
			c[g_id] = b[g_id] + b[g_id - offset];
		} else {
			// Just copy the values
			c[g_id] = b[g_id];
		}
			
		// Memory barrier, so that all actions complete, before we ...
		barrier(CLK_LOCAL_MEM_FENCE);
		// ... swap the buffers.
		SWAP(b,c);
	}
	
	/* Copy result to global memory
	 * If you need a EXCLUSIVE PREFIX SCAN, uncomment the following line
	 * 
	 * EXCLUSIVE: dst[g_id] = c[g_id];
	 * INCLUSIVE: dst[g_id] = c[g_id-1];		// BUT exclude the FIRST item
	 * 
	 */
	if(g_id>0)
		dst[g_id] = b[g_id-1];
	else
		dst[g_id] = 0;
}

/* Improved Hillis-Steele prefix sum scan using Down-Sweep 
 * The result is either a INCLUSIVE or EXCLUSIVE prefix scan, depending on the
 * last line of code of the kernel.
 * */
#if 0
__kernel void prefix_sum(__global int* dst, __global int* src, __local int* b, __local int* c) {
	const int g_id = get_global_id(0);
	const int len  = get_global_size(0);
	int offset = 1;
	
	// Coalesced copy
	{
		const int value = src[g_id]; //(g_id>0 && g_id <len)?src[g_id]:0;
		b[g_id] = value;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Start rightmost and build up sum
	for(int i = len>>1; i > 0; i >>= 1) {
		if(g_id < i) {
			int ai = offset * (2*g_id+1) - 1;
			int bi = offset * (2*g_id+2) - 1;
			b[bi] += b[ai];
		}
		offset <<= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// Clear last element
	if(g_id == len-1) c[len-1] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Traverse down, build prefix scan
	for(int i=1;i<len;i<<=1) {
		offset >>= 1;
		
		if(g_id < i) {
			int ai = offset * (2*g_id+1) - 1;
			int bi = offset * (2*g_id+2) - 1;
			int tmp = b[ai];
			b[ai] = b[bi];
			b[bi] = tmp;
		}
		
//		SWAP(b,c);
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(g_id>0)
		dst[g_id] = b[g_id-1];
	else
		dst[g_id] = 0;
}

#endif
#endif

__kernel void prefix_sum_sort(volatile __global int* indices, __global int* keys, __global int* prefix_sums) {
	const int g_id    = get_global_id  (0);
	// Key of this element
	const int key = keys[g_id];
	// When we have the key, get the next available index
	int dest_index = atomic_inc(prefix_sums + key);
	indices[dest_index] = g_id;
	
}


