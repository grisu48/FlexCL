// the type used to represent a triple of floats
typedef struct {
	float x, y, z;
} triple;

// ----- some operators -----
#define SMOOTHING_LENGTH 0.5

triple add(triple a, triple b) {
	triple ret = { a.x + b.x, a.y + b.y, a.z + b.z };
	return ret;
}
triple sub(triple a, triple b) {
	triple ret = { a.x - b.x, a.y - b.y, a.z - b.z };
	return ret;
}
triple div_(triple a, triple b) {
	triple ret = { a.x / b.x, a.y / b.y, a.z / b.z };
	return ret;
}

triple mul_s(triple a, float s) {
	triple ret = { a.x * s, a.y * s, a.z * s };
	return ret;
}

triple div_s(triple a, float s) {
	triple ret = { a.x / s, a.y / s, a.z / s };
	return ret;
}

float norm(triple t) {
	return sqrt(t.x*t.x + t.y*t.y + t.z*t.z);
}

__kernel void compute_force (__global triple* position, __global float* mass ,__global triple* velocity , __global triple* force) {

	size_t global_id = get_global_id(0);
	size_t global_size = get_global_size(0);

	for(int k = 0; k < global_size; k++) {
		if(global_id != k) {
			// compute distance vector
			triple dist = sub(position[k], position[global_id]);

			// compute absolute distance
			float r = norm(dist) + SMOOTHING_LENGTH;

			// compute strength of force (G = 1 (who cares))
			//			F = G * (m1 * m2) / r^2
			float f = (mass[global_id] * mass[k]) / (r*r);

			// compute current contribution to force
			float s = f / r;
			triple cur = mul_s(dist,s);

			// accumulate force
			force[global_id] = add(force[global_id], cur);
		}
		
	}
}


__kernel void update (__global triple* position, __global float* mass ,__global triple* velocity , __global triple* force) {

	size_t global_id = get_global_id(0);

	// update velocity
	velocity[global_id] = add(velocity[global_id], div_s(force[global_id], mass[global_id]));

	// update position
	position[global_id] = add(position[global_id], velocity[global_id]);
	
	// reset force
	force[global_id].x = 0.0f;
	force[global_id].y = 0.0f;
	force[global_id].z = 0.0f;	
}


