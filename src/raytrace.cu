#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "common.h"

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
  r = glm::clamp(r, 0.0f, 1.0f);
  g = glm::clamp(g, 0.0f, 1.0f);
  b = glm::clamp(b, 0.0f, 1.0f);

  // notice switch red and blue to counter the GL_BGRA
  return (int(r*255.0)<<16) | (int(g*255.0)<<8) | int(b*255.0);
}

__global__ void raytraceKernel(
  uint *pbo_out, uint w, uint h,
  float time
  ) 
{  
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;

  float fade = (glm::sin(time+y*(16.0/WINDOW_H))+1.0)/2.0;

  pbo_out[y*w + x] = rgbToInt(fade*1.0, 0.0, fade*0.5);
}

extern "C" 
void raytrace(
  uint *pbo_out, uint w, uint h, 
  float time
  )
{
  dim3 block(8,8);
	dim3 grid(w/block.x,h/block.y);
	raytraceKernel<<<grid, block>>>(pbo_out,w,h,time);
}