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
  glm::vec3 campos, glm::vec3 A, glm::vec3 B, glm::vec3 C,
  float time
  ) 
{  
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;

  glm::vec2 uv((float)x/w, (float)y/h);
  
  glm::vec3 ro = campos+C
    + (2.0f*uv.x-1.0f)*A
    + (2.0f*uv.y-1.0f)*B;
  glm::vec3 rd = glm::normalize(ro-campos);

  pbo_out[y*w + x] = rgbToInt(rd.x, rd.y, rd.z);
}

extern "C" 
void raytrace(
  uint *pbo_out, uint w, uint h,
  glm::vec3 campos, glm::vec3 A, glm::vec3 B, glm::vec3 C,
  float time
  )
{
  dim3 block(8,8);
	dim3 grid(w/block.x,h/block.y);
	raytraceKernel<<<grid, block>>>(
    pbo_out,w,h,
    campos,A,B,C,
    time);
}