#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "common.h"
#include "Object.inl"
#include "Ray.inl"

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
  r = glm::clamp(r, 0.0f, 1.0f);
  g = glm::clamp(g, 0.0f, 1.0f);
  b = glm::clamp(b, 0.0f, 1.0f);

  // notice switch red and blue to counter the GL_BGRA
  return (int(r*255.0)<<16) | (int(g*255.0)<<8) | int(b*255.0);
}
__device__ int rgbToInt(glm::vec3 c)
{
  c = glm::clamp(c, 0.0f, 1.0f);

  // notice switch red and blue to counter the GL_BGRA
  return (int(c.r*255.0)<<16) | (int(c.g*255.0)<<8) | int(c.b*255.0);
}

__global__ void raytraceKernel(
  uint *pbo_out, 
  const uint w, const uint h,
  const glm::vec3 campos, const glm::vec3 A, const glm::vec3 B, const glm::vec3 C,
  const Object::Object* scene, const uint sceneSize,
  const float time)
{  
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;

  glm::vec2 uv((float)x/w, (float)y/h);
  
  Ray::Ray ray;
  ray.m_pos = campos+C
    + (2.0f*uv.x-1.0f)*A
    + (2.0f*uv.y-1.0f)*B;
  ray.m_dir = glm::normalize(ray.m_pos-campos);

  Ray::Hit hit = Ray::intersect(ray, *scene[0].m_mesh);

  glm::vec3 outcolor;
  
  if (hit.m_id < 0) {
    outcolor = ray.m_dir;
  }
  else {
    outcolor = scene[hit.m_id].m_material.m_color;
  }

  pbo_out[y*w + x] = rgbToInt(outcolor);
  //pbo_out[y*w + x] = rgbToInt(rd);
}

extern "C" 
void raytrace(
  uint *pbo_out, 
  const uint w, const uint h,
  const glm::vec3& campos, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C,
  const Object::Object* scene, const uint sceneSize,
  const float time)
{
  dim3 block(8,8);
	dim3 grid(w/block.x,h/block.y);
	raytraceKernel<<<grid, block>>>(
    pbo_out,w,h,
    campos,A,B,C,
    scene,sceneSize,
    time);
}