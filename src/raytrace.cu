#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "common.h"
#include "Object.inl"
#include "Ray.inl"
#include "Utils.inl"

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

__global__ void initBuffers(
  const uint w, const uint h,
  const glm::vec3 campos, const glm::vec3 A, const glm::vec3 B, const glm::vec3 C,
  Ray::Ray* rays, glm::vec3* col,
  glm::vec3* film, uint filmAccumNum)
{
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;
  uint idx = y*w + x;

  // calc camera rays
  glm::vec2 uv((float)x/w, (float)y/h);  
  rays[idx].m_pos = campos+C + (2.0f*uv.x-1.0f)*A + (2.0f*uv.y-1.0f)*B;
  rays[idx].m_dir = glm::normalize(rays[idx].m_pos-campos);

  // reset color buffer
  col[idx] = glm::vec3(1.0f);

  if (filmAccumNum==1)
    film[idx] = glm::vec3(0.0f);
}

__global__ void calcColorKernel(
  const uint w, const uint h, const float time,
  const Object::Object* scene, const uint sceneSize,  
  Ray::Ray* rays,
  glm::vec3* col)
{ 
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;
  uint idx = y*w + x;

  Ray::Hit hit = Ray::intersectScene(rays[idx], scene, sceneSize);
  
  if (hit.m_id < 0) {
    col[idx] = glm::vec3(0.0f);
  }
  else {
    col[idx] *= scene[hit.m_id].m_material.m_color * scene[hit.m_id].m_material.m_brdf;
    //rays[idx].m_dir = glm::reflect(rays[idx].m_dir, hit.m_nor);
    rays[idx].m_dir = Utils::randVectorHem(glm::vec3(x,y,time),hit.m_nor);
    rays[idx].m_pos = hit.m_pos + EPS*rays[idx].m_dir;
  }
}

__global__ void accumColorKernel(
  const uint w, const uint h,
  uint* pbo_out,
  glm::vec3* col,
  glm::vec3* film, const float filmAccumNum)
{
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;
  uint idx = y*w + x;

  film[idx] += col[idx];
  pbo_out[idx] = rgbToInt(film[idx]/filmAccumNum);
}


extern "C"
void raytrace(
  uint* pbo_out, const uint w, const uint h, const float time,
  const glm::vec3& campos, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C,
  const Object::Object* scene_d, const uint sceneSize,
  Ray::Ray* rays_d,
  glm::vec3* col_d,
  glm::vec3* film_d, const uint filmAccumNum)
{
  dim3 block(8,8);
	dim3 grid(w/block.x,h/block.y);
  initBuffers<<<grid, block>>>(w,h,campos,A,B,C,rays_d,col_d,film_d,filmAccumNum);
  for (int i=0; i<PATH_DEPTH; ++i)
    calcColorKernel<<<grid, block>>>(w,h,time,scene_d,sceneSize,rays_d,col_d);
  accumColorKernel<<<grid, block>>>(w,h,pbo_out,col_d,film_d,filmAccumNum);
}