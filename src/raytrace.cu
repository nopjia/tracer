#include <cuda_runtime.h>
#include <curand.h>
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

__global__ void testKernel(
  const uint w, const uint h,
  const glm::vec3 campos, const glm::vec3 A, const glm::vec3 B, const glm::vec3 C,
  uint* pbo_out, 
  const Object::Object* scene, const uint sceneSize)
{
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;
  uint idx = y*w + x;

  // calc camera rays
  glm::vec2 uv((float)x/w, (float)y/h);  
  Ray::Ray ray;
  ray.m_pos = campos+C + (2.0f*uv.x-1.0f)*A + (2.0f*uv.y-1.0f)*B;
  ray.m_dir = glm::normalize(ray.m_pos-campos);

  glm::vec3 lightDir(0.267261, 0.801784, 0.534522);
  Ray::Hit hit = Ray::intersectScene(ray, scene, sceneSize);
  //Ray::Hit hit = Ray::intersect(ray, *scene[0].m_mesh);

  glm::vec3 col;
  if (hit.m_id < 0) {
    col = ray.m_dir;
  }
  else {
    col = scene[hit.m_id].m_material.m_color * scene[hit.m_id].m_material.m_brdf;
    col *= glm::max(glm::dot(lightDir,hit.m_nor),0.0f);
  }

  pbo_out[idx] = rgbToInt(col);
}

__global__ void initBuffersKernel(
  const uint w, const uint h,
  const glm::vec3 campos, const glm::vec3 A, const glm::vec3 B, const glm::vec3 C,
  Ray::Ray* rays, glm::vec3* col, uint* flags,
  glm::vec3* film, uint filmIters)
{
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;
  uint idx = y*w + x;
  
  // calc camera rays
  glm::vec2 uv((float)x/w, (float)y/h);
  rays[idx].m_pos = campos+C + (2.0f*uv.x-1.0f)*A + (2.0f*uv.y-1.0f)*B;
  rays[idx].m_dir = glm::normalize(rays[idx].m_pos-campos);

  // reset buffers
  col[idx] = glm::vec3(1.0f);

  if (filmIters==1)
    film[idx] = glm::vec3(0.0f);

  flags[idx] = THFL_PATH_RUN;
}

__global__ void calcColorKernel(
  const uint w, const uint h, const float time,
  const Object::Object* scene, const uint sceneSize,
  glm::vec3* rand,
  uint* flags,
  Ray::Ray* rays,
  glm::vec3* col,
  const uint depth)
{ 
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;
  uint idx = y*w + x;

  if (!flags[idx]&THFL_PATH_RUN)
    return;

  Ray::Hit hit = Ray::intersectScene(rays[idx], scene, sceneSize);
  
  if (hit.m_id < 0) {
    col[idx] = glm::vec3(0.0f);
  }
  else {
    if (scene[hit.m_id].m_material.m_emit > 0.0f) {
      col[idx] *= scene[hit.m_id].m_material.m_color* scene[hit.m_id].m_material.m_emit;
      flags[idx] &= !THFL_PATH_RUN;
    }
    else {
      if (depth == PATH_DEPTH-1) {
        col[idx] = glm::vec3(0.0f);
      }
      else {
        col[idx] *= scene[hit.m_id].m_material.m_color * scene[hit.m_id].m_material.m_brdf;
        rays[idx].m_dir = Utils::randVectorHem(rand[idx].x,rand[idx].y,hit.m_nor);
        rays[idx].m_pos = hit.m_pos + EPS*rays[idx].m_dir;
      }
    }    
  }
}

__global__ void accumColorKernel(
  const uint w, const uint h,
  uint* pbo_out,
  glm::vec3* col,
  glm::vec3* film, const float filmIters)
{
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;
  uint idx = y*w + x;

  film[idx] += col[idx];
  pbo_out[idx] = rgbToInt(film[idx]/filmIters);
}

extern "C"
void raytrace(
  uint* pbo_out, const uint w, const uint h, const float time,
  const glm::vec3& campos, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C,
  const Object::Object* scene_d, const uint sceneSize,
  glm::vec3* rand_d,
  uint* flags_d,
  Ray::Ray* rays_d,
  glm::vec3* col_d,
  glm::vec3* film_d, const uint filmIters)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, time*10.0f);
  curandGenerateUniform(gen, (float*)rand_d, 3*w*h);

  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
	dim3 grid(w/block.x,h/block.y);

  initBuffersKernel<<<grid, block>>>(w,h,campos,A,B,C,rays_d,col_d,flags_d,film_d,filmIters);
  for (int i=0; i<PATH_DEPTH; ++i)
    calcColorKernel<<<grid, block>>>(w,h,time,scene_d,sceneSize,rand_d,flags_d,rays_d,col_d,i);
  accumColorKernel<<<grid, block>>>(w,h,pbo_out,col_d,film_d,filmIters);
}

extern "C"
void testtrace(
  uint* pbo_out, const uint w, const uint h, const float time,
  const glm::vec3& campos, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C,
  const Object::Object* scene_d, const uint sceneSize)
{
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
	dim3 grid(w/block.x,h/block.y);
  testKernel<<<grid, block>>>(w,h,campos,A,B,C,pbo_out,scene_d,sceneSize);
}