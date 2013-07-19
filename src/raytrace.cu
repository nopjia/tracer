#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/count.h>
#include <thrust/copy.h>
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

__global__ void raytraceKernel(
  const uint w, const uint h,
  const glm::vec3 campos, const glm::vec3 A, const glm::vec3 B, const glm::vec3 C,
  uint* pbo_out, 
  const Object::Object* scene, const uint sceneSize)
{
  uint idx = blockIdx.x*blockDim.x + threadIdx.x;
  uint x = idx % w;
  uint y = idx / w;

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
    if (scene[hit.m_id].m_material.m_emit > 0.0f)
      col = scene[hit.m_id].m_material.m_color;
    else {
      col = scene[hit.m_id].m_material.m_color * scene[hit.m_id].m_material.m_brdf;
      col *= glm::max(glm::dot(lightDir,hit.m_nor),0.0f);
    }
  }

  pbo_out[idx] = rgbToInt(col);
}

__global__ void initBuffersKernel(
  const uint w, const uint h,
  const glm::vec3 campos, const glm::vec3 A, const glm::vec3 B, const glm::vec3 C,
  const float lensRadius, const float focalDist,
  glm::vec3* rand,
  Ray::Ray* rays, glm::vec3* col, int* indices,
  glm::vec3* film, uint filmIters)
{
  uint idx = blockIdx.x*blockDim.x + threadIdx.x;
  uint x = idx % w;
  uint y = idx / w;
  
  // calc camera rays
  glm::vec2 uv(
    (float)x/w + (2.0f*rand[idx].x-1.0f)/w, 
    (float)y/h + (2.0f*rand[idx].y-1.0f)/h
  );
  rays[idx].m_pos = campos+C + (2.0f*uv.x-1.0f)*A + (2.0f*uv.y-1.0f)*B;
  rays[idx].m_dir = glm::normalize(rays[idx].m_pos-campos);

  // focal blur
#ifdef FOCAL_BLUR
  glm::vec3 fpt = focalDist*rays[idx].m_dir+rays[idx].m_pos;
  glm::vec2 randdisk = lensRadius*Utils::randPointDisk(rand[idx].x,rand[idx].y,rand[idx].z);
  rays[idx].m_pos += randdisk.x*glm::normalize(A) + randdisk.y*glm::normalize(B);
  rays[idx].m_dir = glm::normalize(fpt-rays[idx].m_pos);
#endif

  // reset buffers
  col[idx] = glm::vec3(1.0f);

  indices[idx] = idx;

  if (filmIters==1)
    film[idx] = glm::vec3(0.0f);
}

__global__ void calcColorKernel(
  const uint size, const float time,
  const Object::Object* scene, const uint sceneSize,
  glm::vec3* rand,
  Ray::Ray* rays,
  glm::vec3* col,
  int* indices,
  const int depth)
{
  uint idx = blockIdx.x*blockDim.x + threadIdx.x;

  // bounds check
  if (idx >= size)
    return;

  // indicate terminate path
  if (indices[idx] == -1)
    return;

  // intersection test
  Ray::Hit hit = Ray::intersectScene(rays[idx], scene, sceneSize);
  
  // intersects nothing, kill path
  if( hit.m_id < 0 ) {
    col[idx] = glm::vec3(0.0f);   // BLACK
    indices[idx] = -1;
    return;
  }

  // intersects light, kill path
  if (scene[hit.m_id].m_material.m_emit > 0.0f) {
    col[idx] *= scene[hit.m_id].m_material.m_color*scene[hit.m_id].m_material.m_emit;
    indices[idx] = -1;
    return;
  }

  // at max depth, not seen light, does not contribute color
  if (depth == PATH_DEPTH-1) {
    col[idx] = glm::vec3(0.0f);   // BLACK
    return;
  }

  // else, compute color, bounce

  col[idx] *= scene[hit.m_id].m_material.m_color;

  // cycle thru rand array with depth
  uint randidx = (idx + depth) % (size);
  rays[idx].m_dir = Material::bounce(scene[hit.m_id].m_material,
    rays[idx].m_dir, hit.m_nor, rand[randidx]);

  rays[idx].m_pos = hit.m_pos + EPS*rays[idx].m_dir;
}

__global__ void accumColorKernel(
  uint* pbo_out,
  int* indices,
  glm::vec3* col,
  glm::vec3* film, const float filmIters)
{
  uint idx = blockIdx.x*blockDim.x + threadIdx.x;

  film[idx] += col[idx];

#ifdef GAMMA_CORRECT
  pbo_out[idx] = rgbToInt( glm::pow(film[idx]/filmIters, glm::vec3(1.0f/2.2f)) );
#else
  pbo_out[idx] = rgbToInt(film[idx]/filmIters);
#endif
}

__global__ void testRand(
  uint* pbo_out,
  glm::vec3* rand
  )
{
  uint i = blockIdx.x*blockDim.x + threadIdx.x;
  pbo_out[i] = rgbToInt(rand[i]);
}

extern "C"
void pathtrace(
  uint* pbo_out, const uint w, const uint h, const float time,
  const glm::vec3& campos, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C,
  const float lensRadius, const float focalDist,
  const Object::Object* scene_d, const uint sceneSize,
  glm::vec3* rand_d,
  Ray::Ray* rays_d,
  glm::vec3* col_d,
  int* idx_d,
  glm::vec3* film_d, const uint filmIters)
{
  uint pixSize = w*h;

  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, time*100.0f);
  curandGenerateUniform(gen, (float*)rand_d, 3*pixSize);
  
  uint blockSize = BLOCK_SIZE;
  uint gridSize = pixSize/blockSize + (pixSize%blockSize==0 ? 0:1);
  
  // INIT BUFFERS
  initBuffersKernel<<<gridSize, blockSize>>>(
    w,h,campos,A,B,C,lensRadius,focalDist,rand_d,rays_d,col_d,idx_d,film_d,filmIters
  );
  
  // PATH TRACE
  for (int i=1; i<PATH_DEPTH; ++i) {
    calcColorKernel<<<gridSize, blockSize>>>(
      pixSize,time,scene_d,sceneSize,rand_d,rays_d,col_d,idx_d,i
    );
  }

  // ACCUM OUTPUT
  accumColorKernel<<<gridSize, blockSize>>>(
    pbo_out,idx_d,col_d,film_d,filmIters);
}

extern "C"
void raytrace1(
  uint* pbo_out, const uint w, const uint h, const float time,
  const glm::vec3& campos, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C,
  const Object::Object* scene_d, const uint sceneSize)
{
  uint pixSize = w*h;
  uint blockSize = BLOCK_SIZE;
  uint gridSize = pixSize/blockSize + (pixSize%blockSize==0 ? 0:1);
  raytraceKernel<<<gridSize,blockSize>>>(w,h,campos,A,B,C,pbo_out,scene_d,sceneSize);
}