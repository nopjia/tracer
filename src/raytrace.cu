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

template <typename T>
struct path_alive : public thrust::unary_function<T,bool>
{
  __host__ __device__
  bool operator()(T x)
  {
    return x >= 0;
  }
};

__global__ void compactKernel(
  const uint size,
  int* indices,
  Ray::Ray* rays,
  glm::vec3* col
  )
{
  uint glid = blockIdx.x*blockDim.x + threadIdx.x;
  uint thid = threadIdx.x;

  // stencil 
  __shared__ int stencil[BLOCK_SIZE];
  stencil[thid] = indices[glid]>=0 ? 1 : 0;

  // scatter array
  __shared__ int addr[BLOCK_SIZE];
  addr[thid] = stencil[thid];
  __syncthreads();
  // inclusive scan
  for (uint offset=1; offset<size; offset*=2)
    if (thid >= offset)
      addr[thid] = addr[thid] + addr[thid-offset];
    __syncthreads();
  // convert to exclusive/shift right
  addr[thid] = thid==0 ? 0 : addr[thid-1];
  __syncthreads();

  // write out temp
  __shared__ int indices_out[BLOCK_SIZE];
  __shared__ Ray::Ray rays_out[BLOCK_SIZE];
  __shared__ glm::vec3 col_out[BLOCK_SIZE];
  if (glid<size && stencil[thid]==1) {
    indices_out[addr[thid]] = indices[glid];
    rays_out[addr[thid]] = rays[glid];
    col_out[addr[thid]] = col[glid];
  }
  __syncthreads();

  // out
  indices[glid] = indices_out[thid];
  rays[glid] = rays_out[thid];
  col[glid] = col_out[thid];
}

__global__ void accumColorKernel(
  uint* pbo_out,
  const uint size,
  int* indices,
  glm::vec3* col,
  glm::vec3* film, const float filmIters)
{
  uint idx = blockIdx.x*blockDim.x + threadIdx.x;

#ifndef STREAM_COMPACT
  film[idx] += col[idx];
#else
  // scatter write
  if (indices[idx]>=0 && idx < size)
    film[indices[idx]] += col[idx];
#endif

#ifdef GAMMA_CORRECT
  pbo_out[idx] = rgbToInt( glm::pow(film[idx]/filmIters, glm::vec3(1.0f/2.2f)) );
#else
  pbo_out[idx] = rgbToInt(film[idx]/filmIters);
#endif
  //pbo_out[idx] = rgbToInt(col[idx]);
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
  curandSetPseudoRandomGeneratorSeed(gen, 100.0f);
  curandGenerateUniform(gen, (float*)rand_d, 3*pixSize);
  
  uint blockSize = BLOCK_SIZE;
  uint gridSize = pixSize/blockSize + (pixSize%blockSize==0 ? 0:1);


  // INIT BUFFERS

  initBuffersKernel<<<gridSize, blockSize>>>(
    w,h,campos,A,B,C,lensRadius,focalDist,rand_d,rays_d,col_d,idx_d,film_d,filmIters
  );


  // PATH TRACE
#ifndef STREAM_COMPACT
  for (int i=1; i<PATH_DEPTH; ++i) {
    calcColorKernel<<<gridSize, blockSize>>>(
      pixSize,time,scene_d,sceneSize,rand_d,rays_d,col_d,idx_d,i
    );
  }
#else
  // first time
  calcColorKernel<<<gridSize, blockSize>>>(
    pixSize,time,scene_d,sceneSize,rand_d,rays_d,col_d,idx_d,0
  );

  uint prevGridSize = gridSize;
  uint prevSize = pixSize;

  thrust::device_ptr<int> idx_ptr(idx_d);
  thrust::device_ptr<Ray::Ray> rays_ptr(rays_d);
  thrust::device_ptr<glm::vec3> col_ptr(col_d);

  for (int i=1; i<2; ++i) {
    // get compacted size
    uint currSize = thrust::count_if(
      idx_ptr, idx_ptr+prevSize,
      path_alive<int>());
    
    uint currGridSize = currSize/blockSize + (currSize%blockSize==0 ? 0:1);

    //// compact buffers
    //compactKernel<<<prevGridSize, blockSize>>>(
    //  prevSize,
    //  idx_d,
    //  rays_d,
    //  col_d
    //);

    // compact buffers
    thrust::copy_if(
      rays_ptr, rays_ptr+prevSize, idx_ptr, rays_ptr, 
      path_alive<int>());
    thrust::copy_if(
      col_ptr, col_ptr+prevSize, idx_ptr, col_ptr, 
      path_alive<int>());
    thrust::copy_if(
      idx_ptr, idx_ptr+prevSize, idx_ptr,
      path_alive<int>());

    //// path trace with temp compacted buffers
    //calcColorKernel<<<currGridSize, blockSize>>>(
    //  currSize,time,scene_d,sceneSize,rand_d,
    //  rays_d,
    //  col_d,
    //  idx_d,
    //  i
    //);

    prevSize = currSize;
    prevGridSize = currGridSize;
  }
#endif

  // ACCUM OUTPUT

  // MISSING: write non-compacted to col
  // need another col_out buffer, pre film
  accumColorKernel<<<gridSize, blockSize>>>(
    pbo_out,prevSize,idx_d,col_d,film_d,filmIters);

  //testRand<<<gridSize, blockSize>>>(pbo_out,rand_d);

  //cudaFree(idx_d);
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