#include "Renderer.h"

//---------------------------------------------------------
// CUDA Declaration
//---------------------------------------------------------

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
  glm::vec3* film_d, const uint filmIters);

extern "C"
void raytrace1(
  uint* pbo_out, const uint w, const uint h, const float time,
  const glm::vec3& campos, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C,
  const Object::Object* scene_d, const uint sceneSize);


//---------------------------------------------------------
// Renderer Class
//---------------------------------------------------------

Renderer::Renderer(uint w, uint h) : 
  filmIters(0),
  mode(RAYTRACE)
{
  image_width = w;
  image_height = h;
}

Renderer::~Renderer() {
  freeCUDAMemory();
}

void Renderer::init() {
  fullScreenQuad.begin();
  initPBO();
  initCUDAMemory();
}

void Renderer::initPBO() {
  // initialize the PBO for transferring data from CUDA to openGL
  uint num_texels = image_width * image_height;
  uint size_tex_data = sizeof(GLubyte) * num_texels * 4;
  void *data = malloc(size_tex_data);

  // test init buffer
  for (int i=0; i<size_tex_data; i+=4) {
    uchar *datam = (uchar*)data;
    datam[i+0] = 0;
    datam[i+1] = 0;
    datam[i+2] = 255.0 * i / (float)size_tex_data;
    datam[i+3] = 255;
  }

  // create buffer object
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_ARRAY_BUFFER, pbo);
  glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
  free(data);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // register this buffer object with CUDA
  checkCudaErrors(cudaGLRegisterBufferObject(pbo));
  SDK_CHECK_ERROR_GL();

  // create the texture that we use to visualize the ray-tracing result
  glActiveTexture(GL_TEXTURE0 + RENDER_TEXTURE);
  glGenTextures(1, &result_texture);
  glBindTexture(GL_TEXTURE_2D, result_texture);

  // set basic parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // buffer data
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  SDK_CHECK_ERROR_GL();

  // unbind
  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0 + UNUSED_TEXTURE);
}

void Renderer::initCUDAMemory() {
  cudaMalloc(&rays_d, image_width*image_height*sizeof(Ray::Ray));
  cudaMalloc(&col_d, image_width*image_height*sizeof(glm::vec3));
  cudaMalloc(&film_d, image_width*image_height*sizeof(glm::vec3));
  cudaMalloc(&rand_d, image_width*image_height*sizeof(glm::vec3));
  cudaMalloc(&idx_d, image_width*image_height*sizeof(int));
}

void Renderer::freeCUDAMemory() {
  cudaFree(rays_d);
  cudaFree(col_d);
  cudaFree(film_d);
  cudaFree(rand_d);
  cudaFree(idx_d);
  
  // TODO: free scene_d
}

void Renderer::initScene(Object::Object* scene, uint size) {
  size_t meshMemSize = sizeof(Mesh::Mesh);
  size_t objectMemSize = sizeof(Object::Object);

  size_t sceneMemSize = size*objectMemSize;
  Object::Object* scene_hd = (Object::Object*)malloc(sceneMemSize);

  sceneSize = size;

  for (int i=0; i<size; ++i) {
    Object::Object& obj = scene[i];

    memcpy(&scene_hd[i], &obj, objectMemSize);

    Mesh::Mesh* mesh_hd = (Mesh::Mesh*)malloc(meshMemSize);
    memcpy(mesh_hd, obj.m_mesh, meshMemSize);

    size_t vertsMemSize = obj.m_mesh->m_numVerts*sizeof(glm::vec3);
    cudaMalloc(&mesh_hd->m_verts, vertsMemSize);
    cudaMemcpy(mesh_hd->m_verts, obj.m_mesh->m_verts, vertsMemSize, cudaMemcpyHostToDevice);
    
    size_t normsMemSize = obj.m_mesh->m_numNorms*sizeof(glm::vec3);
    cudaMalloc(&mesh_hd->m_norms, normsMemSize);
    cudaMemcpy(mesh_hd->m_norms, obj.m_mesh->m_norms, normsMemSize, cudaMemcpyHostToDevice);

    size_t facesMemSize = obj.m_mesh->m_numFaces*sizeof(Mesh::Face);
    cudaMalloc(&mesh_hd->m_faces, facesMemSize);
    cudaMemcpy(mesh_hd->m_faces, obj.m_mesh->m_faces, facesMemSize, cudaMemcpyHostToDevice);

    cudaMalloc(&scene_hd[i].m_mesh, meshMemSize);
    cudaMemcpy(scene_hd[i].m_mesh, mesh_hd, meshMemSize, cudaMemcpyHostToDevice);

    free(mesh_hd);
  }

  cudaMalloc(&scene_d, sceneMemSize);
  cudaMemcpy(scene_d, scene_hd, sceneMemSize, cudaMemcpyHostToDevice);

  free(scene_hd);
}

void Renderer::render(const Camera& camera, float time) {

	// calc cam vars
  glm::vec3 A,B,C;
  {
    // camera ray
    C = glm::normalize(camera.getLookAt()-camera.getPosition());

    // calc A (screen x)
    // calc B (screen y) then scale down relative to aspect
    // fov is for screen x axis
    A = glm::normalize(glm::cross(C,camera.getUp()));
    B = 1.0f/camera.getAspect()*glm::normalize(glm::cross(A,C));

    // scale by FOV
    float tanFOV = tan(glm::radians(camera.getFOV()));
    A *= tanFOV;
    B *= tanFOV;
  }

  // cuda call
  unsigned int* out_data;
	checkCudaErrors(cudaGLMapBufferObject((void**)&out_data, pbo));
  
  if (mode == RAYTRACE) {
    raytrace1(out_data, image_width, image_height, time,
      camera.getPosition(), A, B, C,
      scene_d, sceneSize);
  }
  else if (mode == PATHTRACE) {
    ++filmIters;
    filmIters = 1;

    pathtrace(out_data, image_width, image_height, time,
      camera.getPosition(), A, B, C,
      camera.m_lensRadius, camera.m_focalDist,
      scene_d, sceneSize,
      rand_d, rays_d, col_d, idx_d,
      film_d, filmIters);
  }

	checkCudaErrors(cudaGLUnmapBufferObject(pbo));

	// download texture from destination PBO
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glActiveTexture(GL_TEXTURE0 + RENDER_TEXTURE);
	glBindTexture(GL_TEXTURE_2D, result_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glActiveTexture(GL_TEXTURE0 + UNUSED_TEXTURE);

	SDK_CHECK_ERROR_GL();

  
  fullScreenQuad.display();
}

void Renderer::resetFilm() {
  filmIters = 0;
}

uint Renderer::getIterations() {
  return filmIters;
}

void Renderer::setMode(Mode m) {
  mode = m;
}