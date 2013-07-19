#ifndef RENDERER_H
#define RENDERER_H

#include "common.h"
#include "FullScreenQuad.h"
#include "Camera.h"
#include "Object.inl"
#include "Ray.inl"

class Renderer 
{
public:
  enum Mode {RAYTRACE, PATHTRACE};

  Renderer(uint w, uint h);
  ~Renderer();

  void init();
  void initScene(Object::Object* scene, uint size);
  void render(const Camera& camera, float time);

  void updateScene(int index, const Object::Object& obj);
  void setMode(Mode m);
  void resetFilm();
  uint getIterations();  

private:
  void initPBO();
  void initCUDAMemory();
  void raytrace();
  void freeCUDAMemory();

  uint image_width;
  uint image_height;

  // GL stuff
  GLuint pbo;               // pbo for CUDA and openGL
  GLuint result_texture;    // render result copied to this openGL texture
  FullScreenQuad fullScreenQuad;

  // cuda buffers
  Object::Object* scene_d;  // pointer to device
  Ray::Ray* rays_d;
  glm::vec3* col_d;
  glm::vec3* film_d;
  glm::vec3* rand_d;
  int* idx_d;
  
  // cuda inbetween buffers
  Object::Object* scene_hd;
  
  uint sceneSize;
  uint filmIters;
  Mode mode;
};

#endif  // RENDERER_H