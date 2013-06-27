#include "Utils.h"
#include "common.h"
#include "FullScreenQuad.h"
#include "Camera.h"
#include "Object.inl"
#include "Mesh.h"
#include "Ray.inl"

namespace {
  uint image_width = WINDOW_W / PIXSCALE;
  uint image_height = WINDOW_H / PIXSCALE;
  int mouseX, mouseY;
  int mouseButtons = 0;   // 0x1 left, 0x2 middle, 0x4 right
  float timer = 0.0f;
  uint frameCount = 0, timeBase = 0;

  GLuint pbo;               // pbo for CUDA and openGL
  GLuint result_texture;    // render result copied to this openGL texture
  FullScreenQuad fullScreenQuad;
  ThirdPersonCamera camera;

  std::vector<Object::Object*> scene;
  Object::Object* scene_d;  // pointer to device
  Ray::Ray* rays_d;
  glm::vec3* col_d;
  glm::vec3* film_d;
  uint filmIters = 0;
  bool moved = false;
  glm::vec3* rand_d;
  uint* flags_d;
}

// global methods
void initGL();
void initCUDA (int argc, char **argv);
void initPBO();
void initMemoryCUDA();
void loadScene();
void loadSceneCUDA();
void raytrace();
void resize(int width, int height);
void draw();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

extern "C"
void raytrace(
  uint* pbo_out, const uint w, const uint h, const float time,
  const glm::vec3& campos, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C,
  const Object::Object* scene_d, const uint sceneSize,
  glm::vec3* rand_d,
  uint* flags_d,
  Ray::Ray* rays_d,
  glm::vec3* col_d,
  glm::vec3* film_d, const uint filmIters);

void mytest() {
  Ray::Ray ray;
  Ray::Hit hit;

  ray.m_pos = glm::vec3(0.0f, 0.5f, -5.0f);
  ray.m_dir = glm::vec3(0.0f, 0.0f, 1.0f);
  
  //Mesh::Mesh& mesh = *scene.data()[0]->m_mesh;
  //Mesh::Triangle* tris = new Mesh::Triangle[mesh.m_numFaces];
  //for (int i=0; i<mesh.m_numFaces; ++i) {
  //  std::printf("%i f %u %u %u\n", i, mesh.m_faces[i].m_v[0]+1, mesh.m_faces[i].m_v[1]+1, mesh.m_faces[i].m_v[2]+1);
  //  tris[i] = Mesh::getTriangle(mesh, i);
  //}

  hit = Ray::intersect(ray, *scene.data()[1]->m_mesh);  
  //hit = Ray::intersect(ray, *scene[1]);
  //hit = Ray::intersectScene(ray, (const Object::Object**)scene.data(), scene.size());
  hit;
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  // request version 4.2
  glutInitContextVersion(4, 2);
  // core profile 
  glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
  glutInitContextProfile(GLUT_CORE_PROFILE);

  // double buffered, depth, color w/ alpha
  glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(WINDOW_W, WINDOW_H);
  glutCreateWindow("GL Window");
  
  // register callbacks  
  glutDisplayFunc(draw);
  glutIdleFunc(draw);
  glutReshapeFunc(resize);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  //glutSpecialFunc(special);

  if (gl3wInit()) {
    std::cerr << "Failed to initialize." << std::endl;
    return -1;
  }
  if (!gl3wIsSupported(4,2)) {
    std::cerr << "OpenGL 4.2 not supported" << std::endl;
    return -1;
  }
  
  initGL();
  initCUDA(argc, argv);
  initPBO();
  initMemoryCUDA();
  loadScene();
  loadSceneCUDA();

  //mytest();

  glutMainLoop();
  cudaThreadExit();
  
  return 0;
}

void initGL() {  
  std::cout << "OpenGL " << glGetString(GL_VERSION) 
    << "\nGLSL " << glGetString(GL_SHADING_LANGUAGE_VERSION)
    << std::endl;

  glClearColor(0.25f, 0.25f, 0.25f, 1.0f);

  glDisable(GL_DEPTH_TEST);
  
  fullScreenQuad.begin();
  camera.setAspectRatio(WINDOW_W, WINDOW_H);
  camera.zoom(-5.0f);
  camera.update();
}

void resize(int width, int height) {
  glViewport(0, 0, width, height);
  camera.setAspectRatio(WINDOW_W, WINDOW_H);
}

void getFPS() {
  ++frameCount;
  uint currTime = glutGet(GLUT_ELAPSED_TIME);

  uint elapsed = currTime - timeBase;
  if (elapsed > 1000) {
    float fps = frameCount*1000.0f/(elapsed);
    float milisecs = elapsed / frameCount;
    timeBase = currTime;
    frameCount = 0;

    char buffer[32];
    sprintf(buffer, "%.4f : %.0f : %u", fps, milisecs, filmIters);
    glutSetWindowTitle(buffer);
  }

}

void draw() {
  timer += DELTA_T;

  glClear(GL_COLOR_BUFFER_BIT);

  camera.update();

  raytrace();
  fullScreenQuad.display();

  glutSwapBuffers();
  getFPS();
}

void keyboard(unsigned char key, int x, int y) {
  switch(key) {
  case(27) : exit(0);
  }
}

void mouse(int button, int state, int x, int y) {
  printf("click %x\n", button);

  // set/clear bits
  if (state == GLUT_DOWN) {
    mouseButtons |= 0x1 << button;
  } else if (state == GLUT_UP) {
    mouseButtons &= ~(0x1 << button);
  }

  mouseX = x;
  mouseY = y;
}

void motion(int x, int y) {
  float dx, dy;
  dx = (float)(x - mouseX);
  dy = (float)(y - mouseY);
   
  if (mouseButtons & 0x1) {
    const float FACTOR = -0.05f;
    camera.rotate(FACTOR*dx, FACTOR*dy);
    moved = true;
  }
  else if (mouseButtons & 0x2) {
    const float FACTOR = 0.05f;
    camera.pan(-FACTOR*dx, FACTOR*dy);
    moved = true;
  }
  else if (mouseButtons & 0x4) {
    const float FACTOR = 0.05f;
    camera.zoom(FACTOR*dy);
    moved = true;
  }

  mouseX = x;
  mouseY = y;
}

void initCUDA (int argc, char **argv) {
  if (checkCmdLineFlag(argc, (const char **)argv, "device"))
  {
    gpuGLDeviceInit(argc, (const char **)argv);
  }
  else 
  {
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
  }
}

void initPBO() {
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

void raytrace() {

	// calc cam vars
  glm::vec3 A,B,C;
  {
    // camera ray
    C = glm::normalize(camera.getLookAt()-camera.getPosition());

    // calc A (screen x)
    // calc B (screen y) then scale down relative to aspect
    // fov is for screen x axis
    A = glm::normalize(glm::cross(C,camera.getUp()));
    B = glm::float32(-1.0/(camera.getAspect()))*glm::normalize(glm::cross(A,C));

    // scale by FOV
    float tanFOV = tan(glm::radians(camera.getFOV()));
    A *= tanFOV;
    B *= tanFOV;
  }

  // film  
  if (moved) {
    filmIters = 1;
    moved = false;
  }
  else {
    ++filmIters;
  }

  // cuda call
  unsigned int* out_data;
	checkCudaErrors(cudaGLMapBufferObject((void**)&out_data, pbo));
  
  raytrace(out_data, image_width, image_height, timer,
    camera.getPosition(),A,B,C,
    scene_d, scene.size(),
    rand_d, flags_d, rays_d, col_d,
    film_d, filmIters);

	checkCudaErrors(cudaGLUnmapBufferObject(pbo));

	// download texture from destination PBO
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glActiveTexture(GL_TEXTURE0 + RENDER_TEXTURE);
	glBindTexture(GL_TEXTURE_2D, result_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glActiveTexture(GL_TEXTURE0 + UNUSED_TEXTURE);

	SDK_CHECK_ERROR_GL();
}

void loadScene() {
  Object::Object* obj;

  obj = Object::newObject(Mesh::loadObj("data/unitcube.obj"));
  //Object::rotate(*obj, glm::angleAxis(55.0f, glm::vec3(0.707106781186547524400844362104849039, 0.707106781186547524400844362104849039, 0.0f)));
  //Object::scale(*obj, glm::vec3(0.5f, 2.0f, 1.0f));
  //Object::translate(*obj, glm::vec3(4.0f, -2.0f, 1.0f));
  Object::scale(*obj, glm::vec3(5.0f, 0.5f, 5.0f));
  Object::translate(*obj, glm::vec3(0.0f, 5.0f, 0.0f));
  obj->m_material.m_color = glm::vec3(1.0f);
  obj->m_material.m_emit = 10.0f;
  scene.push_back(obj);

  obj = Object::newObject(Mesh::loadObj("data/unitcube_inv.obj"));
  Object::scale(*obj, 10.0f);
  scene.push_back(obj);

  obj = Object::newObject(Mesh::loadObj("data/icosahedron.obj"));  
  Object::translate(*obj, glm::vec3(0.0f, -5.0f, 0.0f));
  obj->m_material.m_color = glm::vec3(1.0, 1.0, 0.0);
  scene.push_back(obj);
}

void initMemoryCUDA() {
  cudaMalloc(&rays_d, image_width*image_height*sizeof(Ray::Ray));
  cudaMalloc(&col_d, image_width*image_height*sizeof(glm::vec3));
  cudaMalloc(&film_d, image_width*image_height*sizeof(glm::vec3));
  cudaMalloc(&rand_d, image_width*image_height*sizeof(glm::vec3));
  cudaMalloc(&flags_d, image_width*image_height*sizeof(uint));
}

void loadSceneCUDA() {
  size_t meshMemSize = sizeof(Mesh::Mesh);
  size_t objectMemSize = sizeof(Object::Object);

  size_t sceneMemSize = scene.size()*objectMemSize;
  Object::Object* scene_hd = (Object::Object*)malloc(sceneMemSize);

  for (int i=0; i<scene.size(); ++i) {
    Object::Object& obj = *(scene[i]);

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