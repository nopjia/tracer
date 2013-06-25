#include "Utils.h"
#include "common.h"
#include "FullScreenQuad.h"
#include "Camera.h"
#include "Object.inl"
#include "Mesh.h"

namespace {
  int mouseX, mouseY;
  int mouseButtons = 0;   // 0x1 left, 0x2 middle, 0x4 right
  float timer = 0.0f;

  uint image_width = WINDOW_W;
  uint image_height = WINDOW_H;

  GLuint pbo;               // pbo for CUDA and openGL
  GLuint result_texture;    // render result copied to this openGL texture
  FullScreenQuad fullScreenQuad;
  ThirdPersonCamera camera;

  std::vector<Object::Object*> scene;
  Object::Object* scene_d;  // pointer to device
}

// global methods
void initGL();
void initCUDA (int argc, char **argv);
void initPBO();
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
  uint *pbo_out, uint w, uint h,
  glm::vec3 campos, glm::vec3 A, glm::vec3 B, glm::vec3 C,
  Object::Object* scene_data, float time);

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
  loadScene();
  loadSceneCUDA();

  glutMainLoop();
  cudaThreadExit();
  
  return 0;
}

void initGL() {  
  std::cout << "OpenGL " << glGetString(GL_VERSION) 
    << "\nGLSL " << glGetString(GL_SHADING_LANGUAGE_VERSION)
    << std::endl;

  glClearColor(0.25f, 0.25f, 0.25f, 1.0f);

  // back face culling
  //glEnable(GL_CULL_FACE);
  //glCullFace(GL_BACK);
  //glFrontFace(GL_CCW);

  // depth testing
  glDisable(GL_DEPTH_TEST);
  //glEnable(GL_DEPTH_TEST);
  //glDepthMask(GL_TRUE);
  //glDepthFunc(GL_LEQUAL);
  //glDepthRange(0.0f, 1.0f);  
  //glClearDepth(1.0f);

  //glEnable(GL_TEXTURE_2D);
  //glDisable(GL_LIGHTING);
  
  fullScreenQuad.begin();
  camera.setAspectRatio(WINDOW_W, WINDOW_H);
  camera.zoom(-5.0f);
}

void resize(int width, int height) {
  glViewport(0, 0, width, height);
  camera.setAspectRatio(WINDOW_W, WINDOW_H);
}

void draw() {
  timer += DELTA_T;

  glClear(GL_COLOR_BUFFER_BIT);

  camera.update();

  raytrace();
  fullScreenQuad.display();

  glutSwapBuffers();
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
    const float FACTOR = 0.1f;
    camera.rotate(FACTOR*dx, FACTOR*dy);
  }
  else if (mouseButtons & 0x4) {
    const float FACTOR = 0.05f;
    camera.zoom(FACTOR*dy);
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

  // cuda call
  unsigned int* out_data;
	checkCudaErrors(cudaGLMapBufferObject((void**)&out_data, pbo));
  
  raytrace(out_data, image_width, image_height,
    camera.getPosition(),A,B,C,
    scene_d,
    timer);

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
  Mesh::Mesh* decaMesh = Mesh::loadObj("data/dodecahedron.obj");
  Object::Object* decaObj = Object::newObject(decaMesh);
  scene.push_back(decaObj);
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

    size_t facesMemSize = obj.m_mesh->m_numFaces*sizeof(Mesh::Triangle);
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