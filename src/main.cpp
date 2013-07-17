#include "common.h"
#include "Renderer.h"
#include "Camera.h"
#include "Object.inl"
#include "Mesh.h"
#include "Ray.inl"

//---------------------------------------------------------
// DECLARATIONS
//---------------------------------------------------------

namespace {
  int mouseX, mouseY;
  int mouseButtons = 0;   // 0x1 left, 0x2 middle, 0x4 right
  float timer = 0.0f;
  uint frameCount = 0, timeBase = 0;
    
  ThirdPersonCamera camera;
  Renderer renderer(WINDOW_W/PIXSCALE, WINDOW_H/PIXSCALE);

  std::vector<Object::Object> scene;
}

void initGL();
void initCUDA (int argc, char **argv);
void initScene();
void resize(int width, int height);
void draw();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

//---------------------------------------------------------
// INITIALIZE
//---------------------------------------------------------

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

  initScene();

  renderer.init();
  renderer.initScene(scene.data(), scene.size());

  glutMainLoop();
  cudaThreadExit();

  std::cout << "Exit Program" << std::endl;
  
  return 0;
}

void initGL() {  
  std::cout << "OpenGL " << glGetString(GL_VERSION) 
    << "\nGLSL " << glGetString(GL_SHADING_LANGUAGE_VERSION)
    << std::endl;
  glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
  glDisable(GL_DEPTH_TEST);
}

void initCUDA (int argc, char **argv) {
  if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    gpuGLDeviceInit(argc, (const char **)argv);
  else
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
}

//---------------------------------------------------------
// WINDOW CALLBACKS
//---------------------------------------------------------

void resize(int width, int height) {
  glViewport(0, 0, width, height);
  camera.setAspect(WINDOW_W, WINDOW_H);
}

void getFPS() {
  ++frameCount;
  uint currTime = glutGet(GLUT_ELAPSED_TIME);

  uint elapsed = currTime - timeBase;
  if (elapsed > 1000) {
    float fps = frameCount*1000.0f/elapsed;
    timeBase = currTime;
    frameCount = 0;

    char buffer[32];
    sprintf(buffer, "%.4f : %u", fps, renderer.getIterations());
    glutSetWindowTitle(buffer);
  }
}

void draw() {
  timer += DELTA_T;

  glClear(GL_COLOR_BUFFER_BIT);

  camera.update();
  renderer.render(camera, timer);

  glutSwapBuffers();
  getFPS();
}

void keyboard(unsigned char key, int x, int y) {
  switch(key) {
    case(27) : exit(0); break;
    case('1') : renderer.setMode(Renderer::RAYTRACE); break;
    case('2') : renderer.setMode(Renderer::PATHTRACE); break;
    case('f') : 
      camera.m_focalDist += 0.1f; 
      renderer.resetFilm();
      break;
    case('F') : 
      if (camera.m_focalDist > 0.0f) {
        camera.m_focalDist -= 0.1f;
        renderer.resetFilm();
      }
      break;
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
    renderer.resetFilm();
  }
  else if (mouseButtons & 0x2) {
    const float FACTOR = 0.05f;
    camera.pan(-FACTOR*dx, FACTOR*dy);
    renderer.resetFilm();
  }
  else if (mouseButtons & 0x4) {
    const float FACTOR = 0.05f;
    camera.zoom(FACTOR*dy);
    renderer.resetFilm();
  }

  mouseX = x;
  mouseY = y;
}

//---------------------------------------------------------
// Scene
//---------------------------------------------------------

void initScene() {  
  camera.setFOV(FOV);
  camera.setAspect(WINDOW_W, WINDOW_H);
  camera.zoom(-10.0f);
  camera.update();
  
  const glm::vec3 BOX_HDIM (5.0f);

  Object::Object* obj;

  Mesh::Mesh* planemesh = Mesh::loadObj("data/unitplane.obj");
  // bottom -y
  obj = Object::newObject(planemesh);
  Object::scale(*obj, BOX_HDIM*2.0f);
  Object::translate(*obj, glm::vec3(0.0f, -BOX_HDIM.y, 0.0f));
  obj->m_material.m_color = glm::vec3(1.0f);
  scene.push_back(*obj);
  // top +y
  obj = Object::newObject(planemesh);
  Object::rotate(*obj, glm::angleAxis(180.0f, glm::vec3(1.0f, 0.0f, 0.0f)));
  Object::scale(*obj, BOX_HDIM*2.0f);
  Object::translate(*obj, glm::vec3(0.0f, BOX_HDIM.y, 0.0f));
  obj->m_material.m_color = glm::vec3(1.0f, 1.0f, 1.0f);
  obj->m_material.m_emit = 2.0f;
  scene.push_back(*obj);
  // back -z
  obj = Object::newObject(planemesh);
  Object::rotate(*obj, glm::angleAxis(90.0f, glm::vec3(1.0f, 0.0f, 0.0f)));
  Object::scale(*obj, BOX_HDIM*2.0f);
  Object::translate(*obj, glm::vec3(0.0f, 0.0f, -BOX_HDIM.z));
  obj->m_material.m_color = glm::vec3(1.0f);
  scene.push_back(*obj);
  // front +z
  //obj = Object::newObject(planemesh);
  //Object::rotate(*obj, glm::angleAxis(-90.0f, glm::vec3(1.0f, 0.0f, 0.0f)));
  //Object::scale(*obj, BOX_HDIM*2.0f);
  //Object::translate(*obj, glm::vec3(0.0f, 0.0f, BOX_HDIM.z));
  //obj->m_material.m_color = glm::vec3(1.0f);
  //scene.push_back(*obj);
  // right +x
  obj = Object::newObject(planemesh);
  Object::rotate(*obj, glm::angleAxis(90.0f, glm::vec3(0.0f, 0.0f, 1.0f)));
  Object::scale(*obj, BOX_HDIM*2.0f);
  Object::translate(*obj, glm::vec3(BOX_HDIM.x, 0.0f, 0.0f));
  obj->m_material.m_color = glm::vec3(0.0f, 0.0f, 1.0f);
  scene.push_back(*obj);
  // left -x
  obj = Object::newObject(planemesh);
  Object::rotate(*obj, glm::angleAxis(-90.0f, glm::vec3(0.0f, 0.0f, 1.0f)));
  Object::scale(*obj, BOX_HDIM*2.0f);
  Object::translate(*obj, glm::vec3(-BOX_HDIM.x, 0.0f, 0.0f));
  obj->m_material.m_color = glm::vec3(1.0f, 0.0f, 0.0f);
  scene.push_back(*obj);
  
  //// ceiling light
  //obj = Object::newObject(Mesh::newGeometry(Mesh::CUBE));
  //Object::scale(*obj, glm::vec3(BOX_HDIM.x, 1.0f, BOX_HDIM.z));
  //Object::translate(*obj, glm::vec3(0.0f, BOX_HDIM.y, 0.0f));
  //obj->m_material.m_color = glm::vec3(1.0f, 1.0f, 1.0f);
  //obj->m_material.m_emit = 5.0f;
  //scene.push_back(*obj);

  ////obj = Object::newObject(Mesh::loadObj("data/icosahedron.obj"));
  ////obj = Object::newObject(Mesh::loadObj("data/unitcube.obj"));
  //obj = Object::newObject(Mesh::newGeometry(Mesh::CUBE));
  //Object::scale(*obj, glm::vec3(4.0f,4.0f,1.0f));
  ////Object::scale(*obj, 2.0f);
  //obj->m_material.m_color = glm::vec3(1.0f, 1.0f, 0.8f);
  //obj->m_material.m_type = Material::TRANS;
  //obj->m_material.m_n = 1.4f;
  //scene.push_back(*obj);

  //obj = Object::newObject(Mesh::newGeometry(Mesh::SPHERE));
  //Object::scale(*obj, 3.0f);
  //Object::translate(*obj, glm::vec3(-1.5f, -3.0f, -2.0f));
  //obj->m_material.m_color = glm::vec3(1.0f, 1.0f, 0.8f);
  //obj->m_material.m_type = Material::TRANS;
  //obj->m_material.m_n = 1.8f;
  //scene.push_back(*obj);
  //obj = Object::newObject(Mesh::newGeometry(Mesh::CUBE));
  //Object::scale(*obj, glm::vec3(2.0f, 3.0f, 2.0f));
  //Object::rotate(*obj, glm::angleAxis(30.0f, glm::vec3(0.0f, 1.0f, 0.0f)));
  //Object::translate(*obj, glm::vec3(2.0f, -3.0f, 1.0f));
  //obj->m_material.m_color = glm::vec3(0.9f, 1.0f, 0.9f);
  //obj->m_material.m_type = Material::TRANS;
  //obj->m_material.m_n = 1.1f;
  //scene.push_back(*obj);
}