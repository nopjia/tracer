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
  int clickedObjID = -1;
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
      printf("focalDist: %.2f\n", camera.m_focalDist);
      break;
    case('F') : 
      if (camera.m_focalDist > 0.0f) {
        camera.m_focalDist -= 0.1f;
        renderer.resetFilm();
        printf("focalDist: %.2f\n", camera.m_focalDist);
      }
      break;
    case('g') : 
      camera.m_lensRadius += 0.1f; 
      renderer.resetFilm();
      printf("focalRad:  %.2f\n", camera.m_lensRadius);
      break;
    case('G') : 
      if (camera.m_lensRadius > 0.0f) {
        camera.m_lensRadius -= 0.1f;
        renderer.resetFilm();
        printf("focalRad:  %.2f\n", camera.m_lensRadius);
      }
      break;
  }
}

void mouse(int button, int state, int x, int y) {
  //printf("click %x\n", button);

  // set/clear bits
  if (state == GLUT_DOWN) {
    mouseButtons |= 0x1 << button;
  } else if (state == GLUT_UP) {
    mouseButtons &= ~(0x1 << button);
  }

  // pick
  if (state == GLUT_DOWN &&
    glutGetModifiers()==GLUT_ACTIVE_CTRL) {
    Ray::Ray ray;
    {
      glm::vec3 A,B,C;

      // camera ray
      C = glm::normalize(camera.getLookAt()-camera.getPosition());
      A = glm::normalize(glm::cross(C,camera.getUp()));
      B = 1.0f/camera.getAspect()*glm::normalize(glm::cross(A,C));

      // scale by FOV
      float tanFOV = tan(glm::radians(camera.getFOV()));
      A *= tanFOV;
      B *= tanFOV;

      glm::vec2 uv( (float)x/WINDOW_W, (float)(WINDOW_H-y)/WINDOW_H );

      ray.m_pos = camera.getPosition()+C + (2.0f*uv.x-1.0f)*A + (2.0f*uv.y-1.0f)*B;
      ray.m_dir = glm::normalize(ray.m_pos-camera.getPosition());
    }

    Ray::Hit hit = Ray::intersectScene(ray, scene.data(), scene.size());
    if (hit.m_id >= 0) {
      printf("clicked obj %i\n", hit.m_id);
      clickedObjID = hit.m_id;
    }

  }

  mouseX = x;
  mouseY = y;
}

void motion(int x, int y) {
  float dx, dy;
  dx = (float)(x - mouseX);
  dy = (float)(y - mouseY);
  
  if (mouseButtons & 0x1) {
    if (glutGetModifiers()==GLUT_ACTIVE_CTRL) {
      const float FACTOR = 0.025f;
      glm::vec3& trans = 
        FACTOR*dx * glm::normalize(glm::cross(camera.getLookAt()-camera.getPosition(), camera.getUp()))
        - FACTOR*dy * camera.getUp();
      Object::translate(scene[clickedObjID], trans);
      renderer.updateScene(clickedObjID, scene[clickedObjID]);
    }
    else {
      const float FACTOR = -0.05f;
      camera.rotate(FACTOR*dx, FACTOR*dy);
    }

    renderer.resetFilm();
  }
  else if (mouseButtons & 0x2) {
    if (glutGetModifiers()==GLUT_ACTIVE_CTRL) {
      const float FACTOR = 1.0f;
      glm::vec3& xAxis = glm::normalize(glm::cross(camera.getLookAt()-camera.getPosition(), camera.getUp()));
      glm::vec3& yAxis = camera.getUp();
      glm::quat rot = glm::angleAxis(FACTOR*dy, xAxis);
      rot = rot * glm::angleAxis(FACTOR*dx, yAxis);
      Object::rotateIsolate(scene[clickedObjID], rot);
      renderer.updateScene(clickedObjID, scene[clickedObjID]);
    }
    else {
      const float FACTOR = 0.05f;
      camera.pan(-FACTOR*dx, FACTOR*dy);
    }

    renderer.resetFilm();
  }
  else if (mouseButtons & 0x4) {
    if (glutGetModifiers()==GLUT_ACTIVE_CTRL) {
      const float FACTOR = 0.01f;
      Object::scale(scene[clickedObjID], 1.0f+FACTOR*dy);
      renderer.updateScene(clickedObjID, scene[clickedObjID]);
    }
    else {
      const float FACTOR = 0.05f;
      camera.zoom(FACTOR*dy);
    }

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
  camera.zoom(-13.0f);
  camera.m_focalDist = 13.0f;
  camera.update();
  
  const glm::vec3 BOX_HDIM (5.0f);

  Object::Object* obj;
  //Mesh::Mesh* planemesh = Mesh::loadObj("data/unitplane.obj");
  Mesh::Mesh* planemesh = Mesh::newGeometry(Mesh::PLANE);
  
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
  //obj->m_material.m_emit = 1.0f;
  scene.push_back(*obj);
  // back -z
  obj = Object::newObject(planemesh);
  Object::rotate(*obj, glm::angleAxis(90.0f, glm::vec3(1.0f, 0.0f, 0.0f)));
  Object::scale(*obj, BOX_HDIM*2.0f);
  Object::translate(*obj, glm::vec3(0.0f, 0.0f, -BOX_HDIM.z));
  obj->m_material.m_color = glm::vec3(1.0f);
  scene.push_back(*obj);
  //// front +z
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
  //obj->m_material.m_color = glm::vec3(0.5f, 1.0f, 0.5f);
  obj->m_material.m_color = glm::vec3(0.5f, 0.5f, 1.0f);
  scene.push_back(*obj);
  // left -x
  obj = Object::newObject(planemesh);
  Object::rotate(*obj, glm::angleAxis(-90.0f, glm::vec3(0.0f, 0.0f, 1.0f)));
  Object::scale(*obj, BOX_HDIM*2.0f);
  Object::translate(*obj, glm::vec3(-BOX_HDIM.x, 0.0f, 0.0f));
  obj->m_material.m_color = glm::vec3(1.0f, 0.5f, 0.5f);
  scene.push_back(*obj);
  
  // ceiling light
  obj = Object::newObject(Mesh::newGeometry(Mesh::CUBE));
  Object::scale(*obj, glm::vec3(BOX_HDIM.x, 0.5f, BOX_HDIM.z));
  Object::translate(*obj, glm::vec3(0.0f, BOX_HDIM.y, 0.0f));
  obj->m_material.m_color = glm::vec3(1.0f, 1.0f, 1.0f);
  obj->m_material.m_emit = 2.0f;
  scene.push_back(*obj);

  // 1
  ////obj = Object::newObject(Mesh::loadObj("data/icosahedron.obj"));
  //obj = Object::newObject(Mesh::newGeometry(Mesh::CUBE));
  ////Object::scale(*obj, glm::vec3(4.0f,0.5f,4.0f));
  //Object::scale(*obj, 2.0f);  
  ////Object::rotate(*obj, glm::angleAxis(-10.0f, glm::vec3(1.0f, 0.0f, 0.0f)));
  ////Object::rotate(*obj, glm::angleAxis(25.0f, glm::vec3(0.0f, 1.0f, 0.0f)));
  //Object::translate(*obj, glm::vec3(2.0f, -2.0f, 1.0f));
  //obj->m_material.m_color = glm::vec3(1.0f, 1.0f, 0.0f);
  ////obj->m_material.m_type = Material::TRANS;
  //obj->m_material.m_n = 1.8f;
  //scene.push_back(*obj);

  // 2
  //obj = Object::newObject(Mesh::newGeometry(Mesh::SPHERE));
  //Object::scale(*obj, 3.0f);
  //Object::translate(*obj, glm::vec3(-1.5f, -3.0f, -2.0f));
  //obj->m_material.m_color = glm::vec3(1.0f, 1.0f, 0.8f);
  //obj->m_material.m_type = Material::MIRR;
  //obj->m_material.m_n = 1.8f;
  //scene.push_back(*obj);
  //obj = Object::newObject(Mesh::newGeometry(Mesh::CUBE));
  //Object::scale(*obj, glm::vec3(2.0f, 3.0f, 2.0f));
  //Object::rotate(*obj, glm::angleAxis(30.0f, glm::vec3(0.0f, 1.0f, 0.0f)));
  //Object::translate(*obj, glm::vec3(2.0f, -3.0f, 1.0f));
  //obj->m_material.m_color = glm::vec3(0.9f, 1.0f, 0.9f);
  //obj->m_material.m_type = Material::TRANS;
  //obj->m_material.m_n = 1.2f;
  //scene.push_back(*obj);

  // CORNELL BOX SPHERES
  obj = Object::newObject(Mesh::newGeometry(Mesh::SPHERE));
  Object::scale(*obj, 4.0f);
  Object::translate(*obj, glm::vec3(-2.5f, -3.0f, -3.0f));
  //obj->m_material.m_color = glm::vec3(1.0f, 0.0f, 1.0f);
  obj->m_material.m_color = glm::vec3(1.0f, 1.0f, 1.0f);
  obj->m_material.m_type = Material::MIRR;
  scene.push_back(*obj);

  obj = Object::newObject(Mesh::newGeometry(Mesh::SPHERE));
  Object::scale(*obj, 4.0f);
  Object::translate(*obj, glm::vec3(2.5f, -3.0f, 0.0f));
  //Object::translate(*obj, glm::vec3(2.5f, -3.0f, 1.5f));
  obj->m_material.m_color = glm::vec3(1.0f, 1.0f, 1.0f);
  obj->m_material.m_type = Material::TRANS;
  obj->m_material.m_n = 1.6f;
  scene.push_back(*obj);

  obj = Object::newObject(Mesh::loadObj("data/icosahedron.obj"));
  //obj = Object::newObject(Mesh::newGeometry(Mesh::CUBE));
  Object::scale(*obj, 1.5f);
  //Object::rotate(*obj, glm::angleAxis(45.0f, glm::vec3(0.57735f)));
  Object::rotate(*obj, glm::angleAxis(-10.0f, glm::vec3(1.0f, 0.0f, 0.0f)));
  Object::rotate(*obj, glm::angleAxis(25.0f, glm::vec3(0.0f, 1.0f, 0.0f)));
  Object::translate(*obj, glm::vec3(-1.0f, -3.7f, 3.0f));
  obj->m_material.m_color = glm::vec3(1.0f, 1.0f, 0.0f);
  scene.push_back(*obj);
}