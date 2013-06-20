#include <iostream>
#include <gl3w.h>
#include <freeglut.h>


// constants
#define WINDOW_W 640
#define WINDOW_H 480

namespace {
  int mouseX, mouseY;
  int mouseButtons = 0;   // 0x1 left, 0x2 middle, 0x4 right
}

// global methods
void initGL();
void resize(int width, int height);
void draw();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

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
  
  glutMainLoop();

  return 0;
}

void initGL() {  
  std::cout << "OpenGL " << glGetString(GL_VERSION) 
    << "\nGLSL " << glGetString(GL_SHADING_LANGUAGE_VERSION);
  
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

  glEnable(GL_TEXTURE_2D);

  //glDisable(GL_LIGHTING);
  //glEnable(GL_TEXTURE_2D);
  //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
}

void resize(int width, int height) {  
  glViewport(0, 0, width, height);

  /*
  // reset projection matrix
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // calculate aspect ratio
  gluPerspective(CAM_FOV, (GLfloat)width/(GLfloat)height, CAM_NEAR, CAM_FAR);

  // reset modelview matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  */
}

void draw() {
  glClear(GL_COLOR_BUFFER_BIT);
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

  /*
  if (mouseButtons & 0x1) {
    rotateX += dy * 0.2f;
    rotateY += dx * 0.2f;
  }
  else if (mouseButtons & 0x4) {
    translateZ += dy * 0.01f;
  }
  */

  mouseX = x;
  mouseY = y;
}