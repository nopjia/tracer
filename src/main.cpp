#include <iostream>
#include <gl3w.h>
#include <freeglut.h>


// constants
#define WINDOW_W 640
#define WINDOW_H 480


// display callback
void display() {
  glClear(GL_COLOR_BUFFER_BIT);
  glutSwapBuffers();
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
  glutCreateWindow("OpenGL Test");

  if (gl3wInit()) {
    std::cerr << "Failed to initialize." << std::endl;
    return -1;
  }
  if (!gl3wIsSupported(4, 2)) {
    std::cerr << "OpenGL 4.2 not supported" << std::endl;
    return -1;
  }

  std::cout << "OpenGL " << glGetString(GL_VERSION) 
    << "\nGLSL " << glGetString(GL_SHADING_LANGUAGE_VERSION);
  glClearColor(0,0,0,0);
  glutDisplayFunc(display);
  glutMainLoop();
  return 0;
}

