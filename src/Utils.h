#ifndef UTILS_H
#define UTILS_H

// GL
#include <gl3w.h>
#include <freeglut.h>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>

// GLM libraries
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_interpolation.hpp>
#include <glm/gtc/half_float.hpp>
#include <glm/gtc/type_precision.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtx/random.hpp>
#include <glm/gtx/quaternion.hpp>

// STL
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <vector>
#include <iterator>
#include <string>
#include <cstring>
#include <map>
#include <set>
#include <utility>
#include <limits>
#include <algorithm>

typedef unsigned char byte;
typedef unsigned char uchar;
typedef unsigned int uint;

const std::string SOURCE_DIRECTORY = std::string("src/");
const std::string DATA_DIRECTORY = std::string("data/");
const std::string SHADER_DIRECTORY = std::string(SOURCE_DIRECTORY + "shaders/");
const std::string SCENE_DIRECTORY = std::string(DATA_DIRECTORY + "scenes/");
const std::string IMAGE_DIRECTORY = std::string(DATA_DIRECTORY + "images/");
const std::string MESH_DIRECTORY = std::string(DATA_DIRECTORY + "meshes/");

namespace Utils
{
  struct Ray
  {
    glm::vec3 position;
    glm::vec3 direction;
    Ray transform(glm::mat4 transformationMatrix)
    {
      Ray transformed;
      transformed.position = glm::vec3(transformationMatrix * glm::vec4(position, 1.0f));
      transformed.direction = glm::normalize(glm::vec3(transformationMatrix * glm::vec4(direction, 0.0)));                
      return transformed;
    }
  };
  
  Ray getPickingRay(int x, int y, int width, int height, float nearPlane, float farPlane, glm::mat4 viewMatrix, glm::mat4 projectionMatrix);
  float rayBoundingBoxIntersect(Ray &r, glm::vec3 min, glm::vec3 max);

  std::string loadFile(std::string const & Filename);
  
  inline int roundToNextPowerOf2(int x)
  {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
  }

  inline int roundToNextMultiple(int numToRound, int multiple)
  {
    return numToRound == 0 ? 0 : ((numToRound - 1) / multiple + 1) * multiple;
  }

  inline void printMatrix(glm::mat4& matrix)
  {
    std::cout << "==========PRINTING MATRIX==========" << std::endl;
    std::cout << matrix[0].x << " " << matrix[1].x << " " << matrix[2].x << " " << matrix[3].x << std::endl;
    std::cout << matrix[0].y << " " << matrix[1].y << " " << matrix[2].y << " " << matrix[3].y << std::endl;
    std::cout << matrix[0].z << " " << matrix[1].z << " " << matrix[2].z << " " << matrix[3].z << std::endl;
    std::cout << matrix[0].w << " " << matrix[1].w << " " << matrix[2].w << " " << matrix[3].w << std::endl;
  }

  inline void printVec3(glm::vec3& vector)
  {
    std::cout << vector[0] << " " << vector[1] << " " << vector[2] << std::endl;
  }

  inline void printVec4(glm::vec4& vector)
  {
    std::cout << vector[0] << " " << vector[1] << " " << vector[2] << " " << vector[3] << std::endl;
  }

  inline void printQuat(glm::fquat& quaternion)
  {
    std::cout << quaternion[0] << " " << quaternion[1] << " " << quaternion[2] << " " << quaternion[3] << std::endl;
  }

  namespace OpenGL
  {
    struct OpenGLTimer
    {
      // To use:
      // 1) timer.begin() to initialize
      // 2) timer.startTimer() before GL command
      // 3) timer.stopTimer() after GL command
      // 4) uint total = timer.getElapsedTime()

      GLuint queryObject;
      uint totalTime;
      void begin()
      {
        glGenQueries(1, &queryObject);  
      }
      void startTimer()
      {
        glBeginQuery(GL_TIME_ELAPSED, queryObject);
      }
      void stopTimer()
      {
        glEndQuery(GL_TIME_ELAPSED);
        glGetQueryObjectuiv(queryObject, GL_QUERY_RESULT, &totalTime);
      }
      uint getElapsedTime()
      {
        return totalTime;
      }
    };

    inline void setRenderState(bool enableCulling, bool enableDepth, bool enableColor)
    {
      if(enableCulling) glEnable(GL_CULL_FACE);
      else glDisable(GL_CULL_FACE);

      if(enableDepth) glEnable(GL_DEPTH_TEST);
      else glDisable(GL_DEPTH_TEST);

      if(enableColor) glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
      else glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);
    }

    inline void clearColor()
    {
      float clearColor[4] = {0.0f,0.0f,0.0f,0.0f};
      glClearBufferfv(GL_COLOR, 0, clearColor);
    }
    inline void clearDepth()
    {
      float clearDepth = 1.0f;
      glClearBufferfv(GL_DEPTH, 0, &clearDepth);
    }
    inline void clearColorAndDepth()
    { 
      clearColor();
      clearDepth();
    }

    bool checkProgram(GLuint ProgramName);
    

    bool checkShader(GLuint ShaderName, const char* Source);

    GLuint createShader(GLenum Type, std::string const & Source);
    
    bool checkError(const char* Title);
  
    // Returns the shader program
    GLuint createShaderProgram(std::string& vertexShader, std::string& fragmentShader);

    bool checkFramebuffer(GLuint FramebufferName);
  
    bool checkExtension(char const * String);
  
    static void APIENTRY debugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, GLvoid* userParam);
  }
}

#endif  // UTILS_H