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
  namespace Math
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
    Ray getPickingRay(int x, int y, int width, int height, float nearPlane, float farPlane, glm::mat4 viewMatrix, glm::mat4 projectionMatrix)
    {
      float winWidth = (float)width;
      float winHeight = (float)height;
      float winX = (float)x;
      float winY = (float)y;
      float winZClose = 0.0f;
      float winZFar = 1.0f;

      //Window to NDC
      glm::vec4 closePoint;
      closePoint.x = 2.0f*(winX/winWidth) - 1.0f;
      closePoint.y = 2.0f*((winHeight - winY)/winHeight) - 1.0f;
      closePoint.z = 2.0f*(winZClose) - 1.0f;
      closePoint.w = 1.0f;

      glm::vec4 farPoint;
      farPoint.x = 2.0f*(winX/winWidth) - 1.0f;
      farPoint.y = 2.0f*((winHeight - winY)/winHeight) - 1.0f;
      farPoint.z = 2.0f*(winZFar) - 1.0f;
      farPoint.w = 1.0f;

      //NDC to clip
      closePoint *= nearPlane;
      farPoint *= farPlane;
      glm::mat4 invProjectionMatrix = glm::inverse(projectionMatrix);

      closePoint = invProjectionMatrix * closePoint;
      farPoint = invProjectionMatrix * farPoint;

      glm::mat4 invViewMatrix = glm::inverse(viewMatrix);
      closePoint = invViewMatrix * closePoint;
      farPoint = invViewMatrix * farPoint;

      Ray r;
      r.position = glm::vec3(closePoint);
      r.direction = glm::normalize(glm::vec3(farPoint-closePoint));
      return r;
    }
    float rayBoundingBoxIntersect(Ray &r, glm::vec3 min, glm::vec3 max)
    {
      glm::vec3 tMin = (min-r.position) / r.direction;
      glm::vec3 tMax = (max-r.position) / r.direction;
      glm::vec3 t1 = glm::min(tMin, tMax);
      glm::vec3 t2 = glm::max(tMin, tMax);
      float tNear = glm::max(glm::max(t1.x, t1.y), t1.z);
      float tFar = glm::min(glm::min(t2.x, t2.y), t2.z);
      float t = -1.0f;
      if (tNear<tFar && tFar>0.0)
        t = tNear > 0.0 ? tNear : tFar;
      return t;
    }
  }

  std::string loadFile(std::string const & Filename)
  {
    std::ifstream stream(Filename.c_str(), std::ios::in);

    if(!stream.is_open())
      return "";

    std::string Line = "";
    std::string Text = "";

    while(getline(stream, Line))
      Text += "\n" + Line;

    stream.close();

    return Text;
  }

  int roundToNextPowerOf2(int x)
  {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
  }

  int roundToNextMultiple(int numToRound, int multiple)
  {
    return numToRound == 0 ? 0 : ((numToRound - 1) / multiple + 1) * multiple;
  }

  void printMatrix(glm::mat4& matrix)
  {
    std::cout << "==========PRINTING MATRIX==========" << std::endl;
    std::cout << matrix[0].x << " " << matrix[1].x << " " << matrix[2].x << " " << matrix[3].x << std::endl;
    std::cout << matrix[0].y << " " << matrix[1].y << " " << matrix[2].y << " " << matrix[3].y << std::endl;
    std::cout << matrix[0].z << " " << matrix[1].z << " " << matrix[2].z << " " << matrix[3].z << std::endl;
    std::cout << matrix[0].w << " " << matrix[1].w << " " << matrix[2].w << " " << matrix[3].w << std::endl;
  }

  void printVec3(glm::vec3& vector)
  {
    std::cout << vector[0] << " " << vector[1] << " " << vector[2] << std::endl;
  }

  void printVec4(glm::vec4& vector)
  {
    std::cout << vector[0] << " " << vector[1] << " " << vector[2] << " " << vector[3] << std::endl;
  }

  void printQuat(glm::fquat& quaternion)
  {
    std::cout << quaternion[0] << " " << quaternion[1] << " " << quaternion[2] << " " << quaternion[3] << std::endl;
  }

  std::vector<std::string> parseSpaceSeparatedString(std::string& configData)
  {
    std::vector<std::string> tokens;
    std::istringstream iss(configData);
    std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(), std::back_inserter<std::vector<std::string> >(tokens));
    return tokens;
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

    void setRenderState(bool enableCulling, bool enableDepth, bool enableColor)
    {
      if(enableCulling) glEnable(GL_CULL_FACE);
      else glDisable(GL_CULL_FACE);

      if(enableDepth) glEnable(GL_DEPTH_TEST);
      else glDisable(GL_DEPTH_TEST);

      if(enableColor) glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
      else glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);
    }

    int screenWidth;
    int screenHeight;

    void setViewport(int width, int height)
    {
      glViewport(0,0,width,height);
    }

    void setScreenSizedViewport()
    {
      glViewport(0, 0, Utils::OpenGL::screenWidth, Utils::OpenGL::screenHeight);
    }

    void setScreenSize(int width, int height)
    {
      Utils::OpenGL::screenWidth = width;
      Utils::OpenGL::screenHeight = height;
      setScreenSizedViewport();
    }

    void clearColor()
    {
      float clearColor[4] = {0.0f,0.0f,0.0f,0.0f};
      glClearBufferfv(GL_COLOR, 0, clearColor);
    }
    void clearDepth()
    {
      float clearDepth = 1.0f;
      glClearBufferfv(GL_DEPTH, 0, &clearDepth);
    }
    void clearColorAndDepth()
    { 
      clearColor();
      clearDepth();
    }

    bool checkProgram(GLuint ProgramName)
    {
      if(!ProgramName)
        return false;

      GLint Result = GL_FALSE;
      glGetProgramiv(ProgramName, GL_LINK_STATUS, &Result);

      //fprintf(stdout, "Linking program\n");
      int InfoLogLength;
      glGetProgramiv(ProgramName, GL_INFO_LOG_LENGTH, &InfoLogLength);
      if(InfoLogLength > 0)
      {
        std::vector<char> Buffer(std::max(InfoLogLength, int(1)));
        glGetProgramInfoLog(ProgramName, InfoLogLength, NULL, &Buffer[0]);
        fprintf(stdout, "%s\n", &Buffer[0]);
      }

      return Result == GL_TRUE;
    }

    bool checkShader(GLuint ShaderName, const char* Source)
    {
      if(!ShaderName)
        return false;

      GLint Result = GL_FALSE;
      glGetShaderiv(ShaderName, GL_COMPILE_STATUS, &Result);

//fprintf(stdout, "Compiling shader\n");
//fprintf(stdout, "Compiling shader\n%s...\n", Source);
      int InfoLogLength;
      glGetShaderiv(ShaderName, GL_INFO_LOG_LENGTH, &InfoLogLength);
      if(InfoLogLength > 0)
      {
        std::vector<char> Buffer(InfoLogLength);
        glGetShaderInfoLog(ShaderName, InfoLogLength, NULL, &Buffer[0]);
        fprintf(stdout, "%s\n", &Buffer[0]);
      }

      return Result == GL_TRUE;
    }

    GLuint createShader(GLenum Type, std::string const & Source)
    {
      bool Validated = true;
      GLuint Name = 0;

      if(!Source.empty())
      {
std::string globalsShader = SHADER_DIRECTORY + "globals"; //should probably offload the globals loading to a different place
std::string SourceContent = Utils::loadFile(globalsShader) + '\n' + Utils::loadFile(Source);
char const * SourcePointer = SourceContent.c_str();
Name = glCreateShader(Type);
glShaderSource(Name, 1, &SourcePointer, NULL);
glCompileShader(Name);
Validated = Utils::OpenGL::checkShader(Name, SourcePointer);
}

return Name;
}
bool checkError(const char* Title)
{
  int Error;
  if((Error = glGetError()) != GL_NO_ERROR)
  {
    std::string ErrorString;
    switch(Error)
    {
      case GL_INVALID_ENUM:
      ErrorString = "GL_INVALID_ENUM";
      break;
      case GL_INVALID_VALUE:
      ErrorString = "GL_INVALID_VALUE";
      break;
      case GL_INVALID_OPERATION:
      ErrorString = "GL_INVALID_OPERATION";
      break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:
      ErrorString = "GL_INVALID_FRAMEBUFFER_OPERATION";
      break;
      case GL_OUT_OF_MEMORY:
      ErrorString = "GL_OUT_OF_MEMORY";
      break;
      default:
      ErrorString = "UNKNOWN";
      break;
    }
    fprintf(stdout, "OpenGL Error(%s): %s\n", ErrorString.c_str(), Title);
  }
  return Error == GL_NO_ERROR;
}

// Returns the shader program
GLuint createShaderProgram(std::string& vertexShader, std::string& fragmentShader)
{
  printf("Compiling:\n%s\n%s\n", vertexShader.c_str(), fragmentShader.c_str());
  GLuint vertexShaderObject = Utils::OpenGL::createShader(GL_VERTEX_SHADER, vertexShader);
  GLuint fragmentShaderObject = Utils::OpenGL::createShader(GL_FRAGMENT_SHADER, fragmentShader);

  GLuint shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShaderObject);
  glAttachShader(shaderProgram, fragmentShaderObject);
  glDeleteShader(vertexShaderObject);
  glDeleteShader(fragmentShaderObject);

  glLinkProgram(shaderProgram);
  Utils::OpenGL::checkProgram(shaderProgram);

  return shaderProgram;
}

bool checkFramebuffer(GLuint FramebufferName)
{
  GLenum Status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  switch(Status)
  {
    case GL_FRAMEBUFFER_UNDEFINED:
    fprintf(stdout, "OpenGL Error(%s)\n", "GL_FRAMEBUFFER_UNDEFINED");
    break;
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
    fprintf(stdout, "OpenGL Error(%s)\n", "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT");
    break;
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
    fprintf(stdout, "OpenGL Error(%s)\n", "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT");
    break;
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
    fprintf(stdout, "OpenGL Error(%s)\n", "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER");
    break;
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
    fprintf(stdout, "OpenGL Error(%s)\n", "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER");
    break;
    case GL_FRAMEBUFFER_UNSUPPORTED:
    fprintf(stdout, "OpenGL Error(%s)\n", "GL_FRAMEBUFFER_UNSUPPORTED");
    break;
    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
    fprintf(stdout, "OpenGL Error(%s)\n", "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE");
    break;
    case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
    fprintf(stdout, "OpenGL Error(%s)\n", "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS");
    break;
  }

  return Status != GL_FRAMEBUFFER_COMPLETE;
}
bool checkExtension(char const * String)
{
  GLint ExtensionCount = 0;
  glGetIntegerv(GL_NUM_EXTENSIONS, &ExtensionCount);
  for(GLint i = 0; i < ExtensionCount; ++i)
  {
    std::string extensionName = std::string((char const*)glGetStringi(GL_EXTENSIONS, i));
    printf((extensionName + "\n").c_str());
    if(extensionName == std::string(String))
      return true;
  }
  return false;
}
static void APIENTRY debugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, GLvoid* userParam)
{
  char debSource[32], debType[32], debSev[32];
  if(source == GL_DEBUG_SOURCE_API_ARB)
    strcpy(debSource, "OpenGL");
  else if(source == GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB)
    strcpy(debSource, "Windows");
  else if(source == GL_DEBUG_SOURCE_SHADER_COMPILER_ARB)
    strcpy(debSource, "Shader Compiler");
  else if(source == GL_DEBUG_SOURCE_THIRD_PARTY_ARB)
    strcpy(debSource, "Third Party");
  else if(source == GL_DEBUG_SOURCE_APPLICATION_ARB)
    strcpy(debSource, "Application");
  else if(source == GL_DEBUG_SOURCE_OTHER_ARB)
    strcpy(debSource, "Other");

  if(type == GL_DEBUG_TYPE_ERROR_ARB)
    strcpy(debType, "error");
  else if(type == GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB)
    strcpy(debType, "deprecated behavior");
  else if(type == GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB)
    strcpy(debType, "undefined behavior");
  else if(type == GL_DEBUG_TYPE_PORTABILITY_ARB)
    strcpy(debType, "portability");
  else if(type == GL_DEBUG_TYPE_PERFORMANCE_ARB)
    strcpy(debType, "performance");
  else if(type == GL_DEBUG_TYPE_OTHER_ARB)
    strcpy(debType, "message");

  if(severity == GL_DEBUG_SEVERITY_HIGH_ARB)
    strcpy(debSev, "high");
  else if(severity == GL_DEBUG_SEVERITY_MEDIUM_ARB)
    strcpy(debSev, "medium");
  else if(severity == GL_DEBUG_SEVERITY_LOW_ARB)
    strcpy(debSev, "low");

  fprintf(stderr,"%s: %s(%s) %d: %s\n", debSource, debType, debSev, id, message);
}
}
}

#endif  // UTILS_H