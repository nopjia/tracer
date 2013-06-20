#ifndef COMMON_H
#define COMMON_H

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

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/half_float.hpp>
#include <glm/gtc/type_precision.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtx/random.hpp>
#include <glm/gtx/quaternion.hpp>

#include <iostream>

typedef unsigned char byte;
typedef unsigned char uchar;
typedef unsigned int uint;

// constants
#define WINDOW_W 640
#define WINDOW_H 480

// Vertex attribute indexes
const uint POSITION_ATTR            = 0;
const uint NORMAL_ATTR              = 1;
const uint UV_ATTR                  = 2;

#endif  // COMMON_H