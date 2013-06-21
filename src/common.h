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
const uint WINDOW_W = 640;
const uint WINDOW_H = 480;

// Vertex attribute indexes
const uint POSITION_ATTR                    = 0;
const uint NORMAL_ATTR                      = 1;
const uint UV_ATTR                          = 2;

// Uniform buffer objects binding points
const uint PER_FRAME_UBO_BINDING            = 0;
const uint LIGHT_UBO_BINDING                = 1;
const uint MESH_MATERIAL_ARRAY_BINDING      = 2;
const uint POSITION_ARRAY_BINDING           = 3;

// Sampler binding points
const uint NON_USED_TEXTURE                 = 0;
const uint RENDER_TEXTURE                   = 1;

// Max values
const uint MAX_TEXTURE_ARRAYS               = 10;
const uint NUM_OBJECTS_MAX                  = 500;
const uint NUM_MESHES_MAX                   = 500;
const uint MAX_POINT_LIGHTS                 = 8;

#endif  // COMMON_H