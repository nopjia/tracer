#ifndef MATERIAL_INL
#define MATERIAL_INL

#include "common.h"

#include <glm/glm.hpp>

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

namespace Material {

  struct Material {
    glm::vec3 color;
    float emissivity;
  };

}

#endif  // MATERIAL_INL