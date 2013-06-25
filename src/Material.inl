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
    Material() : m_color(glm::vec3(0.5)), m_emissivity(0.0){}

    glm::vec3 m_color;
    float m_emissivity;
  };

}

#endif  // MATERIAL_INL