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

  enum Type {DIFF, MIRR};

  struct Material {
    HOST DEVICE Material() : 
      m_color(glm::vec3(0.5f)), 
      m_emit(0.0f),
      m_brdf(1.0f),
      m_type(DIFF){}

    glm::vec3 m_color;
    float m_emit;
    float m_brdf;
    Type m_type;
  };

}

#endif  // MATERIAL_INL