#ifndef MATERIAL_INL
#define MATERIAL_INL

#include "common.h"
#include "Utils.inl"

#include <glm/glm.hpp>

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

namespace Material {

  enum Type {DIFF, TRANS, MIRR};

  struct Material {
    HOST DEVICE Material() : 
      m_color(glm::vec3(0.5f)), 
      m_emit(0.0f),
      m_brdf(1.0f),
      m_n(1.0f),
      m_type(DIFF){}

    glm::vec3 m_color;
    float m_emit;
    float m_brdf;
    float m_n;
    Type m_type;
  };

  // forward declarations
  HOST DEVICE extern inline float reflectance(const glm::vec3& nor, const glm::vec3& inc, const float n1, const float n2);
  HOST DEVICE extern inline glm::vec3 bounce(const Material& mat, const glm::vec3& ro, const glm::vec3& nor, const glm::vec3& randvec);

//---------------------------------------------------------
// Function Implementation
//---------------------------------------------------------

  float reflectance(const glm::vec3& inc, const glm::vec3& nor, const float n1, const float n2) {
    float r0 = (n1-n2)/(n1+n2);
    r0 *= r0;
    float cosI = -glm::dot(nor, inc);
    if (n1 > n2) {
      float n = n1/n2;
      float sinT2 = n*n*(1.0-cosI*cosI);
      if (sinT2 > 1.0) return 1.0; // total internal reflection
      cosI = glm::sqrt(1.0-sinT2);
    }
    float x = 1.0 - cosI;
    return r0 + (1.0-r0) * x*x*x*x*x;
  }

  glm::vec3 bounce(const Material& mat, const glm::vec3& ro, const glm::vec3& nor, const glm::vec3& randvec) {
    if (mat.m_type == DIFF) {
      return Utils::randVectorHem(randvec.x,randvec.y,nor);
    }
    else if (mat.m_type == MIRR) {
      return glm::reflect(ro, nor);
    }
    else if (mat.m_type == TRANS) {
      float n1, n2;
      glm::vec3 nnor;      

      // from outside
      if (glm::dot(ro,nnor) < 0.0f) {        
        n1 = 1.0f;
        n2 = mat.m_n;
        nnor = nor;
      }
      // from inside transparent medium
      else {
        n1 = mat.m_n;
        n2 = 1.0f;
        nnor = -nor;
      }
      
      float refl = reflectance(ro, nnor, n1, n2);
      if (randvec.x < refl)
        return glm::reflect(ro, nnor);
      else
        return glm::refract(ro, nnor, n1/n2);
    }
  }

}

#endif  // MATERIAL_INL