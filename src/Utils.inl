#ifndef UTILS_INL
#define UTILS_INL

#include "common.h"

#include <glm/glm.hpp>

namespace Utils {

  HOST DEVICE extern inline float rand(glm::vec3 seed) {
      return glm::fract(glm::sin(glm::dot(seed,glm::vec3(93.5734, 12.9898, 78.2331))) * 43758.5453);
  }

  // http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
  HOST DEVICE extern inline glm::vec3 randVector(float rand1, float rand2) {
    float phi = rand1*2.0f*M_PI;
    float theta = glm::acos( rand2*2.0f-1.0f );
    return glm::vec3 (
      glm::sin(theta)*glm::cos(phi),
      glm::sin(theta)*glm::sin(phi),
      glm::cos(theta)
    );
  }

  HOST DEVICE extern inline glm::vec2 randPointDisk(float rand1, float rand2, float rand3) {
    float t = 2.0f*M_PI*rand1;
    float u = rand2 + rand3;
    float r = u > 1.0f ? 2.0f-u : u;
    return glm::vec2(r*glm::cos(t), r*glm::sin(t));
  }

  HOST DEVICE extern inline glm::vec3 randVectorHem(float rand1, float rand2, glm::vec3 nor) {
    glm::vec3 v = randVector(rand1, rand2);
    if (glm::dot(v,nor) < 0)
      v = -v;
    return v;
  }

}

#endif  // UTILS_INL