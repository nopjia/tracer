#ifndef MESH_INL
#define MESH_INL

#include "common.h"

#include <glm/glm.hpp>

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

namespace Mesh {

  struct Triangle {
    glm::ivec3 m_v;
  };

  struct Mesh {
    uint m_numVerts, m_numFaces;
    glm::vec3* m_verts;
    Triangle* m_faces;
    glm::vec3 m_bmin, m_bmax;
  };

}

#endif  // MESH_INL