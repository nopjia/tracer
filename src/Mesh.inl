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

//---------------------------------------------------------
// Declaration
//---------------------------------------------------------

  struct Face {
    glm::uvec3 m_v;
    glm::uvec3 m_n;
  };

  struct Triangle {
    glm::vec3 m_v[3];
    glm::vec3 m_n[3];
  };

  enum MeshType {MESH, SPHERE};

  struct Mesh {
    uint m_numVerts, m_numNorms, m_numFaces;
    glm::vec3* m_verts;
    glm::vec3* m_norms;
    Face* m_faces;
    glm::vec3 m_bmin, m_bmax;
    MeshType m_type;
  };

  // forward declarations
  HOST DEVICE extern inline Triangle getTriangle(const Mesh& mesh, const uint index);

//---------------------------------------------------------
// Function Implementation
//---------------------------------------------------------

  Triangle getTriangle(const Mesh& mesh, const uint idx) {
    Triangle tri;

    if (idx > mesh.m_numFaces-1)
      return tri;

    tri.m_v[0] = mesh.m_verts[mesh.m_faces[idx].m_v[0]];
    tri.m_v[1] = mesh.m_verts[mesh.m_faces[idx].m_v[1]];
    tri.m_v[2] = mesh.m_verts[mesh.m_faces[idx].m_v[2]];

    tri.m_n[0] = mesh.m_norms[mesh.m_faces[idx].m_n[0]];
    tri.m_n[1] = mesh.m_norms[mesh.m_faces[idx].m_n[1]];
    tri.m_n[2] = mesh.m_norms[mesh.m_faces[idx].m_n[2]];

    return tri;
  }

}

#endif  // MESH_INL