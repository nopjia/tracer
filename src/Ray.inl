#ifndef RAY_INL
#define RAY_INL

#include "common.h"
#include "Object.inl"
#include "Mesh.inl"

#include <glm/glm.hpp>

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

namespace Ray {

//---------------------------------------------------------
// Declaration
//---------------------------------------------------------

  struct Ray {
    glm::vec3 m_pos;
    glm::vec3 m_dir;
  };

  struct Hit {
    HOST DEVICE Hit() : m_t(-1.0f), m_id(-1){}

    float m_t;
    int m_id;  // object ID
    glm::vec3 m_pos;
    glm::vec3 m_nor;
  };

  // forward declarations
  HOST DEVICE extern inline Ray transform(const Ray& ray, const glm::mat4& m);
  HOST DEVICE extern inline Hit transform(const Hit& hit, const glm::mat4& m);
  HOST DEVICE extern inline Hit intersectScene(const Ray& ray, const Object::Object* scene, const uint size);
  HOST DEVICE extern inline Hit intersect(const Ray& ray, const Object::Object& obj);
  HOST DEVICE extern inline Hit intersect(const Ray& ray, const Mesh::Mesh& mesh);
  HOST DEVICE extern inline Hit intersect(const Ray& ray, const Mesh::Triangle& tri);
  
//---------------------------------------------------------
// Function Implementation
//---------------------------------------------------------
  
  Ray transform(const Ray& ray, const glm::mat4& m) {
    glm::vec3 pos   ( m*glm::vec4(ray.m_pos, 1.0f) );
    glm::vec3 point ( m*glm::vec4(ray.m_pos+ray.m_dir, 1.0f) );
    glm::vec3 dir   ( glm::normalize(point-pos) );

    Ray r = {pos, dir};
    return r;
  }

  Hit transform(const Hit& hit, const glm::mat4& m) {
    Hit h(hit);
    h.m_pos = glm::vec3( m*glm::vec4(hit.m_pos, 1.0f) );
    h.m_nor = glm::normalize(glm::vec3( m*glm::vec4(hit.m_nor, 0.0f) ));
    return h;
  }

  Hit intersectScene(const Ray& ray, const Object::Object* scene, const uint size) {
    float mindist = FLT_MAX;
    Hit minhit;

    for (int i=0; i<size; ++i) {
      Hit h ( intersect(ray, scene[i]) );
      if (h.m_t > 0.0f) {
        glm::vec3 subtract = ray.m_pos - h.m_pos;
        float dist = glm::dot(subtract,subtract);
        if (dist<mindist) {
          mindist = dist;
          minhit = h;
          minhit.m_id = i;
        }
      }
    }

    return minhit;
  }

  Hit intersect(const Ray& ray, const Object::Object& obj) {
    // transform ray, world to object space
    Ray r ( transform(ray, obj.m_matrixi) );

    // intersection test
    Hit hit ( intersect(r, *obj.m_mesh) );

    if (hit.m_t < 0.0f) {
      return Hit();
    }
    else {
      // transfrom hit, object to world space
      return transform(hit, obj.m_matrix);
    }

    return hit;
  }

  Hit intersect(const Ray& ray, const Mesh::Mesh& mesh) {
    if (mesh.m_type == Mesh::SPHERE) {
      glm::vec3 l = -ray.m_pos;
      float s = glm::dot(l,ray.m_dir);
      float l2 = glm::dot(l,l);
      if (s < 0.0f && l2 > 0.25f) return Hit();
      float m2 = l2 - s*s;
      if (m2 > 0.25f) return Hit();
      float q = glm::sqrt(0.25f-m2);
      Hit hit;
      if (l2 > 0.25f) hit.m_t = s - q;
      else hit.m_t = s + q;

      hit.m_pos = ray.m_pos + ray.m_dir*hit.m_t;
      hit.m_nor = glm::normalize(hit.m_pos);
      return hit;
    }

    else if (mesh.m_type == Mesh::MESH) {
      glm::vec3 tMin = (mesh.m_bmin-ray.m_pos) / ray.m_dir;
      glm::vec3 tMax = (mesh.m_bmax-ray.m_pos) / ray.m_dir;
      glm::vec3 t1 = glm::min(tMin, tMax);
      glm::vec3 t2 = glm::max(tMin, tMax);
      float tNear = glm::max(glm::max(t1.x, t1.y), t1.z);
      float tFar = glm::min(glm::min(t2.x, t2.y), t2.z);    
      if (tNear > tFar || tFar < 0.0f) return Hit();
    
      // loop triangles and return intersection
      Hit hit;
      hit.m_t = FLT_MAX;
      for (int i=0; i<mesh.m_numFaces; ++i) {      
        Hit thit = intersect(ray, Mesh::getTriangle(mesh,i));
        if (thit.m_t > 0.0f && thit.m_t < hit.m_t) {
          hit = thit;
        }
      }

      hit.m_pos = ray.m_pos + ray.m_dir*hit.m_t;

      #ifndef TRI_NORM_INTERP
      hit.m_nor = mesh.m_norms[mesh.m_faces[hit.m_id].m_n[0]];  // hack no interp
      #endif

      return hit;
    }
  }

  Hit intersect(const Ray& ray, const Mesh::Triangle& tri) {
    glm::vec3 e1 = tri.m_v[1]-tri.m_v[0];
    glm::vec3 e2 = tri.m_v[2]-tri.m_v[0];

    glm::vec3 pvec = glm::cross(ray.m_dir, e2);
    float det = glm::dot(e1, pvec);
    if (glm::abs(det) < EPS)
      return Hit();

    det = 1.0f / det;
    glm::vec3 tvec = ray.m_pos - tri.m_v[0];
    glm::vec3 bary;
    bary.x = glm::dot(tvec, pvec) * det;
    if (bary.x < 0.0f || bary.x > 1.0f)
      return Hit();

    glm::vec3 qvec = glm::cross(tvec, e1);
    bary.y = glm::dot(ray.m_dir, qvec) * det;
    if (bary.y < 0.0f || (bary.x + bary.y) > 1.0f)
      return Hit();

    Hit hit;
    hit.m_t = glm::dot(e2, qvec) * det;

#ifdef TRI_NORM_INTERP
    bary.z = 1.0f - bary.x - bary.y;
    hit.m_nor = tri.m_n[0]*bary.z 
              + tri.m_n[1]*bary.x 
              + tri.m_n[2]*bary.y;
#endif

    return hit;
  }
}

#endif  // RAY_INL