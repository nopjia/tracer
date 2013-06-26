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
  HOST DEVICE extern inline float intersect(const Ray& ray, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2);
  
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
    h.m_nor = glm::vec3( m*glm::vec4(hit.m_nor, 0.0f) );
    return h;
  }

  Hit intersectScene(const Ray& ray, const Object::Object* scene, const uint size) {
    float mindist = FLT_MAX;
    Hit minhit;

    for (int i=0; i<size; ++i) {
      Hit h ( intersect(ray, scene[i]) );
      if (h.m_id > 0) {
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

    if (hit.m_id < 0) {
      return Hit();
    }
    else {
      // transfrom hit, object to world space
      return transform(hit, obj.m_matrix);
    }

    return hit;
  }

  Hit intersect(const Ray& ray, const Mesh::Mesh& mesh) {
    float tmin = FLT_MIN;
    float tmax = FLT_MAX;
    for (int i=0; i<3; ++i) {
      if (glm::abs(ray.m_dir[i]) < EPS) {
        // ray parallel to slab
        if (ray.m_pos[i] < mesh.m_bmin[i] || ray.m_pos[i] > mesh.m_bmax[i]) {
          return Hit();
        }
      }
      else {
        // compute intersect t with near and far plane
        float t1 = (mesh.m_bmin[i] - ray.m_pos[i]) / ray.m_dir[i];
        float t2 = (mesh.m_bmax[i] - ray.m_pos[i]) / ray.m_dir[i];
        // make t1 intersection with near plane, swap
        if (t1 > t2) { float temp = t1; t1 = t2; t2 = temp; }
        // compute intersect slab intervals
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
        // exit with no collision, empty slab intersection
        if (tmin > tmax || tmax < 0.0f) return Hit();
      }
    }
    // ray intersects all 3 slabs, return
    Hit hit;
    hit.m_id = 1; // YES HIT at Mesh level    
    
    // loop triangles and return intersection
    hit.m_t = FLT_MAX;
    for (int i=0; i<mesh.m_numFaces; ++i) {      
      float thit = intersect(ray, 
        mesh.m_verts[mesh.m_faces[i].m_v[0]], 
        mesh.m_verts[mesh.m_faces[i].m_v[1]], 
        mesh.m_verts[mesh.m_faces[i].m_v[2]]);
      if (thit > 0.0f && thit < hit.m_t)
        hit.m_t = thit;
    }

    hit.m_pos = ray.m_pos + ray.m_dir*hit.m_t;
    return hit;
  }

  float intersect(const Ray& ray, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) {
    glm::vec3 e1 = p1-p0;
    glm::vec3 e2 = p2-p0;

    glm::vec3 pvec = glm::cross(ray.m_dir, e2);
    float det = glm::dot(e1, pvec);
    if (glm::abs(det) < EPS)
      return -1.0f;

    det = 1.0f / det;
    glm::vec3 tvec = ray.m_pos- p0;
    float u = glm::dot(tvec, pvec) * det;
    if (u < 0.0f || u > 1.0f)
      return -1.0f;

    glm::vec3 qvec = glm::cross(tvec, e1);
    float v = glm::dot(ray.m_dir, qvec) * det;
    if (v < 0.0f || (u + v) > 1.0f)
      return -1.0f;

    return glm::dot(e2, qvec) * det;
  }
}

#endif  // RAY_INL