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

  struct Ray {
    glm::vec3 m_pos;
    glm::vec3 m_dir;
  };

  struct Hit {
    HOST DEVICE Hit() : m_id(-1){}

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
    h.m_nor = glm::vec3( m*glm::vec4(hit.m_nor, 0.0f) );
    return h;
  }

  Hit intersectScene(const Ray& ray, const Object::Object* scene, const uint size) {
    float mindist = FLT_MAX;
    Hit minhit;

    for (int i=0; i<size; ++i) {
      Hit h ( intersect(ray, scene[i]) );
      if (h.m_id > 0) {
        float dist = glm::distance(ray.m_pos, h.m_pos);
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
    Ray r ( transform(ray, glm::inverse(Object::getModelMatrix(obj))) );

    // intersection test
    Hit hit ( intersect(r, *obj.m_mesh) );

    if (hit.m_id < 0) {
      return Hit();
    }
    else {
      // transfrom hit, object to world space
      return transform(hit, Object::getModelMatrix(obj));
    }
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
    hit.m_pos = ray.m_pos + ray.m_dir*tmin;
    return hit;
    
    // TODO: loop triangles and return intersection
  }


}

#endif  // RAY_INL