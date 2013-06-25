#ifndef OBJECT_INL
#define OBJECT_INL

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include "Mesh.inl"

namespace Object {

//---------------------------------------------------------
// Struct
//---------------------------------------------------------

  struct Object {
    glm::vec3 m_scale;
    glm::vec3 m_translation;
    glm::quat m_rotationQuat;
    Mesh::Mesh* m_mesh;
  };

  // forward declarations
  HOST DEVICE extern inline Object* newObject(Mesh::Mesh* mesh);
  HOST DEVICE extern inline glm::mat4 getModelMatrix(const Object& obj);
  HOST DEVICE extern inline void translate(Object& obj, const glm::vec3& amount);
  HOST DEVICE extern inline void rotate(Object& obj, const glm::quat& quaternion);
  HOST DEVICE extern inline void scale(Object& obj, const float amount);
  HOST DEVICE extern inline void scale(Object& obj, const glm::vec3& amount);

//---------------------------------------------------------
// Function Implementation
//---------------------------------------------------------

  Object* newObject(Mesh::Mesh* mesh) {
    Object* obj = (Object*)malloc(sizeof(Object));
    obj->m_scale = glm::vec3(1.0f);
    obj->m_translation = glm::vec3(0.0f);
    obj->m_rotationQuat = glm::quat();
    obj->m_mesh = mesh;
    return obj;
  }

  glm::mat4 getModelMatrix(const Object& obj) {
    glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), obj.m_scale);
    glm::mat4 rotationMatrix = glm::toMat4(obj.m_rotationQuat);
    glm::mat4 transformationMatrix = rotationMatrix * scaleMatrix;
    transformationMatrix[3] = glm::vec4(obj.m_translation, 1.0f);
    return transformationMatrix;
  }

  void translate(Object& obj, const glm::vec3& amount) {
    obj.m_translation += amount;
  }

  void rotate(Object& obj, const glm::quat& quaternion)
  {
    obj.m_rotationQuat = quaternion * obj.m_rotationQuat;
  }

  void scale(Object& obj, const float amount) {
    obj.m_scale *= amount;
  }

  void scale(Object& obj, const glm::vec3& amount) {
    obj.m_scale *= amount;
  }
}

#endif  // OBJECT_INL