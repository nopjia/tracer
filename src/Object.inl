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
#include "Material.inl"

namespace Object {

//---------------------------------------------------------
// Struct
//---------------------------------------------------------

  struct Object {
    glm::mat4 m_matrix;
    glm::mat4 m_matrixi;
    Mesh::Mesh* m_mesh;
    Material::Material m_material;
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
    obj->m_matrix = glm::mat4();
    obj->m_matrixi = glm::mat4();
    obj->m_mesh = mesh;
    obj->m_material = Material::Material();

    return obj;
  }

  //glm::mat4 getModelMatrix(const Object& obj) {
  //  glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), obj.m_scale);
  //  glm::mat4 rotationMatrix = glm::toMat4(obj.m_rotationQuat);
  //  glm::mat4 transformationMatrix = rotationMatrix * scaleMatrix;
  //  transformationMatrix[3] = glm::vec4(obj.m_translation, 1.0f);
  //  return transformationMatrix;
  //}

#define UPDATE_MAT_I() (obj.m_matrixi = glm::inverse(obj.m_matrix))

  void translate(Object& obj, const glm::vec3& amount) {
    obj.m_matrix[3] += glm::vec4(amount, 1.0f);
    UPDATE_MAT_I();
  }

  void rotate(Object& obj, const glm::quat& quaternion)
  {
    obj.m_matrix = glm::toMat4(quaternion) * obj.m_matrix;
    UPDATE_MAT_I();
  }

  void scale(Object& obj, const float amount) {
    obj.m_matrix = glm::scale(obj.m_matrix, glm::vec3(amount));
    UPDATE_MAT_I();
  }

  void scale(Object& obj, const glm::vec3& amount) {
    obj.m_matrix = glm::scale(obj.m_matrix, amount);
    UPDATE_MAT_I();
  }
}

#endif  // OBJECT_INL