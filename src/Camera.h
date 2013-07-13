#ifndef CAMERA_H
#define CAMERA_H

#include "common.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera
{

public:
  Camera();
  virtual ~Camera();

  void rotateDegrees(float x, float y);
  virtual void rotate(float x, float y) = 0;
  virtual void pan(float x, float y) = 0;
  virtual void zoom(float delta) = 0;
  virtual void update() = 0;

  void setAspect(uint screenWidth, uint screenHeight);
  void setFOV(float fov);
  void setFarNearPlanes(float nearPlane, float farPlane);

  glm::vec3 getPosition();
  glm::vec3 getLookAt();
  glm::vec3 getUp();
  float getAspect();
  float getFOV();  

  float m_lensRadius;
  float m_focalDist;

protected:
  float m_currXZRads;
  float m_currYRads;

  glm::vec3 m_position;
  glm::vec3 m_upDir;
  glm::vec3 m_lookDir;
  glm::vec3 m_rightDir;
  glm::vec3 m_lookAt;

  float m_nearPlane;
  float m_farPlane;
  float m_fov; //degrees
  float m_aspect;

  float m_rotationAmount;
  float m_panAmount;
  float m_zoomAmount;
};

struct ThirdPersonCamera : public Camera
{
public:
  ThirdPersonCamera();
  virtual ~ThirdPersonCamera();

  virtual void rotate(float x, float y);
  virtual void pan(float x, float y);
  virtual void zoom(float delta);
  virtual void update();

protected:
  float m_radius;
};

struct FirstPersonCamera : public Camera
{
public:
  FirstPersonCamera();
  virtual ~FirstPersonCamera();

  virtual void rotate(float x, float y);
  virtual void pan(float x, float y);
  virtual void zoom(float delta);
  virtual void update();
};

#endif  // CAMERA_H