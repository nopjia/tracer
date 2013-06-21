#include "Camera.h"

Camera::Camera() :
m_rotationAmount(0.005f),
m_panAmount(0.04f),
m_zoomAmount(0.05f),
m_currXZRads(0.0f),
m_currYRads(0.0f),
m_nearPlane(0.1f),
m_farPlane(1000.0f),
m_fieldOfView(FOV),
m_aspectRatio(1.0f),
m_upDir(0.0f, 1.0f, 0.0f),
m_lookDir(0.0f, 0.0f, 1.0f),
m_rightDir(glm::cross(m_lookDir, m_upDir))
{
}

Camera::~Camera() 
{
}

void Camera::setAspectRatio(uint screenWidth, uint screenHeight)
{
  m_aspectRatio = (float)screenWidth / screenHeight;
}

void Camera::setFarNearPlanes(float nearPlane, float farPlane)
{
  m_nearPlane = nearPlane;
  m_farPlane = farPlane;
}

void Camera::rotateDegrees(float x, float y)
{
  rotate(glm::radians(x), glm::radians(y));
}

glm::vec3 Camera::getPosition()
{
  return m_position;
}
glm::vec3 Camera::getLookAt()
{
  return m_lookAt;
}
glm::vec3 Camera::getUp()
{
  return m_upDir;
}
float Camera::getAspect() {
  return m_aspectRatio;
}
float Camera::getFOV() {
  return m_fieldOfView;
}


//---------------------------------------------------------
// THIRD PERSON CAMERA
//---------------------------------------------------------

ThirdPersonCamera::ThirdPersonCamera() : Camera(),
m_radius(0.0f)
{
}

ThirdPersonCamera::~ThirdPersonCamera()
{
}

void ThirdPersonCamera::rotate(float x, float y)
{
  m_currXZRads += x;
  m_currYRads += y;
}

void ThirdPersonCamera::pan(float x, float y)
{
  m_lookAt += x * glm::normalize(glm::cross(m_lookDir, m_upDir));
  m_lookAt += y * m_upDir;
}

void ThirdPersonCamera::zoom(float distance)
{
  m_radius -= distance;
}

void ThirdPersonCamera::update()
{
  float cosa = cosf(m_currXZRads);
  float sina = sinf(m_currXZRads);

  glm::vec3 currPos(sina, 0.0f, cosa);
  glm::vec3 UpRotAxis(currPos.z, currPos.y, -currPos.x);

  glm::mat4 xRotation = glm::rotate(glm::mat4(1.0f), glm::degrees(m_currYRads), UpRotAxis);
  currPos = glm::vec3(xRotation * glm::vec4(currPos, 0.0));
  glm::vec3 tempVec = currPos * float(m_radius);

  m_position = tempVec + m_lookAt;
  m_upDir = glm::normalize(glm::cross(currPos, UpRotAxis));
  m_lookDir = glm::normalize(m_lookAt - m_position);
  m_rightDir = glm::cross(m_lookDir, m_upDir);
}

//---------------------------------------------------------
// FIRST PERSON CAMERA
//---------------------------------------------------------

FirstPersonCamera::FirstPersonCamera() : Camera() 
{
}

FirstPersonCamera::~FirstPersonCamera()
{
}

void FirstPersonCamera::rotate(float x, float y)
{
  m_currXZRads += x;
  m_currYRads -= y;
}

void FirstPersonCamera::pan(float x, float y)
{
  m_position += x * glm::normalize(glm::cross(m_lookDir, m_upDir));
  m_position += y * m_upDir;
}

void FirstPersonCamera::zoom(float distance)
{
  m_position += distance * m_lookDir;
}

void FirstPersonCamera::update()
{
  float cosa = cosf(m_currXZRads);
  float sina = sinf(m_currXZRads);

  glm::vec3 currPos(sina, 0.0f, cosa);
  glm::vec3 UpRotAxis(currPos.z, currPos.y, -currPos.x);

  glm::mat4 xRotation = glm::rotate(glm::mat4(1.0f), glm::degrees(m_currYRads), UpRotAxis);
  currPos = glm::vec3(xRotation * glm::vec4(currPos, 0.0));

  m_lookDir = glm::normalize(currPos);
  m_upDir = glm::normalize(glm::cross(currPos, UpRotAxis));
  m_rightDir = glm::cross(m_lookDir, m_upDir);
}