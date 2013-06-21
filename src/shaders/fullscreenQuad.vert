layout(location = POSITION_ATTR) in vec2 position;

out gl_PerVertex
{
   vec4 gl_Position;
};

out vec2 vUV;

void main()
{
  vUV = (position+1.0)/2.0;

  gl_Position = vec4(position, 0.0, 1.0);
}