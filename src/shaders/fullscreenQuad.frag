layout(location = 0) out vec4 fragColor;
layout(binding = RENDER_TEXTURE) uniform sampler2D tRender;
in vec2 vUV;

void main()
{  
  fragColor = vec4(vUV.x, vUV.y, 0.0, 1.0);
  fragColor = texture2D(tRender, vUV);
}