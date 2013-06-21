#version 330
#extension GL_ARB_shading_language_420pack : enable

uniform mat4 modelMatrix;
uniform mat4 viewProjectionMatrix;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

out block
{
	vec3 position;
	vec3 normal;
	vec2 texcoord;
} vertexData;

/*-------------------------
		Main
---------------------------*/

void main()
{
	vec4 worldPosition = modelMatrix * vec4(position, 1.0);
	gl_Position = viewProjectionMatrix * worldPosition;
    
    vertexData.position = vec3(worldPosition);
	vertexData.normal = normalize(mat3(modelMatrix) * normal);
	vertexData.texcoord = texcoord;
}
