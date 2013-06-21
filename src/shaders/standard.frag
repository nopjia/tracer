#version 330
#extension GL_ARB_shading_language_420pack : enable

//------------------------
//      Material
//------------------------

struct Material
{
	vec4 diffuseColor;
	vec4 specularColor;
	int getColorFromTexture;
};

uniform Material material;


//------------------------
//      Lighting
//------------------------

#define LIGHTING_UBO_INDEX    0
#define MAX_NUMBER_OF_LIGHTS  8

struct DirLight
{
    vec3 color;
    vec3 direction;
};

struct PointLight
{
    vec3 color;
    vec4 position; //position.w contains attenuation factor
};

struct Lighting
{
    PointLight pointLights[MAX_NUMBER_OF_LIGHTS];
    DirLight dirLights[MAX_NUMBER_OF_LIGHTS];
    uint numPointLights; 
    uint numDirLights;
};

layout(binding = LIGHTING_UBO_INDEX) uniform LightingUBO
{
	Lighting lighting;
};

uniform vec3 cameraPos;

//---------------------------------
//      Vertex shader outputs
//---------------------------------

in block
{
	vec3 position;
	vec3 normal;
	vec2 texcoord;
	
} vertexData;


//---------------------------------
//      Texture
//---------------------------------

layout (binding = 0) uniform sampler2D colorTexture;

//---------------------------------
//      Output
//---------------------------------

layout (location = 0, index = 0) out vec4 fragColor;





#define KD 0.6
#define KA 0.1
#define KS 0.3
#define SPEC 20.0

vec3 computeLighting(vec3 viewDir, vec3 lightDir, float lightAttenuation, vec3 lightColor, vec3 diffuse, vec3 normal, float specular)
{
    vec3 reflectedLight = reflect(lightDir, normal);
    float diffuseTerm = max(dot(lightDir, normal), 0.0);

    //phong
    float specularTerm = pow( max(dot(reflectedLight,viewDir), 0.0), SPEC );

    vec3 cout = 
        KA * diffuse + 
        KD * diffuse * diffuseTerm * lightColor * lightAttenuation +
        KS * specular * specularTerm * lightColor * lightAttenuation;

    cout = min(cout, 1.0);
    return cout;
}


void main()
{
    vec3 diffuse;
    vec3 specular = material.specularColor.rgb;
    vec3 normal = normalize(vertexData.normal);
    
    if(material.getColorFromTexture == 0)
	{
		diffuse = material.diffuseColor.rgb;
	}
	else
	{
		diffuse = texture(colorTexture, vertexData.texcoord).rgb;
	}
    
    
    vec3 viewDir = normalize(vertexData.position - cameraPos);
    
    vec3 cout = vec3(0.0);
    
    for(uint i = 0U; i < lighting.numDirLights; i++)
    {
        DirLight dirLight = lighting.dirLights[i];
        vec3 lightDir = -dirLight.direction;
        vec3 lightColor = dirLight.color;
        float lightAttenuation = 1.0;
        cout += computeLighting(viewDir, lightDir, lightAttenuation, lightColor, diffuse, normal, specular);
    }
    
    for(uint i = 0U; i < lighting.numPointLights; i++)
    {
        PointLight pointLight = lighting.pointLights[i];
        vec3 lightDifference = pointLight.position.xyz - vertexData.position;
        vec3 lightDir = normalize(lightDifference);
        vec3 lightColor = pointLight.color;
        float lightAttenuation = 1.0;
        cout += computeLighting(viewDir, lightDir, lightAttenuation, lightColor, diffuse, normal, specular);
    }
    
    fragColor = vec4(cout, 1.0);

    //fragColor = vec4(normal, 1.0);
}
