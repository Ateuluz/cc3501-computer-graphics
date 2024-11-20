#version 330 core

out vec4 fragColor;

in vec3 fragNormal;
in vec3 fragPosition;
in vec2 fragUV;

uniform vec3 lightPosition1; 
uniform vec3 lightPosition2; 
uniform vec3 lightPosition3; 
uniform vec3 viewPosition;
uniform vec3 Ka;
uniform vec3 Kd;
uniform vec3 Ks;
uniform float shininess;
uniform sampler2D samplerTex;
uniform vec3 Ld1;
uniform vec3 Ls1;
uniform vec3 La1;
uniform vec3 La2;
uniform vec3 Ld2;
uniform vec3 Ls2;
uniform vec3 La3;
uniform vec3 Ld3;
uniform vec3 Ls3;

//vec3 La1 = vec3(0,0.5,1);
//vec3 Ld1 = vec3(0,0.5,1);
//vec3 Ls1 = vec3(0,0.5,1);

//vec3 La2 = vec3(1,0.5,0);
//vec3 Ld2 = vec3(1,0.5,0);
//vec3 Ls2 = vec3(1,0.5,0);

//vec3 La3 = vec3(1,0.5,0);
//vec3 Ld3 = vec3(1,0.5,0);
//vec3 Ls3 = vec3(1,0.5,0);

float constantAttenuation = 0.01;
float linearAttenuation = 0.01;
float quadraticAttenuation = 0.01;

void main()
{

    vec3 normalizedNormal = normalize(fragNormal);
    vec3 viewDir = normalize(viewPosition - fragPosition);

    // Light 1
    
    // attenuation
    // fragment normal has been interpolated, so it does not necessarily have norm equal to 1
    vec3 toLight1 = lightPosition1 - fragPosition;
    vec3 lightDir1 = normalize(toLight1);
    float distToLight1 = length(toLight1);
    float attenuation1 = constantAttenuation
        + linearAttenuation * distToLight1
        + quadraticAttenuation * distToLight1 * distToLight1;

    // ambient
    vec3 ambient1 = Ka * La1;

    // diffuse
    float diff1 = max(dot(normalizedNormal, lightDir1), 0.0);
    vec3 diffuse1 = Kd * Ld1 * diff1;
    
    // specular
    vec3 reflectDir1 = reflect(-lightDir1, normalizedNormal);  
    float spec1 = pow(max(dot(viewDir, reflectDir1), 0.0), shininess);
    vec3 specular1 = Ks * Ls1 * spec1;

    
    //===============================================


    // ambient
    vec3 ambient2 = Ka * La2;
    
    // diffuse
    vec3 toLight2 = lightPosition2 - fragPosition;
    vec3 lightDir2 = normalize(toLight2);
    float diff2 = max(dot(normalizedNormal, lightDir2), 0.0);
    vec3 diffuse2 = Kd * Ld2 * diff2;
    
    // specular
    vec3 reflectDir2 = reflect(-lightDir2, normalizedNormal);  
    float spec2 = pow(max(dot(viewDir, reflectDir2), 0.0), shininess);
    vec3 specular2 = Ks * Ls2 * spec2;

    // attenuation
    float distToLight2 = length(toLight2);
    float attenuation2 = constantAttenuation
        + linearAttenuation * distToLight2
        + quadraticAttenuation * distToLight2 * distToLight2;


    //===============================================


    vec3 ambient3 = Ka * La3;
    
    // diffuse
    vec3 toLight3 = lightPosition3 - fragPosition;
    vec3 lightDir3 = normalize(toLight3);
    float diff3 = max(dot(normalizedNormal, lightDir3), 0.0);
    vec3 diffuse3 = Kd * Ld3 * diff3;
    
    // specular
    vec3 reflectDir3 = reflect(-lightDir3, normalizedNormal);  
    float spec3 = pow(max(dot(viewDir, reflectDir3), 0.0), shininess);
    vec3 specular3 = Ks * Ls3 * spec3;

    // attenuation
    float distToLight3 = length(toLight3);
    float attenuation3 = constantAttenuation
        + linearAttenuation * distToLight3
        + quadraticAttenuation * distToLight3 * distToLight3;


    //===============================================


    vec3 ambient0 = ambient1 + ambient2 + ambient3;
    
    vec3 diffuse = (diffuse1 + specular1) / attenuation1
                 + (diffuse2 + specular2) / attenuation2
                 + (diffuse3 + specular3) / attenuation3;

    vec4 frag_og_color = texture(samplerTex, fragUV);

    vec3 result = (ambient0 + diffuse) * frag_og_color.rgb;

    fragColor = vec4(result, 1.0);
}