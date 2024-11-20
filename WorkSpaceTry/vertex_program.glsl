#version 330 core

in vec3 position;
in vec2 uv;
in vec3 normal;

out vec3 fragPosition;
out vec2 fragUV;
out vec3 fragNormal;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    //fragPosition = vec3(view * projection * transform * vec4(position, 1.0));
    fragPosition = vec3(projection * view * transform * vec4(position, 1.0));
    fragUV = uv;
    fragNormal = mat3(transpose(inverse(transform))) * normal;  
    
    gl_Position = vec4(fragPosition, 1.0);
}