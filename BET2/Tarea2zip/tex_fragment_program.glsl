#version 330

in vec3 frag_position;
in vec2 frag_texcoord;
in vec3 frag_normal;

uniform vec3 Kd;
uniform vec3 Ks;
uniform vec3 Ka;
uniform float ns;

uniform vec3 light_position;
uniform vec3 view_position;

uniform sampler2D sampler_tex;

out vec4 outColor;

void main()
{
    vec3 Ld = vec3(1, 1, 1);
    vec3 Ls = vec3(1, 1, 1);
    vec3 La = vec3(1, 1, 1);

    // ambient
    vec3 ambient = Ka * La;

    // componente difuso

    float constantAttenuation = 0.001;
    float linearAttenuation = 0.01;
    float quadraticAttenuation = 0.001; 

    vec3 normal = normalize(frag_normal);
    vec3 to_light = light_position - frag_position;
    vec3 light_dir = normalize(to_light);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = Kd * Ld * diff;

    // componente especular
    vec3 view_dir = normalize(view_position - frag_position);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), ns);
    vec3 specular = Ks * Ls * spec;

    // atenuacion
    float dist_to_light = length(to_light);
    float attenuation = constantAttenuation 
        + linearAttenuation * dist_to_light
        + quadraticAttenuation * dist_to_light * dist_to_light;

    vec4 frag_og_color = texture(sampler_tex, frag_texcoord);

    vec3 result = (ambient * ((diffuse + specular) / attenuation)) * frag_og_color.rgb;
    outColor = vec4(result, 1.0);
}