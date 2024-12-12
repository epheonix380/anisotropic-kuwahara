#version 330 core

#define PI 3.14159265358979323846

uniform sampler2D display;
uniform vec2 res;

in vec2 fragCoord;
out vec4 fragColor;

void main() {
    vec2 uv = fragCoord.xy / res.xy;
    vec3 g = texture(display, uv).xyz;

    float lambda1 = 0.5 * (g.y + g.x +
        sqrt(g.y * g.y - 2.0 * g.x * g.y + g.x * g.x + 4.0 * g.z * g.z));
    float lambda2 = 0.5 * (g.y + g.x -
        sqrt(g.y * g.y - 2.0 * g.x * g.y + g.x * g.x + 4.0 * g.z * g.z));
    
    vec2 v = vec2(lambda1 - g.x, -g.z);
    vec2 t;
    if (length(v) > 0.0) {
        t = normalize(v);
    } else {
        t = vec2(0.0, 1.0);
    }

    float phi = atan(t.y, t.x);
    float A = (lambda1 + lambda2 > 0.0) ?
              (lambda1 - lambda2) / (lambda1 + lambda2) : 0.0;

    fragColor = vec4(t, phi, A);
}
