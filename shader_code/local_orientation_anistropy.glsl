#include "config.glsl"

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec3 g = texture(iChannel0, uv).xyz;

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
