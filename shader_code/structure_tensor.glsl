#include "config.glsl"

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 src_size = iResolution.xy;
    vec2 uv = fragCoord.xy / src_size;
    vec2 d = 1.0 / src_size;

    vec3 c = texture(iChannel0, uv).xyz;

    vec3 u = (
        -1.0 * texture(iChannel0, uv + vec2(-d.x, -d.y)).xyz +
        -2.0 * texture(iChannel0, uv + vec2(-d.x, 0.0)).xyz +
        -1.0 * texture(iChannel0, uv + vec2(-d.x, d.y)).xyz +
        +1.0 * texture(iChannel0, uv + vec2(d.x, -d.y)).xyz +
        +2.0 * texture(iChannel0, uv + vec2(d.x, 0.0)).xyz +
        +1.0 * texture(iChannel0, uv + vec2(d.x, d.y)).xyz
    ) / 4.0;

    vec3 v = (
        -1.0 * texture(iChannel0, uv + vec2(-d.x, -d.y)).xyz +
        -2.0 * texture(iChannel0, uv + vec2(0.0, -d.y)).xyz +
        -1.0 * texture(iChannel0, uv + vec2(d.x, -d.y)).xyz +
        +1.0 * texture(iChannel0, uv + vec2(-d.x, d.y)).xyz +
        +2.0 * texture(iChannel0, uv + vec2(0.0, d.y)).xyz +
        +1.0 * texture(iChannel0, uv + vec2(d.x, d.y)).xyz
    ) / 4.0;

    fragColor = vec4(dot(u, u), dot(v, v), dot(u, v), 1.0);
}
