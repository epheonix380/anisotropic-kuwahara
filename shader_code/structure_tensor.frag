#version 330 core

#define PI 3.14159265358979323846
#define RAD 5

uniform sampler2D display;
uniform vec2 res;

in vec2 fragCoord;
out vec4 fragColor;

struct Window {
    int x1, y1, x2, y2;
};

void main() {
    vec2 src_size = res.xy;
    vec2 uv = fragCoord.xy / src_size;
    vec2 d = 1.0 / src_size;

    vec3 c = texture(display, uv).xyz;

    vec3 u = (
        -1.0 * texture(display, uv + vec2(-d.x, -d.y)).xyz +
        -2.0 * texture(display, uv + vec2(-d.x, 0.0)).xyz +
        -1.0 * texture(display, uv + vec2(-d.x, d.y)).xyz +
        +1.0 * texture(display, uv + vec2(d.x, -d.y)).xyz +
        +2.0 * texture(display, uv + vec2(d.x, 0.0)).xyz +
        +1.0 * texture(display, uv + vec2(d.x, d.y)).xyz
    ) / 4.0;

    vec3 v = (
        -1.0 * texture(display, uv + vec2(-d.x, -d.y)).xyz +
        -2.0 * texture(display, uv + vec2(0.0, -d.y)).xyz +
        -1.0 * texture(display, uv + vec2(d.x, -d.y)).xyz +
        +1.0 * texture(display, uv + vec2(-d.x, d.y)).xyz +
        +2.0 * texture(display, uv + vec2(0.0, d.y)).xyz +
        +1.0 * texture(display, uv + vec2(d.x, d.y)).xyz
    ) / 4.0;

    fragColor = vec4(dot(u, u), dot(v, v), dot(u, v), 1.0);
}
