#version 330 core

uniform sampler2D display;
uniform vec2 res;
uniform int radius;
#define PI 3.14159265358979323846

in vec2 fragCoord;
out vec4 fragColor;

void main() {
    vec2 src_size = res.xy;
    vec2 uv = gl_FragCoord.xy / src_size;
    float n = float ((radius + 1) * (radius + 1));
    vec3 m[4];
    vec3 s[4];
    for ( int k = 0; k < 4; ++k) {
        m[k] = vec3 (0.0);
        s[k] = vec3 (0.0);
    }
    struct Window { int x1, y1, x2 , y2; };
    Window W[4] = Window[4](
    Window( -radius , -radius , 0 , 0 ),
    Window( 0, -radius , radius , 0 ),
    Window( 0 , 0, radius , radius ),
    Window( -radius , 0 , 0, radius )
    );
    for ( int k = 0; k < 4; ++k) {
        for ( int j = W[k].y1; j <= W[k].y2; ++j) {
            for ( int i = W[k].x1; i <= W[k].x2; ++i) {
                vec3 c = texture (display , uv + vec2 (i,j) / src_size).rgb;
                m[k] += c;
                s[k] += c * c;
            }
        }
    }
    float min_sigma2 = 1e+2;
    for ( int k = 0; k < 4; ++k) {
        m[k] /= n;
        s[k] = abs(s[k] / n - m[k] * m[k]);
        float sigma2 = s[k].r + s[k].g + s[k].b;
        if (sigma2 < min_sigma2) {
            min_sigma2 = sigma2;
            gl_FragColor = vec4 (m[k], 1.0);
        }
    }
}
