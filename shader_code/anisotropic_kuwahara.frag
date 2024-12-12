#version 330 core

uniform sampler2D display;
uniform sampler2D src;
uniform vec2 res;
uniform int N ;
uniform int radius;
uniform float q ;
const float PI = 3.14159265358979323846;

in vec2 fragCoord;
out vec4 fragColor;


void main (void) {
    vec2 src_size = res.xy;
    vec2 uv = fragCoord.xy / src_size;
    vec4 m[8];
    vec3 s[8];
    for ( int k = 0; k < N; ++k) {
        m[k] = vec4 (0.0);
        s[k] = vec3 (0.0);
    }
    float piN = 2.0 * PI / float (N);
    mat2 X = mat2 (cos(piN), sin(piN), -sin(piN), cos(piN));
    for ( int j = -radius; j <= radius; ++j ) {
        for ( int i = -radius; i <= radius; ++i ) {
            vec2 v = 0.5 * vec2 (i,j) / float (radius);
            if ( dot (v,v) <= 0.25) {
                vec3 c = texture (src , uv + vec2 (i,j) / src_size).rgb;
                for ( int k = 0; k < N; ++k) {
                    float w = texture (display, vec2 (0.5, 0.5) + v).x;
                    m[k] += vec4 (c * w, w);
                    s[k] += c * c * w;
                    v *= X;
                }
            }
        }
    }
    vec4 o = vec4 (0.0);
    for ( int k = 0; k < N; ++k) {
        m[k].rgb /= m[k].w;
        s[k] = abs(s[k] / m[k].w - m[k].rgb * m[k].rgb);
        float sigma2 = s[k].r + s[k].g + s[k].b;
        float w = 1.0 / (1.0 + pow(255.0 * sigma2 , 0.5 * q));
        o += vec4 (m[k].rgb * w, w);
    }
    fragColor = vec4 (o.rgb / o.w, 1.0);
}