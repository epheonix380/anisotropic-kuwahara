// src image

uniform sampler2D display1;
uniform sampler2D display2;
uniform vec2 res;
uniform int RAD;
#define PI 3.14159265358979323846

struct Window {
    int x1, y1, x2, y2;
};

const int N = 8;
const float q = 2.0;

void main() {
    vec2 src_size = res.xy;
    vec2 uv = gl_FragCoord.xy / src_size;

    vec4 m[8];
    vec3 s[8];
    for (int k = 0; k < N; ++k) {
        m[k] = vec4(0.0);
        s[k] = vec3(0.0);
    }

    float piN = 2.0 * PI / float(N);
    mat2 X = mat2(cos(piN), sin(piN), -sin(piN), cos(piN));

    for (int j = -RAD; j <= RAD; ++j) {
        for (int i = -RAD; i <= RAD; ++i) {
            vec2 v = 0.5 * vec2(float(i), float(j)) / float(RAD);
            if (dot(v, v) <= 0.25) {
                vec3 c = texture(display1, uv + vec2(float(i), float(j)) / src_size).rgb;
                for (int k = 0; k < N; ++k) {

                    float w = texture(display2, vec2(0.5, 0.5) + v).x;
                    m[k] += vec4(c * w, w);
                    s[k] += c * c * w;
                    v *= X;
                }
            }
        }
    }

    vec4 o = vec4(0.0);
    for (int k = 0; k < N; ++k) {
        m[k].rgb /= m[k].w;
        s[k] = abs(s[k] / m[k].w - m[k].rgb * m[k].rgb);
        float sigma2 = s[k].r + s[k].g + s[k].b;
        float w = 1.0 / (1.0 + pow(255.0 * sigma2, 0.5 * q));
        o += vec4(m[k].rgb * w, w);
    }

    frag_col = vec4(o.rgb / o.w, 1.0);
}
