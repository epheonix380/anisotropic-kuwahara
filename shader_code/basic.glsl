#include "config.glsl"

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 src_size = iResolution.xy;
    vec2 uv = fragCoord.xy / src_size;
    int radius = RAD;
    float n = float((radius + 1) * (radius + 1));

    vec3 m[4];
    vec3 s[4];
    for (int k = 0; k < 4; ++k) {
        m[k] = vec3(0.0);
        s[k] = vec3(0.0);
    }

    Window W[4] = Window[4](
        Window(-radius, -radius, 0, 0),
        Window(0, -radius, radius, 0),
        Window(0, 0, radius, radius),
        Window(-radius, 0, 0, radius)
    );

    for (int k = 0; k < 4; ++k) {
        for (int j = W[k].y1; j < W[k].y2; ++j) {
            for (int i = W[k].x1; i < W[k].x2; ++i) {
                vec3 c = texture(iChannel0, uv + vec2(float(i), float(j)) / src_size).rgb;
                m[k] += c;
                s[k] += c * c;
            }
        }
    }

    float min_sigma2 = 1e+2;
    vec3 finalColor = vec3(0.0);
    for (int k = 0; k < 4; ++k) {
        m[k] /= n;
        s[k] = abs(s[k] / n - m[k] * m[k]);
        float sigma2 = s[k].r + s[k].g + s[k].b;
        if (sigma2 < min_sigma2) {
            min_sigma2 = sigma2;
            finalColor = m[k];
        }
    }

    fragColor = vec4(finalColor, 1.0);
}