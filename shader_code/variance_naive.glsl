#include "config.glsl"
#iChannel1 "TODO: TRANSFORMATION TEXTURE/BUFFER"
#iChannel2 "TODO: KERNEL TEXTURE/BUFFER"


float alpha = 1.0;
const int N = 8; // number of orientations

void varianceNaive() {
    vec2 src_size = iResolution.xy;
    vec2 uv = fragCoord.xy / src_size;
    
    vec4 t = texture(iChannel1, uv); 

    // ellipse parameters
    float a = RAD * clamp((alpha + t.w) / alpha, 0.1, 2.0);
    float b = RAD * clamp(alpha / (alpha + t.w), 0.1, 2.0);
    float cos_phi = cos(t.z);
    float sin_phi = sin(t.z);

    // rotation and scaling matrices
    mat2 R = mat2(cos_phi, -sin_phi, sin_phi, cos_phi);
    mat2 S = mat2(0.5 / a, 0.0, 0.0, 0.5 / b);
    mat2 SR = S * R;

    // determine bounds for the ellipse
    int max_x = int(sqrt(a * a * cos_phi * cos_phi + b * b * sin_phi * sin_phi));
    int max_y = int(sqrt(a * a * sin_phi * sin_phi + b * b * cos_phi * cos_phi));

    vec4 m[N];
    vec3 s[N];
    for (int k = 0; k < N; ++k) {
        m[k] = vec4(0.0);
        s[k] = vec3(0.0);
    }

    // loop through neighborhood
    for (int j = -max_y; j <= max_y; ++j) {
        for (int i = -max_x; i <= max_x; ++i) {
            vec2 v = SR * vec2(float(i), float(j));
        
            // check if pixel is inside the ellipse
            if (dot(v, v) <= 0.25) {
                vec3 c = texture(iChannel0, uv + vec2(float(i), float(j)) / src_size).rgb;
                
                // loop through orientations
                for (int k = 0; k < N; ++k) {
                    float w = texture(iChannel2, vec2(0.5, 0.5) + v).x;

                    m[k] += vec4(c * w, w);
                    s[k] += c * c * w;
                    v *= mat2(cos(2.0 * 3.14159265359 / float(N)), -sin(2.0 * 3.14159265359 / float(N)),
                              sin(2.0 * 3.14159265359 / float(N)), cos(2.0 * 3.14159265359 / float(N)));
                }
            }
        }
    }
}
