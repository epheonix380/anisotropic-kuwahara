void varianceOptimized() {
    // central pixel contribution
    vec3 c = texture(iChannel0, uv).rgb;
    float w = texture(iChannel1, vec2(0.5, 0.5)).x;
    for (int k = 0; k < N; ++k) {
        m[k] += vec4(c * w, w);
        s[k] += c * c * w;
    }

    // loop through the neighborhood
    for (int j = 0; j <= max_y; ++j) {
        for (int i = -max_x; i <= max_x; ++i) {
            // skip the origin except for positive x
            if ((j != 0) || (i > 0)) {
                vec2 v = SR * vec2(float(i), float(j));
                
                // check if the point is inside the ellipse
                if (dot(v, v) <= 0.25) {
                    // symmetry points
                    vec3 c0 = texture(iChannel0, uv + vec2(float(i), float(j)) / src_size).rgb;
                    vec3 c1 = texture(iChannel0, uv - vec2(float(i), float(j)) / src_size).rgb;
                    
                    // squared colors
                    vec3 cc0 = c0 * c0;
                    vec3 cc1 = c1 * c1;

                    // iChannel1 is the transformation texture
                    // weights for 0123
                    vec4 w0123 = texture(iChannel1, vec2(0.5, 0.5) + v);
                    for (int k = 0; k < 4; ++k) {
                        m[k] += vec4(c0 * w0123[k], w0123[k]);
                        s[k] += cc0 * w0123[k];
                        m[k + 4] += vec4(c1 * w0123[k], w0123[k]);
                        s[k + 4] += cc1 * w0123[k];
                    }

                    // weights for 4567
                    vec4 w4567 = texture(iChannel1, vec2(0.5, 0.5) - v);
                    for (int k = 0; k < 4; ++k) {
                        m[k + 4] += vec4(c0 * w4567[k], w4567[k]);
                        s[k + 4] += cc0 * w4567[k];
                        m[k] += vec4(c1 * w4567[k], w4567[k]);
                        s[k] += cc1 * w4567[k];
                    }
                }
            }
        }
    }
}
