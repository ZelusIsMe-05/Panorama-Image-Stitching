// ============================================================================
// stage3_orb.cpp - Stage 3 (sequential): ORB descriptor (rotated BRIEF)
// ============================================================================
#include "stage3_orb.h"

#include <algorithm>

namespace {

// Small deterministic integer generator built on pcg_hash.
struct PatternRng {
    uint32_t state;
    uint32_t next() { state = pcg_hash(state); return state; }
    // Uniform integer in [-k, k].
    int uniform(int k) { return static_cast<int>(next() % static_cast<uint32_t>(2 * k + 1)) - k; }
};

}  // namespace

std::vector<int8_t> make_orb_pattern(const StitchConfig& cfg) {
    const int half = std::max(cfg.orb_patch_size / 2, 1);
    const int bins = std::max(cfg.orb_angle_bins, 1);

    // Gaussian with sigma = S / 5 approximated by the sum of 3 uniform integers in
    // [-k, k] (variance k(k + 1)), so the pattern does not depend on floating point.
    const double sigma = cfg.orb_patch_size / 5.0;
    const int k = std::max(1, static_cast<int>(std::lround(std::sqrt(sigma * sigma + 0.25) - 0.5)));
    PatternRng rng{pcg_hash(cfg.orb_pattern_seed)};
    auto draw_point = [&](int& x, int& y) {
        do {  // points stay inside the disc of radius S/2, so every rotation stays inside the patch
            x = rng.uniform(k) + rng.uniform(k) + rng.uniform(k);
            y = rng.uniform(k) + rng.uniform(k) + rng.uniform(k);
        } while (x * x + y * y > half * half);
    };
    std::vector<int> base(kOrbBits * 4);
    for (int i = 0; i < kOrbBits; ++i) {
        int* p = &base[i * 4];
        do {
            draw_point(p[0], p[1]);
            draw_point(p[2], p[3]);
        } while (p[0] == p[2] && p[1] == p[3]);
    }

    // Pre-rotated copies, one per angle bin (steered BRIEF).
    std::vector<int8_t> pattern(static_cast<size_t>(bins) * kOrbBits * 4);
    for (int b = 0; b < bins; ++b) {
        const double theta = 2.0 * 3.14159265358979323846 * b / bins;
        const double c = std::cos(theta), s = std::sin(theta);
        for (int i = 0; i < kOrbBits * 2; ++i) {  // every point of every pair
            const double x = base[i * 2], y = base[i * 2 + 1];
            const long rx = std::lround(c * x - s * y), ry = std::lround(s * x + c * y);
            int8_t* out = &pattern[(static_cast<size_t>(b) * kOrbBits * 2 + i) * 2];
            out[0] = static_cast<int8_t>(std::min<long>(std::max<long>(rx, -half), half));
            out[1] = static_cast<int8_t>(std::min<long>(std::max<long>(ry, -half), half));
        }
    }
    return pattern;
}

void compute_descriptors(const GrayImage& img, std::vector<Keypoint>& kps, const StitchConfig& cfg,
                         std::vector<Descriptor>& desc) {
    desc.assign(kps.size(), Descriptor{});
    if (!img.is_valid() || kps.empty()) return;
    const std::vector<int8_t> pattern = make_orb_pattern(cfg);
    const int bins = std::max(cfg.orb_angle_bins, 1);
    const float* src = img.data.data();

    // One independent keypoint per iteration (one thread per keypoint on the GPU).
    for (size_t i = 0; i < kps.size(); ++i) {
        const int cx = static_cast<int>(std::lround(kps[i].x));
        const int cy = static_cast<int>(std::lround(kps[i].y));
        const float theta = ic_angle(src, img.width, img.height, cx, cy, cfg.orb_centroid_radius);
        kps[i].angle = theta;
        const int8_t* bin_pattern = &pattern[static_cast<size_t>(angle_bin(theta, bins)) * kOrbBits * 4];
        orb_descriptor_at(src, img.width, img.height, cx, cy, bin_pattern, desc[i]);
    }
}
