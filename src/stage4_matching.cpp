// ============================================================================
// stage4_matching.cpp - Stage 4 (sequential): brute-force Hamming + ratio test
//                       + cross-check
// ============================================================================
#include "stage4_matching.h"

#include <cstddef>

void match_descriptors(const std::vector<Descriptor>& desc_a, const std::vector<Descriptor>& desc_b,
                       const StitchConfig& cfg, std::vector<Match>& matches) {
    matches.clear();
    const int na = static_cast<int>(desc_a.size()), nb = static_cast<int>(desc_b.size());
    if (na == 0 || nb == 0) return;

    // Cross-check needs the nearest neighbour of every B descriptor in A.
    std::vector<int> nearest_in_a;
    if (cfg.match_cross_check) {
        nearest_in_a.resize(nb);
        for (int j = 0; j < nb; ++j) {
            int d1, d2;
            nearest_two(desc_b[j], desc_a.data(), na, nearest_in_a[j], d1, d2);
        }
    }

    // A -> B: nearest and second nearest, then the two filters.
    for (int i = 0; i < na; ++i) {
        int j, d1, d2;
        nearest_two(desc_a[i], desc_b.data(), nb, j, d1, d2);
        if (!passes_ratio_test(d1, d2, cfg.match_ratio)) continue;
        if (cfg.match_cross_check && nearest_in_a[j] != i) continue;
        matches.push_back(Match{i, j, d1});
    }
}

void matches_to_points(const std::vector<Keypoint>& kps_a, const std::vector<Keypoint>& kps_b,
                       const std::vector<Match>& matches, std::vector<PointMatch>& points) {
    points.resize(matches.size());
    for (size_t k = 0; k < matches.size(); ++k) {
        const Keypoint& a = kps_a[static_cast<size_t>(matches[k].idx_a)];
        const Keypoint& b = kps_b[static_cast<size_t>(matches[k].idx_b)];
        points[k] = PointMatch{a.x, a.y, b.x, b.y};
    }
}
