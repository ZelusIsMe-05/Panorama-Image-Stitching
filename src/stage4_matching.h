// ============================================================================
// stage4_matching.h - Stage 4: brute-force Hamming matching + Lowe ratio test
//                     + cross-check (mutual nearest neighbour)
//
// Data flow:
//   for each descriptor of A (independent, one thread per query on the GPU):
//       scan all of B -> nearest j*, d1, second nearest d2
//   for each descriptor of B: nearest i in A (only needed by the cross-check)
//   keep (i, j*) if d1 < rho * d2 and nearest_A(j*) == i
// Ties are broken by the lowest index, so the result is deterministic and a
// parallel reduction with the same rule reproduces it.
// ============================================================================
#ifndef STAGE4_MATCHING_H
#define STAGE4_MATCHING_H

#include "common.h"

#include <cstdint>
#include <vector>

// ----------------------------------------------------------------------------
// Per-query building blocks (shared with the CUDA kernels)
// ----------------------------------------------------------------------------

/** Number of set bits of a 32-bit word (hardware __popc on the GPU). */
HOST_DEVICE inline int popcount32(uint32_t v) {
#if defined(__CUDA_ARCH__)
    return __popc(v);
#else
    v = v - ((v >> 1) & 0x55555555u);                  // pairs
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);  // nibbles
    v = (v + (v >> 4)) & 0x0F0F0F0Fu;                  // bytes
    return static_cast<int>((v * 0x01010101u) >> 24);  // sum of the 4 bytes
#endif
}

/** Hamming distance d_H(a, b) = popcount(a XOR b), in [0, 256]. */
HOST_DEVICE inline int hamming_distance(const Descriptor& a, const Descriptor& b) {
    int d = 0;
    for (int w = 0; w < 8; ++w) d += popcount32(a.bits[w] ^ b.bits[w]);
    return d;
}

/** Larger than any Hamming distance: "no candidate". */
constexpr int kNoDistance = 257;

/**
 * Scans the n descriptors of `set` for the query: best_idx = nearest (lowest
 * index on ties), d1 = its distance, d2 = smallest distance among the others
 * (kNoDistance if n < 2). best_idx = -1 if n == 0.
 */
HOST_DEVICE inline void nearest_two(const Descriptor& query, const Descriptor* set, int n,
                                    int& best_idx, int& d1, int& d2) {
    best_idx = -1;
    d1 = kNoDistance;
    d2 = kNoDistance;
    for (int j = 0; j < n; ++j) {
        const int d = hamming_distance(query, set[j]);
        if (d < d1) {
            d2 = d1;
            d1 = d;
            best_idx = j;
        } else if (d < d2) {
            d2 = d;
        }
    }
}

/** Lowe ratio test d1 / d2 < rho, written without a division (d2 = 0 always fails). */
HOST_DEVICE inline bool passes_ratio_test(int d1, int d2, float rho) {
    return static_cast<float>(d1) < rho * static_cast<float>(d2);
}

// ----------------------------------------------------------------------------
// Sequential implementations (V0)
// ----------------------------------------------------------------------------

/**
 * Stage 4 entry point: matches every descriptor of A against all descriptors
 * of B. Output ordered by idx_a. V1: one thread per descriptor of A (and one per
 * descriptor of B for the cross-check).
 */
void match_descriptors(const std::vector<Descriptor>& desc_a, const std::vector<Descriptor>& desc_b,
                       const StitchConfig& cfg, std::vector<Match>& matches);

/** Converts matches to point pairs (A point, B point), the input of Stage 5. */
void matches_to_points(const std::vector<Keypoint>& kps_a, const std::vector<Keypoint>& kps_b,
                       const std::vector<Match>& matches, std::vector<PointMatch>& points);

#endif // STAGE4_MATCHING_H
