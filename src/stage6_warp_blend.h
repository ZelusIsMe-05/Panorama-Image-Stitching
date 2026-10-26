// ============================================================================
// stage6_warp_blend.h - Stage 6: cylindrical inverse warping + bilinear
//                       interpolation + distance-weighted feathering + crop
//
// Author  : Nguyen Dang Pha (23120315)
// Sources : M. Brown, D. G. Lowe, IJCV 2007 (cylindrical projection, see
//           below). Implemented from scratch (no external code).
//
// Why a cylinder: H maps B onto the image PLANE of A. When the camera rotates
// by a large angle, the part of B far from A's center is stretched without
// bound on that plane. The panorama is therefore drawn on a cylinder around
// A's camera, where every viewing direction gets the same pixel density.
// The cylinder axis is the rotation axis of the camera between the two shots
// (not A's own "up" direction): if the phone was slightly tilted while
// panning, both images still lie side by side at the same height on the
// cylinder, so the crop keeps more of their height ("wave correction",
// M. Brown and D. Lowe, "Automatic Panoramic Image Stitching using Invariant
// Features", IJCV 2007, Sec. 5).
//
// Data flow:
//   focal length f from H (the f for which R = K^-1 H K is closest to a rotation)
//   -> cylinder axis = rotation axis of R
//   -> project the borders of A and B onto the cylinder -> canvas bounding box
//   -> coverage mask (pixels seen by A or B) -> largest fully covered rectangle
//   -> for every pixel of that rectangle (independent "gather"): cylinder
//      direction -> point on A's plane -> sample A, and B at H^-1 of that point,
//      bilinear; blend by the distance of each sample to its own image border.
// warp_blend_pixel() does all of it for one output pixel, i.e. one fused kernel
// with one thread per output pixel and no write conflicts.
// ============================================================================
#ifndef STAGE6_WARP_BLEND_H
#define STAGE6_WARP_BLEND_H

#include "common.h"

#include <cmath>
#include <cstdint>

// ----------------------------------------------------------------------------
// Per-pixel building blocks (shared with the CUDA kernels)
// ----------------------------------------------------------------------------

/** Channel c (0..2) of pixel (x, y); 1- and 2-channel images are replicated to RGB. */
HOST_DEVICE inline float fetch_channel(const uint8_t* img, int width, int channels, int x, int y, int c) {
    const int ch = channels >= 3 ? c : 0;
    return static_cast<float>(img[(static_cast<size_t>(y) * width + x) * channels + ch]);
}

/**
 * Bilinear interpolation of channel c at real coordinates (u, v), which must lie in
 * [0, width - 1] x [0, height - 1]:
 *   I = (1-du)(1-dv) I00 + du (1-dv) I10 + (1-du) dv I01 + du dv I11
 */
HOST_DEVICE inline float bilinear_sample(const uint8_t* img, int width, int height, int channels,
                                         float u, float v, int c) {
    int u0 = static_cast<int>(u), v0 = static_cast<int>(v);  // floor, since u, v >= 0
    if (u0 > width - 1) u0 = width - 1;
    if (v0 > height - 1) v0 = height - 1;
    const int u1 = u0 + 1 < width ? u0 + 1 : u0;
    const int v1 = v0 + 1 < height ? v0 + 1 : v0;
    const float du = u - static_cast<float>(u0), dv = v - static_cast<float>(v0);
    return (1.0f - du) * (1.0f - dv) * fetch_channel(img, width, channels, u0, v0, c) +
           du * (1.0f - dv) * fetch_channel(img, width, channels, u1, v0, c) +
           (1.0f - du) * dv * fetch_channel(img, width, channels, u0, v1, c) +
           du * dv * fetch_channel(img, width, channels, u1, v1, c);
}

/**
 * Feathering weight: Euclidean distance from (u, v) to the nearest border of
 * its own width x height image, whose pixels cover [-0.5, size - 0.5]. Always
 * > 0 inside the image, so a pixel covered by one image keeps its own color.
 */
HOST_DEVICE inline float border_distance(float u, float v, int width, int height) {
    float d = u + 0.5f;
    const float right = static_cast<float>(width) - 0.5f - u;
    const float top = v + 0.5f;
    const float bottom = static_cast<float>(height) - 0.5f - v;
    if (right < d) d = right;
    if (top < d) d = top;
    if (bottom < d) d = bottom;
    return d;
}

/**
 * Maps output pixel (x, y) to the cylinder and back to image A and image B.
 * Cylinder coordinates in pixels: x_cyl = f * theta, y_cyl = f * h.
 * Pixel (x, y) has x_cyl = x + origin_x, y_cyl = y + origin_y.
 */
struct CylinderView {
    float focal;     // f in pixels (same for A and B, one camera)
    float cx, cy;    // principal point of A (image center)
    float origin_x;  // cylinder coordinates of output pixel (0, 0)
    float origin_y;
    float basis[9];  // rows e_x, e_y (cylinder axis), e_z of the cylinder frame, in A's camera frame
};

/**
 * Viewing direction of output pixel (x, y): r = sin theta e_x + h e_y + cos theta e_z
 * (A's camera frame). Its point on A's image plane in homogeneous coordinates is
 * K r = (f r_x + cx r_z, f r_y + cy r_z, r_z). A sees the pixel at (xa, ya) when
 * r_z > 0; B sees it at (u, v) = H^-1 (K r). A weight is 0 when the image does
 * not cover the pixel, else its feathering weight.
 */
HOST_DEVICE inline void cylinder_sources(const CylinderView& view, const float* h_inv, int x, int y,
                                         int wa, int ha, int wb, int hb,
                                         float& xa, float& ya, float& weight_a,
                                         float& u, float& v, float& weight_b) {
    const float theta = (static_cast<float>(x) + view.origin_x) / view.focal;
    const float h = (static_cast<float>(y) + view.origin_y) / view.focal;
    const float s = sinf(theta), c = cosf(theta);
    const float* e = view.basis;
    const float rx = s * e[0] + h * e[3] + c * e[6];
    const float ry = s * e[1] + h * e[4] + c * e[7];
    const float rz = s * e[2] + h * e[5] + c * e[8];
    const float px = view.focal * rx + view.cx * rz;
    const float py = view.focal * ry + view.cy * rz;
    const float pw = rz;

    weight_a = 0.0f;
    xa = ya = 0.0f;
    if (pw > 1e-6f) {
        xa = px / pw;
        ya = py / pw;
        if (xa >= 0.0f && ya >= 0.0f && xa <= static_cast<float>(wa - 1) && ya <= static_cast<float>(ha - 1))
            weight_a = border_distance(xa, ya, wa, ha);
    }

    weight_b = 0.0f;
    u = v = 0.0f;
    const float w = h_inv[6] * px + h_inv[7] * py + h_inv[8] * pw;
    if (w > 1e-8f) {
        u = (h_inv[0] * px + h_inv[1] * py + h_inv[2] * pw) / w;
        v = (h_inv[3] * px + h_inv[4] * py + h_inv[5] * pw) / w;
        if (u >= 0.0f && v >= 0.0f && u <= static_cast<float>(wb - 1) && v <= static_cast<float>(hb - 1))
            weight_b = border_distance(u, v, wb, hb);
    }
}

/** True if A or B covers output pixel (x, y) (coverage mask used for the crop). */
HOST_DEVICE inline bool is_covered(const CylinderView& view, const float* h_inv, int x, int y,
                                   int wa, int ha, int wb, int hb) {
    float xa, ya, weight_a, u, v, weight_b;
    cylinder_sources(view, h_inv, x, y, wa, ha, wb, hb, xa, ya, weight_a, u, v, weight_b);
    return weight_a > 0.0f || weight_b > 0.0f;
}

/**
 * One output pixel (x, y): samples A at (xa, ya) and B at (u, v) (bilinear) and
 * blends them as I = (d_A I_A + d_B I_B) / (d_A + d_B); black where no image
 * covers the pixel. h_inv maps image A coordinates to image B coordinates.
 * out points to 3 bytes (RGB).
 */
HOST_DEVICE inline void warp_blend_pixel(const uint8_t* a, int wa, int ha, int ca,
                                         const uint8_t* b, int wb, int hb, int cb,
                                         const float* h_inv, const CylinderView& view,
                                         int x, int y, uint8_t* out) {
    float xa, ya, weight_a, u, v, weight_b;
    cylinder_sources(view, h_inv, x, y, wa, ha, wb, hb, xa, ya, weight_a, u, v, weight_b);

    const float total = weight_a + weight_b;
    for (int c = 0; c < 3; ++c) {
        float value = 0.0f;
        if (total > 0.0f) {
            const float color_a = weight_a > 0.0f ? bilinear_sample(a, wa, ha, ca, xa, ya, c) : 0.0f;
            const float color_b = weight_b > 0.0f ? bilinear_sample(b, wb, hb, cb, u, v, c) : 0.0f;
            value = (weight_a * color_a + weight_b * color_b) / total;
        }
        value += 0.5f;  // round to nearest
        out[c] = static_cast<uint8_t>(value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value));
    }
}

// ----------------------------------------------------------------------------
// Sequential implementations (V0)
// ----------------------------------------------------------------------------

/** Inverse of a 3x3 matrix in double precision; false if singular. */
bool invert_homography(const Homography& h, Homography& inv);

/**
 * Focal length (pixels) of the camera, from H alone: with image centers as
 * principal points, a pure rotation gives H ~ K R K^-1 with K = diag(f, f, 1).
 * Returns the f in [0.25, 4] x max(width) for which K^-1 H K (scaled to det 1)
 * is closest to a rotation, i.e. minimizes || M M^T - I ||_F.
 */
float estimate_focal(const Image& a, const Image& b, const Homography& h);

/**
 * Cylinder frame (rows e_x, e_y, e_z, in A's camera frame): e_y is the rotation
 * axis of R = K^-1 H K, i.e. (R32 - R23, R13 - R31, R21 - R12), oriented like A's
 * image y axis; e_z is A's optical axis made orthogonal to e_y; e_x = e_y x e_z.
 * Falls back to A's own axes when the rotation is too small to define an axis
 * or the axis is more than 60 degrees from vertical (not a horizontal pan).
 */
void cylinder_basis(const Image& a, const Image& b, const Homography& h, float focal, float* basis);

/**
 * Cylinder bounding box of A and of the projected borders of B (h maps B -> A).
 * view (with focal and basis already set) gets A's center and the cylinder
 * coordinates of canvas pixel (0, 0). Returns false if the canvas is
 * unreasonably large (a broken homography), so that Stage 6 never allocates a
 * huge image.
 */
bool compute_canvas(const Image& a, const Image& b, const Homography& h,
                    CylinderView& view, int& width, int& height);

/**
 * Largest axis-aligned rectangle of covered pixels in a width x height mask
 * (row-major, 1 = covered), via the "largest rectangle in a histogram" stack
 * algorithm on every row, O(width * height). Returns false if nothing is covered.
 */
bool largest_covered_rect(const uint8_t* mask, int width, int height,
                          int& x0, int& y0, int& rect_width, int& rect_height);

/**
 * Stage 6 entry point: warps A and B onto the cylinder, blends them and, if
 * crop is true, crops the panorama to the largest rectangle with no empty
 * (black) pixel; otherwise the whole canvas is kept with black where no image
 * covers it. Returns false if the canvas cannot be built from h.
 */
bool warp_and_blend(const Image& a, const Image& b, const Homography& h, Image& panorama,
                    bool crop = true);

#endif // STAGE6_WARP_BLEND_H
