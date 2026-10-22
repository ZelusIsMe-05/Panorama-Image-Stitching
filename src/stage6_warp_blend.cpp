// ============================================================================
// stage6_warp_blend.cpp - Stage 6 (sequential): cylindrical inverse warping +
//                         bilinear + distance-weighted feathering + crop
// ============================================================================
#include "stage6_warp_blend.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// Sanity limit, not a tuning parameter: a valid panorama of two overlapping photos
// is far smaller than this many times the input area.
static const double kMaxCanvasAreaFactor = 16.0;

bool invert_homography(const Homography& h, Homography& inv) {
    const float* m = h.m;
    const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], k = m[7], l = m[8];
    const double co[9] = {e * l - f * k, c * k - b * l, b * f - c * e,
                          f * g - d * l, a * l - c * g, c * d - a * f,
                          d * k - e * g, b * g - a * k, a * e - b * d};
    const double det = a * co[0] + b * co[3] + c * co[6];
    if (std::fabs(det) < 1e-12) return false;
    for (int i = 0; i < 9; ++i) inv.m[i] = static_cast<float>(co[i] / det);
    return true;
}

// Hc = T_A^-1 H T_B: H with both images' centers moved to the origin.
static void centered_homography(const Image& a, const Image& b, const Homography& h, double* hc) {
    const double cxa = 0.5 * (a.width - 1), cya = 0.5 * (a.height - 1);
    const double cxb = 0.5 * (b.width - 1), cyb = 0.5 * (b.height - 1);
    for (int c = 0; c < 3; ++c) {
        const double col[3] = {h.m[c], h.m[3 + c], h.m[6 + c]};
        hc[c] = col[0] - cxa * col[2];
        hc[3 + c] = col[1] - cya * col[2];
        hc[6 + c] = col[2];
    }
    for (int r = 0; r < 3; ++r) hc[r * 3 + 2] += cxb * hc[r * 3] + cyb * hc[r * 3 + 1];
}

// R = K^-1 Hc K scaled to det 1; false if Hc is singular.
static bool rotation_from_homography(const double* hc, double f, double* m) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            double v = hc[r * 3 + c];
            if (r == 2) v *= f;  // K^-1 scales row 2 by f relative to rows 0, 1
            if (c == 2) v /= f;  // K scales column 2 by 1/f relative to columns 0, 1
            m[r * 3 + c] = v;
        }
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                       m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (std::fabs(det) < 1e-12) return false;
    const double scale = 1.0 / std::cbrt(det);
    for (int i = 0; i < 9; ++i) m[i] *= scale;
    return true;
}

// || R R^T - I ||_F: how far K^-1 Hc K is from a rotation for this f.
static double rotation_error(const double* hc, double f) {
    double m[9];
    if (!rotation_from_homography(hc, f, m)) return 1e30;
    double err = 0.0;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            double dot = 0.0;
            for (int k = 0; k < 3; ++k) dot += m[r * 3 + k] * m[c * 3 + k];
            const double e = dot - (r == c ? 1.0 : 0.0);
            err += e * e;
        }
    return err;
}

float estimate_focal(const Image& a, const Image& b, const Homography& h) {
    double hc[9];
    centered_homography(a, b, h, hc);

    // Coarse log-spaced search, then a finer one around the best value.
    const double size = std::max(a.width, b.width);
    double best_f = size, best_err = rotation_error(hc, size);
    double lo = std::log(0.25 * size), hi = std::log(4.0 * size);
    for (int pass = 0; pass < 2; ++pass) {
        const int steps = 100;
        for (int i = 0; i <= steps; ++i) {
            const double f = std::exp(lo + (hi - lo) * i / steps);
            const double err = rotation_error(hc, f);
            if (err < best_err) { best_err = err; best_f = f; }
        }
        const double step = (hi - lo) / steps;
        lo = std::log(best_f) - step;
        hi = std::log(best_f) + step;
    }
    return static_cast<float>(best_f);
}

void cylinder_basis(const Image& a, const Image& b, const Homography& h, float focal, float* basis) {
    static const float kIdentity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::copy(kIdentity, kIdentity + 9, basis);
    double hc[9], r[9];
    centered_homography(a, b, h, hc);
    if (!rotation_from_homography(hc, focal, r)) return;

    // Rotation axis = 2 sin(angle) * unit axis; too short means no usable axis.
    double ey[3] = {r[7] - r[5], r[2] - r[6], r[3] - r[1]};
    const double len = std::sqrt(ey[0] * ey[0] + ey[1] * ey[1] + ey[2] * ey[2]);
    if (len < 1e-3) return;
    const double sign = ey[1] < 0.0 ? -1.0 : 1.0;  // point it down, like A's image y axis
    for (double& v : ey) v *= sign / len;
    if (ey[1] < 0.5) return;  // axis more than 60 deg from vertical: not a horizontal pan

    // e_z = A's optical axis (0, 0, 1) minus its component along e_y.
    double ez[3] = {-ey[2] * ey[0], -ey[2] * ey[1], 1.0 - ey[2] * ey[2]};
    const double lz = std::sqrt(ez[0] * ez[0] + ez[1] * ez[1] + ez[2] * ez[2]);
    if (lz < 1e-6) return;
    for (double& v : ez) v /= lz;
    const double ex[3] = {ey[1] * ez[2] - ey[2] * ez[1], ey[2] * ez[0] - ey[0] * ez[2],
                          ey[0] * ez[1] - ey[1] * ez[0]};
    for (int i = 0; i < 3; ++i) {
        basis[i] = static_cast<float>(ex[i]);
        basis[3 + i] = static_cast<float>(ey[i]);
        basis[6 + i] = static_cast<float>(ez[i]);
    }
}

// Cylinder coordinates (pixels) of the direction of homogeneous point (X, Y, W)
// on A's image plane.
static void plane_to_cylinder(double X, double Y, double W, const CylinderView& view,
                              double& x_cyl, double& y_cyl) {
    const double f = view.focal;
    const double r[3] = {(X - view.cx * W) / f, (Y - view.cy * W) / f, W};  // ray in A's camera frame
    const float* e = view.basis;
    const double rx = r[0] * e[0] + r[1] * e[1] + r[2] * e[2];  // ray in the cylinder frame
    const double ry = r[0] * e[3] + r[1] * e[4] + r[2] * e[5];
    const double rz = r[0] * e[6] + r[1] * e[7] + r[2] * e[8];
    x_cyl = f * std::atan2(rx, rz);
    y_cyl = f * ry / std::sqrt(rx * rx + rz * rz);
}

bool compute_canvas(const Image& a, const Image& b, const Homography& h,
                    CylinderView& view, int& width, int& height) {
    view.cx = 0.5f * static_cast<float>(a.width - 1);
    view.cy = 0.5f * static_cast<float>(a.height - 1);
    double x_min = 1e30, y_min = 1e30, x_max = -1e30, y_max = -1e30;
    auto add = [&](double xc, double yc) {
        x_min = std::min(x_min, xc); x_max = std::max(x_max, xc);
        y_min = std::min(y_min, yc); y_max = std::max(y_max, yc);
    };
    // Image borders become curves on the cylinder: sample them densely.
    const int samples = 64;
    for (int img = 0; img < 2; ++img) {
        const int w = img == 0 ? a.width : b.width, ht = img == 0 ? a.height : b.height;
        for (int i = 0; i <= samples; ++i) {
            const double t = static_cast<double>(i) / samples;
            const double pts[4][2] = {{t * (w - 1), 0.0}, {t * (w - 1), ht - 1.0},
                                      {0.0, t * (ht - 1)}, {w - 1.0, t * (ht - 1)}};
            for (const auto& p : pts) {
                double X = p[0], Y = p[1], W = 1.0;
                if (img == 1) {  // B -> A's plane through H
                    const float* m = h.m;
                    X = m[0] * p[0] + m[1] * p[1] + m[2];
                    Y = m[3] * p[0] + m[4] * p[1] + m[5];
                    W = m[6] * p[0] + m[7] * p[1] + m[8];
                }
                double xc, yc;
                plane_to_cylinder(X, Y, W, view, xc, yc);
                add(xc, yc);
            }
        }
    }
    const double w = std::floor(x_max) - std::floor(x_min) + 1.0;
    const double ht = std::floor(y_max) - std::floor(y_min) + 1.0;
    const double input_area = static_cast<double>(a.width) * a.height + static_cast<double>(b.width) * b.height;
    if (!(w * ht <= kMaxCanvasAreaFactor * input_area)) return false;

    view.origin_x = static_cast<float>(std::floor(x_min));
    view.origin_y = static_cast<float>(std::floor(y_min));
    width = static_cast<int>(w);
    height = static_cast<int>(ht);
    return true;
}

bool largest_covered_rect(const uint8_t* mask, int width, int height,
                          int& x0, int& y0, int& rect_width, int& rect_height) {
    std::vector<int> run(static_cast<size_t>(width), 0);  // covered pixels ending at row y, per column
    std::vector<int> stack;
    long long best = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x)
            run[x] = mask[static_cast<size_t>(y) * width + x] ? run[x] + 1 : 0;
        stack.clear();
        for (int x = 0; x <= width; ++x) {
            const int cur = x < width ? run[x] : 0;
            while (!stack.empty() && run[stack.back()] >= cur) {
                const int bar = run[stack.back()];
                stack.pop_back();
                const int left = stack.empty() ? 0 : stack.back() + 1;
                const long long area = static_cast<long long>(bar) * (x - left);
                if (area > best) {
                    best = area;
                    x0 = left; rect_width = x - left;
                    y0 = y - bar + 1; rect_height = bar;
                }
            }
            stack.push_back(x);
        }
    }
    return best > 0;
}

bool warp_and_blend(const Image& a, const Image& b, const Homography& h, Image& panorama) {
    if (!a.is_valid() || !b.is_valid()) return false;
    Homography h_inv;
    if (!invert_homography(h, h_inv)) return false;
    CylinderView view;
    view.focal = estimate_focal(a, b, h);
    cylinder_basis(a, b, h, view.focal, view.basis);
    int width = 0, height = 0;
    if (!compute_canvas(a, b, h, view, width, height)) return false;

    // Coverage mask of the whole canvas, then the largest rectangle with no empty pixel.
    std::vector<uint8_t> mask(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            mask[static_cast<size_t>(y) * width + x] =
                is_covered(view, h_inv.m, x, y, a.width, a.height, b.width, b.height) ? 1 : 0;
    int x0 = 0, y0 = 0, crop_width = 0, crop_height = 0;
    if (!largest_covered_rect(mask.data(), width, height, x0, y0, crop_width, crop_height)) return false;
    view.origin_x += static_cast<float>(x0);
    view.origin_y += static_cast<float>(y0);

    panorama = Image(crop_width, crop_height, 3);
    uint8_t* out = panorama.data.data();
    // One independent output pixel per iteration (one thread per pixel on the GPU).
    for (int y = 0; y < crop_height; ++y)
        for (int x = 0; x < crop_width; ++x)
            warp_blend_pixel(a.data.data(), a.width, a.height, a.channels,
                             b.data.data(), b.width, b.height, b.channels,
                             h_inv.m, view, x, y,
                             out + (static_cast<size_t>(y) * crop_width + x) * 3);
    return true;
}
