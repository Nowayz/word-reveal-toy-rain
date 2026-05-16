#include <mkl.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef MKL8GON_USE_CUDA
#include "visibility_cuda.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mkl8gon {

constexpr int K = 8;
constexpr int FRUITLESS_ROUND_PATIENCE = 10;
constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double INF = 1.0e100;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;

    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}

    Vec2 operator+(const Vec2& b) const { return {x + b.x, y + b.y}; }
    Vec2 operator-(const Vec2& b) const { return {x - b.x, y - b.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2 operator/(double s) const { return {x / s, y / s}; }
    Vec2& operator+=(const Vec2& b) { x += b.x; y += b.y; return *this; }
};

static inline double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
static inline double cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }
static inline double norm2(Vec2 a) { return dot(a, a); }
static inline double norm(Vec2 a) { return std::sqrt(norm2(a)); }
static inline Vec2 lerp(Vec2 a, Vec2 b, double t) { return a * (1.0 - t) + b * t; }
static inline Vec2 perp(Vec2 a) { return {-a.y, a.x}; }

struct Poly8 {
    std::array<Vec2, K> v{};
};

struct PolyN {
    std::vector<Vec2> v;
};

struct EdgeSample {
    int edge = 0;
    double t = 0.5;
};

struct Options {
    int alphaThreshold = 10;
    int starts = 48;
    int seed = 1;
    int maxActiveObject = 1400;
    int edgeSamplesPerEdge = 40;
    int maxOuterPerClearance = 6;
    int mklMaxIterations = 160;
    int mklMaxTrialStepIterations = 80;
    double denseEdgeStep = 0.35;
    double finalClearance = 0.0;
    double initialMargin = 8.0;
    double startVariance = 32.0;
    double angleVariance = 0.18;
    double minGainPct = 0.05;
    double pointGainPct = 0.1;
    double gaMutation = 3.0;
    int maxPoints = 24;
    int gaPopulation = 96;
    int gaGenerations = 80;
    int pointCandidates = 128;
    int minPoints = 3;
    int pointPatience = 5;
    int cpuThreads = 0;
    bool autoPoints = false;
    bool genetic = false;
    bool pointVisibility = false;
    bool useCuda = true;
    bool fastClearances = false;
    bool polish = true;
    bool verbose = true;
    bool showUi = true;
};

static double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

static double softplus(double z, double beta = 4.0) {
    const double bz = beta * z;
    if (bz > 40.0) return z;
    if (bz < -40.0) return std::exp(bz) / beta;
    return std::log1p(std::exp(bz)) / beta;
}

static std::string fmt(Vec2 p) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << "[" << p.x << ", " << p.y << "]";
    return os.str();
}

static std::string numberedOutputPath(const std::string& outputPath, int vertices) {
    const size_t dot = outputPath.find_last_of('.');
    std::ostringstream suffix;
    suffix << "-points-" << vertices;
    if (dot == std::string::npos) return outputPath + suffix.str() + ".png";
    return outputPath.substr(0, dot) + suffix.str() + outputPath.substr(dot);
}

static double signedArea(const Poly8& p) {
    double a = 0.0;
    for (int i = 0; i < K; ++i) {
        const Vec2& u = p.v[i];
        const Vec2& v = p.v[(i + 1) % K];
        a += cross(u, v);
    }
    return 0.5 * a;
}

static double area(const Poly8& p) {
    return std::abs(signedArea(p));
}

static double signedArea(const PolyN& p) {
    double a = 0.0;
    const int n = static_cast<int>(p.v.size());
    for (int i = 0; i < n; ++i) {
        a += cross(p.v[i], p.v[(i + 1) % n]);
    }
    return 0.5 * a;
}

static double area(const PolyN& p) {
    return std::abs(signedArea(p));
}

static Vec2 centroidOfPoints(const std::vector<Vec2>& pts) {
    if (pts.empty()) return {0.0, 0.0};
    Vec2 c{0.0, 0.0};
    for (auto p : pts) c += p;
    return c / static_cast<double>(pts.size());
}

static Poly8 ensureCCW(Poly8 p) {
    if (signedArea(p) < 0.0) {
        std::reverse(p.v.begin(), p.v.end());
    }
    return p;
}

static PolyN ensureCCW(PolyN p) {
    if (signedArea(p) < 0.0) {
        std::reverse(p.v.begin(), p.v.end());
    }
    return p;
}

static Poly8 lerpPoly(const Poly8& a, const Poly8& b, double t) {
    Poly8 r;
    for (int i = 0; i < K; ++i) r.v[i] = lerp(a.v[i], b.v[i], t);
    return r;
}

static std::vector<double> encodeRawXY(const Poly8& p) {
    std::vector<double> x(2 * K);
    for (int i = 0; i < K; ++i) {
        x[2 * i + 0] = p.v[i].x;
        x[2 * i + 1] = p.v[i].y;
    }
    return x;
}

static Poly8 decodeRawXY(const double* x) {
    Poly8 p;
    for (int i = 0; i < K; ++i) {
        p.v[i] = {x[2 * i + 0], x[2 * i + 1]};
    }
    return p;
}

static double angleOf(Vec2 v) {
    return std::atan2(v.y, v.x);
}

static Vec2 normalFromAngle(double angle) {
    return {std::cos(angle), std::sin(angle)};
}

static bool intersectSupportLines(Vec2 n0, double h0, Vec2 n1, double h1, Vec2& out) {
    const double det = cross(n0, n1);
    if (std::abs(det) < 1e-9) return false;
    out = {
        (h0 * n1.y - n0.y * h1) / det,
        (n0.x * h1 - h0 * n1.x) / det
    };
    return true;
}

static Poly8 decodeSupportLines(const double* x, bool* ok = nullptr) {
    Poly8 p;
    bool valid = true;
    std::array<Vec2, K> normals{};
    std::array<double, K> h{};
    for (int i = 0; i < K; ++i) {
        normals[i] = normalFromAngle(x[2 * i + 0]);
        h[i] = x[2 * i + 1];
    }
    for (int i = 0; i < K; ++i) {
        const int j = (i + 1) % K;
        Vec2 v;
        if (!intersectSupportLines(normals[i], h[i], normals[j], h[j], v)) {
            valid = false;
            v = {0.0, 0.0};
        }
        p.v[i] = v;
    }
    if (ok) *ok = valid;
    return ensureCCW(p);
}

static std::optional<Poly8> supportPolygonFromAngles(
    const std::vector<Vec2>& points,
    const std::array<double, K>& angles,
    double clearance
) {
    if (points.empty()) return std::nullopt;
    std::array<Vec2, K> normals{};
    std::array<double, K> h{};
    for (int i = 0; i < K; ++i) {
        normals[i] = normalFromAngle(angles[i]);
        h[i] = -INF;
        for (Vec2 p : points) h[i] = std::max(h[i], dot(normals[i], p));
        h[i] += clearance;
    }

    Poly8 p;
    for (int i = 0; i < K; ++i) {
        Vec2 v;
        if (!intersectSupportLines(normals[i], h[i], normals[(i + 1) % K], h[(i + 1) % K], v)) {
            return std::nullopt;
        }
        p.v[i] = v;
    }
    p = ensureCCW(p);
    return p;
}

static std::array<double, K> supportAnglesFromPoly(const Poly8& poly) {
    std::array<double, K> angles{};
    Poly8 p = ensureCCW(poly);
    double prevAngle = 0.0;
    for (int i = 0; i < K; ++i) {
        Vec2 a = p.v[i];
        Vec2 b = p.v[(i + 1) % K];
        Vec2 edge = b - a;
        const double len = std::max(norm(edge), 1e-12);
        Vec2 n{edge.y / len, -edge.x / len};
        double angle = angleOf(n);
        if (i == 0) {
            prevAngle = angle;
        } else {
            while (angle <= prevAngle) angle += 2.0 * PI;
            prevAngle = angle;
        }
        angles[i] = angle;
    }
    return angles;
}

static bool normalizeAngleCycle(std::array<double, K>& angles, double minGap) {
    std::sort(angles.begin(), angles.end());
    for (int i = 1; i < K; ++i) {
        if (angles[i] - angles[i - 1] < minGap) return false;
    }
    if (angles[0] + 2.0 * PI - angles[K - 1] < minGap) return false;
    return true;
}

static std::vector<double> encodeSupportLines(const Poly8& poly) {
    Poly8 p = ensureCCW(poly);
    std::vector<double> x(2 * K);
    double prevAngle = 0.0;
    for (int i = 0; i < K; ++i) {
        Vec2 a = p.v[i];
        Vec2 b = p.v[(i + 1) % K];
        Vec2 edge = b - a;
        const double len = std::max(norm(edge), 1e-12);
        Vec2 n{edge.y / len, -edge.x / len}; // outward normal for CCW polygon
        double angle = angleOf(n);
        if (i == 0) {
            prevAngle = angle;
        } else {
            while (angle <= prevAngle) angle += 2.0 * PI;
            prevAngle = angle;
        }
        x[2 * i + 0] = angle;
        x[2 * i + 1] = dot(n, a);
    }
    return x;
}

static double pointSegmentDistance(Vec2 p, Vec2 a, Vec2 b) {
    const Vec2 ab = b - a;
    const double d2 = norm2(ab);
    if (d2 <= 1e-20) return norm(p - a);
    const double t = clamp(dot(p - a, ab) / d2, 0.0, 1.0);
    return norm(p - (a + ab * t));
}

static double segmentSegmentDistance(Vec2 a, Vec2 b, Vec2 c, Vec2 d) {
    // For 2D, if they intersect, distance is zero. Otherwise min endpoint-to-segment distance.
    auto orient = [](Vec2 p, Vec2 q, Vec2 r) { return cross(q - p, r - p); };
    auto onSeg = [](Vec2 p, Vec2 q, Vec2 r) {
        return std::min(p.x, r.x) - 1e-9 <= q.x && q.x <= std::max(p.x, r.x) + 1e-9 &&
               std::min(p.y, r.y) - 1e-9 <= q.y && q.y <= std::max(p.y, r.y) + 1e-9 &&
               std::abs(cross(q - p, r - p)) <= 1e-8;
    };
    const double o1 = orient(a, b, c);
    const double o2 = orient(a, b, d);
    const double o3 = orient(c, d, a);
    const double o4 = orient(c, d, b);
    if (((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) &&
        ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))) {
        return 0.0;
    }
    if (onSeg(a, c, b) || onSeg(a, d, b) || onSeg(c, a, d) || onSeg(c, b, d)) return 0.0;
    return std::min({pointSegmentDistance(a, c, d), pointSegmentDistance(b, c, d),
                     pointSegmentDistance(c, a, b), pointSegmentDistance(d, a, b)});
}

static bool pointInPolyInclusive(Vec2 p, const Poly8& poly) {
    // Boundary is considered inside. Uses even-odd crossing rule.
    bool inside = false;
    for (int i = 0, j = K - 1; i < K; j = i++) {
        Vec2 a = poly.v[j];
        Vec2 b = poly.v[i];
        if (pointSegmentDistance(p, a, b) <= 1e-7) return true;
        const bool yi = (b.y > p.y);
        const bool yj = (a.y > p.y);
        if (yi != yj) {
            const double xCross = (a.x - b.x) * (p.y - b.y) / (a.y - b.y + 1e-300) + b.x;
            if (p.x < xCross) inside = !inside;
        }
    }
    return inside;
}

static double pointPolySignedDistance(Vec2 q, const Poly8& p) {
    double d = INF;
    for (int i = 0; i < K; ++i) {
        d = std::min(d, pointSegmentDistance(q, p.v[i], p.v[(i + 1) % K]));
    }
    return pointInPolyInclusive(q, p) ? -d : d;
}

static bool polygonIsSimple(const Poly8& p) {
    if (area(p) < 1e-6) return false;
    for (int i = 0; i < K; ++i) {
        Vec2 a = p.v[i];
        Vec2 b = p.v[(i + 1) % K];
        if (norm(b - a) < 1e-5) return false;
    }
    for (int i = 0; i < K; ++i) {
        Vec2 a = p.v[i];
        Vec2 b = p.v[(i + 1) % K];
        for (int j = i + 1; j < K; ++j) {
            if (std::abs(i - j) <= 1) continue;
            if (i == 0 && j == K - 1) continue;
            Vec2 c = p.v[j];
            Vec2 d = p.v[(j + 1) % K];
            if (segmentSegmentDistance(a, b, c, d) <= 1e-7) return false;
        }
    }
    return true;
}

static bool pointInPolyInclusive(Vec2 p, const PolyN& poly) {
    bool inside = false;
    const int n = static_cast<int>(poly.v.size());
    for (int i = 0, j = n - 1; i < n; j = i++) {
        Vec2 a = poly.v[j];
        Vec2 b = poly.v[i];
        if (pointSegmentDistance(p, a, b) <= 1e-7) return true;
        const bool yi = (b.y > p.y);
        const bool yj = (a.y > p.y);
        if (yi != yj) {
            const double xCross = (a.x - b.x) * (p.y - b.y) / (a.y - b.y + 1e-300) + b.x;
            if (p.x < xCross) inside = !inside;
        }
    }
    return inside;
}

static double pointPolySignedDistance(Vec2 q, const PolyN& p) {
    double d = INF;
    const int n = static_cast<int>(p.v.size());
    for (int i = 0; i < n; ++i) {
        d = std::min(d, pointSegmentDistance(q, p.v[i], p.v[(i + 1) % n]));
    }
    return pointInPolyInclusive(q, p) ? -d : d;
}

static bool polygonIsSimple(const PolyN& p) {
    const int n = static_cast<int>(p.v.size());
    if (n < 3 || area(p) < 1e-6) return false;
    for (int i = 0; i < n; ++i) {
        Vec2 a = p.v[i];
        Vec2 b = p.v[(i + 1) % n];
        if (norm(b - a) < 1e-5) return false;
    }
    for (int i = 0; i < n; ++i) {
        Vec2 a = p.v[i];
        Vec2 b = p.v[(i + 1) % n];
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(i - j) <= 1) continue;
            if (i == 0 && j == n - 1) continue;
            Vec2 c = p.v[j];
            Vec2 d = p.v[(j + 1) % n];
            if (segmentSegmentDistance(a, b, c, d) <= 1e-7) return false;
        }
    }
    return true;
}

struct SdfGrid {
    int w = 0;
    int h = 0;
    std::vector<float> sdf; // positive outside, negative inside

    float at(int x, int y) const {
        x = std::max(0, std::min(w - 1, x));
        y = std::max(0, std::min(h - 1, y));
        return sdf[y * w + x];
    }

    double sample(Vec2 p) const {
        // Pixel center coordinates. Clamp to image extent.
        const double x = clamp(p.x, 0.0, static_cast<double>(w - 1));
        const double y = clamp(p.y, 0.0, static_cast<double>(h - 1));
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = std::min(w - 1, x0 + 1);
        const int y1 = std::min(h - 1, y0 + 1);
        const double dx = x - x0;
        const double dy = y - y0;
        const double v00 = at(x0, y0);
        const double v10 = at(x1, y0);
        const double v01 = at(x0, y1);
        const double v11 = at(x1, y1);
        return (1.0 - dx) * (1.0 - dy) * v00 + dx * (1.0 - dy) * v10 +
               (1.0 - dx) * dy * v01 + dx * dy * v11;
    }
};

struct ImageData {
    int w = 0;
    int h = 0;
    cv::Mat rgba;
    std::vector<uint8_t> mask;
    std::vector<Vec2> objectPixels;
    std::vector<Vec2> boundaryPixels;
    SdfGrid sdf;
};

static bool maskAt(const std::vector<uint8_t>& mask, int w, int h, int x, int y) {
    if (x < 0 || y < 0 || x >= w || y >= h) return false;
    return mask[y * w + x] != 0;
}

static SdfGrid buildJumpFloodSdf(int w, int h, const std::vector<uint8_t>& mask) {
    struct Seed { int x = -1; int y = -1; };
    const int N = w * h;
    std::vector<Seed> nearest(N);

    auto idx = [w](int x, int y) { return y * w + x; };
    auto isBoundary = [&](int x, int y) {
        const bool v = maskAt(mask, w, h, x, y);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                if (maskAt(mask, w, h, x + dx, y + dy) != v) return true;
            }
        }
        return false;
    };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (isBoundary(x, y)) nearest[idx(x, y)] = {x, y};
        }
    }

    int step = 1;
    while (step < std::max(w, h)) step <<= 1;
    step >>= 1;

    auto dist2ToSeed = [](int x, int y, Seed s) {
        if (s.x < 0) return std::numeric_limits<int64_t>::max();
        const int64_t dx = static_cast<int64_t>(x) - s.x;
        const int64_t dy = static_cast<int64_t>(y) - s.y;
        return dx * dx + dy * dy;
    };

    std::vector<Seed> next = nearest;
    for (; step >= 1; step >>= 1) {
        next = nearest;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                Seed best = nearest[idx(x, y)];
                int64_t bestD = dist2ToSeed(x, y, best);
                for (int oy = -step; oy <= step; oy += step) {
                    for (int ox = -step; ox <= step; ox += step) {
                        const int xx = x + ox;
                        const int yy = y + oy;
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                        Seed cand = nearest[idx(xx, yy)];
                        const int64_t d2 = dist2ToSeed(x, y, cand);
                        if (d2 < bestD) { bestD = d2; best = cand; }
                    }
                }
                next[idx(x, y)] = best;
            }
        }
        nearest.swap(next);
    }

    // Final local refinement to reduce JFA artifacts.
    for (int pass = 0; pass < 2; ++pass) {
        next = nearest;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                Seed best = nearest[idx(x, y)];
                int64_t bestD = dist2ToSeed(x, y, best);
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        const int xx = x + ox;
                        const int yy = y + oy;
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                        Seed cand = nearest[idx(xx, yy)];
                        const int64_t d2 = dist2ToSeed(x, y, cand);
                        if (d2 < bestD) { bestD = d2; best = cand; }
                    }
                }
                next[idx(x, y)] = best;
            }
        }
        nearest.swap(next);
    }

    SdfGrid grid;
    grid.w = w;
    grid.h = h;
    grid.sdf.resize(N, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            Seed s = nearest[idx(x, y)];
            double d = 0.0;
            if (s.x >= 0) {
                const double dx = static_cast<double>(x - s.x);
                const double dy = static_cast<double>(y - s.y);
                d = std::sqrt(dx * dx + dy * dy);
            }
            if (maskAt(mask, w, h, x, y)) d = -d;
            grid.sdf[idx(x, y)] = static_cast<float>(d);
        }
    }
    return grid;
}

static ImageData loadImage(const std::string& path, int alphaThreshold) {
    cv::Mat src = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (src.empty()) throw std::runtime_error("Could not read image: " + path);

    cv::Mat rgba;
    if (src.channels() == 4) {
        cv::cvtColor(src, rgba, cv::COLOR_BGRA2RGBA);
    } else if (src.channels() == 3) {
        cv::cvtColor(src, rgba, cv::COLOR_BGR2RGBA);
    } else if (src.channels() == 1) {
        cv::cvtColor(src, rgba, cv::COLOR_GRAY2RGBA);
    } else {
        throw std::runtime_error("Unsupported image channel count");
    }

    ImageData img;
    img.w = rgba.cols;
    img.h = rgba.rows;
    img.rgba = rgba;
    img.mask.assign(img.w * img.h, 0);

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            const cv::Vec4b px = rgba.at<cv::Vec4b>(y, x);
            const uint8_t alpha = (src.channels() == 4) ? px[3] : static_cast<uint8_t>(255);
            const bool inside = alpha > alphaThreshold;
            img.mask[y * img.w + x] = inside ? 1 : 0;
            if (inside) img.objectPixels.push_back({x + 0.5, y + 0.5});
        }
    }

    if (img.objectPixels.empty()) throw std::runtime_error("Alpha mask is empty");

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            if (!maskAt(img.mask, img.w, img.h, x, y)) continue;
            bool boundary = false;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (!maskAt(img.mask, img.w, img.h, x + dx, y + dy)) boundary = true;
                }
            }
            if (boundary) img.boundaryPixels.push_back({x + 0.5, y + 0.5});
        }
    }

    img.sdf = buildJumpFloodSdf(img.w, img.h, img.mask);
    return img;
}

static std::vector<Vec2> deterministicSample(const std::vector<Vec2>& pts, int maxCount, int seed) {
    if (static_cast<int>(pts.size()) <= maxCount) return pts;
    std::vector<int> ids(pts.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<Vec2> out;
    out.reserve(maxCount);
    for (int i = 0; i < maxCount; ++i) out.push_back(pts[ids[i]]);
    return out;
}

static Poly8 supportOctagonStart(
    const std::vector<Vec2>& points,
    double angle,
    double margin,
    const std::array<double, K>& supportOffsets = {}
) {
    std::array<Vec2, K> n{};
    std::array<double, K> h{};
    for (int i = 0; i < K; ++i) {
        const double a = angle + static_cast<double>(i) * PI / 4.0;
        n[i] = {std::cos(a), std::sin(a)};
        h[i] = -INF;
        for (Vec2 p : points) h[i] = std::max(h[i], dot(n[i], p));
        h[i] += margin + supportOffsets[i];
    }

    Poly8 p;
    for (int i = 0; i < K; ++i) {
        const int j = (i + 1) % K;
        const double det = cross(n[i], n[j]);
        if (std::abs(det) < 1e-12) throw std::runtime_error("Degenerate octagon normals");
        // Solve [n_i^T; n_j^T] x = [h_i; h_j].
        p.v[i] = {
            (h[i] * n[j].y - n[i].y * h[j]) / det,
            (n[i].x * h[j] - h[i] * n[j].x) / det
        };
    }
    return ensureCCW(p);
}

static Poly8 scaleFromCenter(Poly8 p, Vec2 c, double factor) {
    for (auto& v : p.v) v = c + (v - c) * factor;
    return p;
}

struct Bounds {
    double x0 = 0.0;
    double y0 = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
};

static Bounds objectBounds(const ImageData& img) {
    Bounds b{INF, INF, -INF, -INF};
    for (Vec2 p : img.objectPixels) {
        b.x0 = std::min(b.x0, p.x);
        b.y0 = std::min(b.y0, p.y);
        b.x1 = std::max(b.x1, p.x);
        b.y1 = std::max(b.y1, p.y);
    }
    return b;
}

struct Validation {
    bool simple = true;
    bool contains = true;
    bool edgeClear = true;
    double minSdfOnEdges = INF;
    int outsideObjectCount = 0;
    double maxOutsideDistance = 0.0;
    double area = 0.0;
};

static Validation validateDense(const Poly8& p, const ImageData& img, double clearance, double edgeStep) {
    Validation v;
    v.area = area(p);
    v.simple = polygonIsSimple(p);
    if (!v.simple) {
        v.contains = false;
        v.edgeClear = false;
        return v;
    }

    int outsideCount = 0;
    double maxOutsideDistance = 0.0;
#ifdef _OPENMP
    #pragma omp parallel
    {
        int localOutsideCount = 0;
        double localMaxOutsideDistance = 0.0;
        #pragma omp for schedule(static)
        for (int qi = 0; qi < static_cast<int>(img.objectPixels.size()); ++qi) {
            Vec2 q = img.objectPixels[qi];
            if (!pointInPolyInclusive(q, p)) {
                ++localOutsideCount;
                localMaxOutsideDistance = std::max(localMaxOutsideDistance, pointPolySignedDistance(q, p));
            }
        }
        #pragma omp critical
        {
            outsideCount += localOutsideCount;
            maxOutsideDistance = std::max(maxOutsideDistance, localMaxOutsideDistance);
        }
    }
#else
    for (int qi = 0; qi < static_cast<int>(img.objectPixels.size()); ++qi) {
        Vec2 q = img.objectPixels[qi];
        if (!pointInPolyInclusive(q, p)) {
            ++outsideCount;
            maxOutsideDistance = std::max(maxOutsideDistance, pointPolySignedDistance(q, p));
        }
    }
#endif
    v.outsideObjectCount = outsideCount;
    v.contains = outsideCount == 0;
    v.maxOutsideDistance = maxOutsideDistance;

    double minSdf = INF;
    int edgeClearViolations = 0;
#ifdef _OPENMP
    #pragma omp parallel
    {
        double localMinSdf = INF;
        int localEdgeClearViolations = 0;
        #pragma omp for schedule(dynamic)
        for (int e = 0; e < K; ++e) {
            Vec2 a = p.v[e];
            Vec2 b = p.v[(e + 1) % K];
            const double len = norm(b - a);
            const int samples = std::max(2, static_cast<int>(std::ceil(len / std::max(edgeStep, 1e-3))));
            for (int i = 0; i <= samples; ++i) {
                const double t = static_cast<double>(i) / static_cast<double>(samples);
                Vec2 s = lerp(a, b, t);
                const double d = img.sdf.sample(s);
                localMinSdf = std::min(localMinSdf, d);
                const double tolerance = (clearance <= 0.0) ? 0.75 : 1e-6;
                if (d < clearance - tolerance) ++localEdgeClearViolations;
            }
        }
        #pragma omp critical
        {
            minSdf = std::min(minSdf, localMinSdf);
            edgeClearViolations += localEdgeClearViolations;
        }
    }
#else
    for (int e = 0; e < K; ++e) {
        Vec2 a = p.v[e];
        Vec2 b = p.v[(e + 1) % K];
        const double len = norm(b - a);
        const int samples = std::max(2, static_cast<int>(std::ceil(len / std::max(edgeStep, 1e-3))));
        for (int i = 0; i <= samples; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(samples);
            Vec2 s = lerp(a, b, t);
            const double d = img.sdf.sample(s);
            minSdf = std::min(minSdf, d);
            const double tolerance = (clearance <= 0.0) ? 0.75 : 1e-6;
            if (d < clearance - tolerance) ++edgeClearViolations;
        }
    }
#endif
    v.minSdfOnEdges = minSdf;
    v.edgeClear = edgeClearViolations == 0;
    return v;
}

static bool isValidDense(const Poly8& p, const ImageData& img, double clearance, double edgeStep) {
    const Validation val = validateDense(p, img, clearance, edgeStep);
    return val.simple && val.contains && val.edgeClear;
}

static Validation validateDense(const PolyN& p, const ImageData& img, double clearance, double edgeStep) {
    Validation v;
    v.area = area(p);
    v.simple = polygonIsSimple(p);
    if (!v.simple) {
        v.contains = false;
        v.edgeClear = false;
        return v;
    }

    int outsideCount = 0;
    double maxOutsideDistance = 0.0;
    for (Vec2 q : img.objectPixels) {
        if (!pointInPolyInclusive(q, p)) {
            ++outsideCount;
            maxOutsideDistance = std::max(maxOutsideDistance, pointPolySignedDistance(q, p));
        }
    }
    v.outsideObjectCount = outsideCount;
    v.contains = outsideCount == 0;
    v.maxOutsideDistance = maxOutsideDistance;

    double minSdf = INF;
    int edgeClearViolations = 0;
    const int n = static_cast<int>(p.v.size());
    for (int e = 0; e < n; ++e) {
        Vec2 a = p.v[e];
        Vec2 b = p.v[(e + 1) % n];
        const double len = norm(b - a);
        const int samples = std::max(2, static_cast<int>(std::ceil(len / std::max(edgeStep, 1e-3))));
        for (int i = 0; i <= samples; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(samples);
            Vec2 s = lerp(a, b, t);
            const double d = img.sdf.sample(s);
            minSdf = std::min(minSdf, d);
            const double tolerance = (clearance <= 0.0) ? 0.75 : 1e-6;
            if (d < clearance - tolerance) ++edgeClearViolations;
        }
    }
    v.minSdfOnEdges = minSdf;
    v.edgeClear = edgeClearViolations == 0;
    return v;
}

static bool isValidDense(const PolyN& p, const ImageData& img, double clearance, double edgeStep) {
    const Validation val = validateDense(p, img, clearance, edgeStep);
    return val.simple && val.contains && val.edgeClear;
}

static std::vector<Vec2> externalContourCandidates(const ImageData& img, int maxCandidates) {
    cv::Mat maskMat(img.h, img.w, CV_8UC1);
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            maskMat.at<uint8_t>(y, x) = img.mask[y * img.w + x] ? 255 : 0;
        }
    }
    const int dilatePixels = std::max(2, static_cast<int>(std::round(std::min(img.w, img.h) * 0.015)));
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(dilatePixels * 2 + 1, dilatePixels * 2 + 1)
    );
    cv::dilate(maskMat, maskMat, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(maskMat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return {};

    int bestContour = 0;
    double bestContourArea = 0.0;
    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        const double a = std::abs(cv::contourArea(contours[i]));
        if (a > bestContourArea) {
            bestContourArea = a;
            bestContour = i;
        }
    }

    const Vec2 c = centroidOfPoints(img.objectPixels);
    std::vector<Vec2> raw;
    raw.reserve(contours[bestContour].size());
    for (const cv::Point& p : contours[bestContour]) {
        Vec2 q{static_cast<double>(p.x) + 0.5, static_cast<double>(p.y) + 0.5};
        Vec2 outward = q - c;
        const double d = norm(outward);
        if (d > 1e-9) q += outward / d * 0.35;
        raw.push_back(q);
    }
    raw = ensureCCW(PolyN{raw}).v;

    const int target = std::max(8, std::min(maxCandidates, static_cast<int>(raw.size())));
    if (static_cast<int>(raw.size()) <= target) return raw;

    std::vector<double> prefix(raw.size() + 1, 0.0);
    for (int i = 0; i < static_cast<int>(raw.size()); ++i) {
        prefix[i + 1] = prefix[i] + norm(raw[(i + 1) % raw.size()] - raw[i]);
    }
    const double perimeter = prefix.back();
    std::vector<Vec2> out;
    out.reserve(target);
    int cursor = 0;
    for (int i = 0; i < target; ++i) {
        const double want = perimeter * static_cast<double>(i) / static_cast<double>(target);
        while (cursor + 1 < static_cast<int>(prefix.size()) && prefix[cursor + 1] < want) ++cursor;
        const int a = cursor % static_cast<int>(raw.size());
        const int b = (a + 1) % static_cast<int>(raw.size());
        const double span = std::max(1e-9, prefix[cursor + 1] - prefix[cursor]);
        out.push_back(lerp(raw[a], raw[b], clamp((want - prefix[cursor]) / span, 0.0, 1.0)));
    }
    return ensureCCW(PolyN{out}).v;
}

static bool visibleChord(Vec2 a, Vec2 b, const ImageData& img, double clearance, double step) {
    const double len = norm(b - a);
    const int samples = std::max(2, static_cast<int>(std::ceil(len / std::max(step, 0.25))));
    const double tolerance = (clearance <= 0.0) ? 1.25 : 1e-6;
    for (int i = 0; i <= samples; ++i) {
        const Vec2 p = lerp(a, b, static_cast<double>(i) / static_cast<double>(samples));
        if (img.sdf.sample(p) < clearance - tolerance) return false;
    }
    return true;
}

struct VisibilityGraph {
    int n = 0;
    std::vector<uint8_t> visible;

    bool at(int i, int j) const {
        return visible[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] != 0;
    }
};

static VisibilityGraph buildVisibilityGraph(const std::vector<Vec2>& candidates, const ImageData& img, const Options& opt) {
    VisibilityGraph graph;
    graph.n = static_cast<int>(candidates.size());
    graph.visible.assign(static_cast<size_t>(graph.n) * static_cast<size_t>(graph.n), 0);

#ifdef MKL8GON_USE_CUDA
    if (opt.useCuda && graph.n >= 16) {
        std::vector<double> xy(static_cast<size_t>(graph.n) * 2u);
        for (int i = 0; i < graph.n; ++i) {
            xy[static_cast<size_t>(i) * 2u + 0u] = candidates[i].x;
            xy[static_cast<size_t>(i) * 2u + 1u] = candidates[i].y;
        }
        const int rc = mkl8gon_build_visibility_cuda(
            xy.data(),
            graph.n,
            img.sdf.sdf.data(),
            img.sdf.w,
            img.sdf.h,
            opt.finalClearance,
            opt.denseEdgeStep,
            graph.visible.data()
        );
        if (rc == 0) return graph;
        if (opt.verbose) std::cerr << "CUDA visibility failed rc=" << rc << "; falling back to CPU\n";
    }
#endif

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i = 0; i < graph.n; ++i) {
        for (int j = 0; j < graph.n; ++j) {
            if (i == j) continue;
            graph.visible[static_cast<size_t>(i) * static_cast<size_t>(graph.n) + static_cast<size_t>(j)] =
                visibleChord(candidates[i], candidates[j], img, opt.finalClearance, opt.denseEdgeStep) ? 1 : 0;
        }
    }
    return graph;
}

static std::optional<PolyN> growPointPolygonUntilValid(PolyN p, const ImageData& img, const Options& opt) {
    p = ensureCCW(p);
    Validation initial = validateDense(p, img, opt.finalClearance, opt.denseEdgeStep);
    if (initial.simple && initial.contains && initial.edgeClear) return p;

    const Vec2 c = centroidOfPoints(img.objectPixels);
    for (int i = 1; i <= 240; ++i) {
        const double scale = 1.0 + 0.0025 * static_cast<double>(i);
        PolyN q = p;
        for (Vec2& v : q.v) v = c + (v - c) * scale;
        q = ensureCCW(q);
        Validation validation = validateDense(q, img, opt.finalClearance, opt.denseEdgeStep);
        if (validation.simple && validation.contains && validation.edgeClear) return q;
    }
    return std::nullopt;
}

static std::optional<PolyN> solveVisibilityPointPolygonK(
    const std::vector<Vec2>& candidates,
    const VisibilityGraph& graph,
    const ImageData& img,
    const Options& opt,
    int vertices
) {
    const int n = static_cast<int>(candidates.size());
    if (n < vertices || vertices < 3) return std::nullopt;

    double bestArea = INF;
    PolyN bestPoly;
    const int anchorStep = std::max(1, n / std::min(n, 48));
    bool cudaDpAttempted = false;

#ifdef MKL8GON_USE_CUDA
    if (opt.useCuda && n <= 160 && vertices <= 16) {
        cudaDpAttempted = true;
        std::vector<double> xy(static_cast<size_t>(n) * 2u);
        for (int i = 0; i < n; ++i) {
            xy[static_cast<size_t>(i) * 2u + 0u] = candidates[i].x;
            xy[static_cast<size_t>(i) * 2u + 1u] = candidates[i].y;
        }
        const int maxCudaCandidates = std::min(64, std::max(1, (n + anchorStep - 1) / anchorStep));
        std::vector<double> outXy(static_cast<size_t>(maxCudaCandidates) * static_cast<size_t>(vertices) * 2u);
        std::vector<double> rawTwiceArea(static_cast<size_t>(maxCudaCandidates), INF);
        int cudaCandidateCount = 0;
        const int rc = mkl8gon_solve_visibility_dp_cuda(
            xy.data(),
            graph.visible.data(),
            n,
            vertices,
            anchorStep,
            maxCudaCandidates,
            outXy.data(),
            rawTwiceArea.data(),
            &cudaCandidateCount
        );
        if (rc == 0) {
            for (int ci = 0; ci < cudaCandidateCount; ++ci) {
                PolyN candidate;
                candidate.v.reserve(vertices);
                for (int i = 0; i < vertices; ++i) {
                    const size_t base = (static_cast<size_t>(ci) * static_cast<size_t>(vertices) + static_cast<size_t>(i)) * 2u;
                    candidate.v.push_back({outXy[base + 0u], outXy[base + 1u]});
                }
                candidate = ensureCCW(candidate);
                std::optional<PolyN> repaired = growPointPolygonUntilValid(candidate, img, opt);
                if (repaired) candidate = *repaired;
                Validation validation = validateDense(candidate, img, opt.finalClearance, opt.denseEdgeStep);
                if (validation.simple && validation.contains && validation.edgeClear && validation.area < bestArea) {
                    bestArea = validation.area;
                    bestPoly = candidate;
                }
            }
        } else if (opt.verbose) {
            std::cerr << "CUDA DP failed rc=" << rc << "; ";
        }
    }
#endif

    if (!cudaDpAttempted || bestArea >= INF / 2.0) {
        if (cudaDpAttempted && opt.verbose) std::cerr << "using fallback candidate path\n";
    std::vector<int> anchors;
    for (int anchor = 0; anchor < n; anchor += anchorStep) anchors.push_back(anchor);

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
    double localBestArea = INF;
    PolyN localBestPoly;
    std::vector<double> dp;
    std::vector<int> prev;
    std::vector<Vec2> verts;

    #ifdef _OPENMP
    #pragma omp for schedule(dynamic)
    #endif
    for (int anchorIndex = 0; anchorIndex < static_cast<int>(anchors.size()); ++anchorIndex) {
        const int anchor = anchors[anchorIndex];
        const int limit = anchor + n;
        const int stride = limit + 1;
        dp.assign(static_cast<size_t>(vertices + 1) * static_cast<size_t>(stride), INF);
        prev.assign(static_cast<size_t>(vertices + 1) * static_cast<size_t>(stride), -1);
        auto idx = [stride](int used, int i) { return static_cast<size_t>(used) * static_cast<size_t>(stride) + static_cast<size_t>(i); };
        dp[idx(1, anchor)] = 0.0;

        for (int used = 1; used < vertices; ++used) {
            for (int i = anchor; i < limit; ++i) {
                const double baseCost = dp[idx(used, i)];
                if (baseCost >= INF / 2.0) continue;
                const int ii = i % n;
                const int minJ = i + 1;
                const int maxJ = limit - (vertices - used - 1);
                for (int j = minJ; j <= maxJ; ++j) {
                    const int jj = j % n;
                    if (!graph.at(ii, jj)) continue;
                    const double cost = baseCost + cross(candidates[ii], candidates[jj]);
                    const size_t nextIdx = idx(used + 1, j);
                    if (cost < dp[nextIdx]) {
                        dp[nextIdx] = cost;
                        prev[nextIdx] = i;
                    }
                }
            }
        }

        for (int end = anchor + vertices - 1; end < limit; ++end) {
            if (dp[idx(vertices, end)] >= INF / 2.0) continue;
            const int ee = end % n;
            if (!graph.at(ee, anchor)) continue;
            const double signedTwiceArea = dp[idx(vertices, end)] + cross(candidates[ee], candidates[anchor]);
            if (signedTwiceArea <= 0.0) continue;

            verts.assign(vertices, {});
            int cur = end;
            bool reconstructed = true;
            for (int used = vertices; used >= 1; --used) {
                verts[used - 1] = candidates[cur % n];
                cur = prev[idx(used, cur)];
                if (used > 1 && cur < 0) reconstructed = false;
            }
            if (!reconstructed) continue;

            PolyN candidate = ensureCCW(PolyN{verts});
            std::optional<PolyN> repaired = growPointPolygonUntilValid(candidate, img, opt);
            if (repaired) candidate = *repaired;
            Validation validation = validateDense(candidate, img, opt.finalClearance, opt.denseEdgeStep);
            if (validation.simple && validation.contains && validation.edgeClear && validation.area < localBestArea) {
                localBestArea = validation.area;
                localBestPoly = candidate;
            }
        }
    }

    #ifdef _OPENMP
    #pragma omp critical
    #endif
    {
        if (localBestArea < bestArea) {
            bestArea = localBestArea;
            bestPoly = localBestPoly;
        }
    }
    }
    }

    if (vertices <= n) {
        std::vector<Vec2> fallback;
        fallback.reserve(vertices);
        for (int i = 0; i < vertices; ++i) {
            fallback.push_back(candidates[(i * n) / vertices]);
        }
        PolyN candidate = ensureCCW(PolyN{fallback});
        std::optional<PolyN> repaired = growPointPolygonUntilValid(candidate, img, opt);
        if (repaired) candidate = *repaired;
        Validation validation = validateDense(candidate, img, opt.finalClearance, opt.denseEdgeStep);
        if (validation.simple && validation.contains && validation.edgeClear && validation.area < bestArea) {
            bestArea = validation.area;
            bestPoly = candidate;
        }
    }
    if (bestArea >= INF / 2.0) return std::nullopt;
    return bestPoly;
}

static PolyN solveVisibilityPointPolygon(
    const ImageData& img,
    const Options& opt,
    const std::function<void(int, const PolyN&, double, double)>& onFeasible = {}
) {
    const std::vector<Vec2> candidates = externalContourCandidates(img, opt.pointCandidates);
    if (candidates.size() < 3) throw std::runtime_error("Could not extract enough contour candidates");
    const VisibilityGraph graph = buildVisibilityGraph(candidates, img, opt);

    std::vector<std::pair<double, PolyN>> feasible;
    double bestArea = INF;
    int fruitlessPointCounts = 0;
    const int minPoints = std::max(3, opt.minPoints);
    const int maxPoints = std::max(minPoints, opt.maxPoints);
    for (int vertices = minPoints; vertices <= maxPoints; ++vertices) {
        std::optional<PolyN> maybe = solveVisibilityPointPolygonK(candidates, graph, img, opt, vertices);
        if (!maybe) {
            if (opt.verbose) std::cerr << "visibility points " << vertices << " no feasible polygon\n";
            continue;
        }

        PolyN candidate = *maybe;
        const double candidateArea = area(candidate);
        if (opt.verbose) {
            std::cerr << "visibility points " << vertices << " area=" << candidateArea;
            if (!feasible.empty()) {
                const double gainPct = 100.0 * (bestArea - candidateArea) / std::max(bestArea, 1e-9);
                std::cerr << " gain=" << gainPct << "%";
            }
            std::cerr << "\n";
        }

        feasible.push_back({candidateArea, candidate});
        if (candidateArea < bestArea) {
            const double gainPct = bestArea < INF / 2.0
                ? 100.0 * (bestArea - candidateArea) / std::max(bestArea, 1e-9)
                : INF;
            bestArea = candidateArea;
            fruitlessPointCounts = (gainPct < opt.pointGainPct) ? fruitlessPointCounts + 1 : 0;
        } else {
            ++fruitlessPointCounts;
        }
        if (onFeasible) onFeasible(vertices, candidate, candidateArea, bestArea);
        if (fruitlessPointCounts >= std::max(1, opt.pointPatience)) {
            if (opt.verbose) {
                std::cerr << "visibility point search stopped after "
                          << fruitlessPointCounts << " low-gain point counts\n";
            }
            break;
        }
    }

    if (feasible.empty()) throw std::runtime_error("Visibility point solver found no feasible polygon");
    const double allowableArea = bestArea * (1.0 + std::max(0.0, opt.pointGainPct) / 100.0);
    for (const auto& candidate : feasible) {
        if (candidate.first <= allowableArea) return ensureCCW(candidate.second);
    }
    return ensureCCW(std::min_element(
        feasible.begin(),
        feasible.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; }
    )->second);
}

struct ViolationSet {
    std::vector<Vec2> outsidePoints;
    std::vector<EdgeSample> badEdgeSamples;
};

static ViolationSet collectViolations(const Poly8& p, const ImageData& img, double clearance, double edgeStep, int maxObj, int maxEdge) {
    ViolationSet out;
    if (!polygonIsSimple(p)) return out;

    for (Vec2 q : img.objectPixels) {
        if (!pointInPolyInclusive(q, p)) {
            out.outsidePoints.push_back(q);
            if (static_cast<int>(out.outsidePoints.size()) >= maxObj) break;
        }
    }

    for (int e = 0; e < K; ++e) {
        Vec2 a = p.v[e];
        Vec2 b = p.v[(e + 1) % K];
        const double len = norm(b - a);
        const int samples = std::max(2, static_cast<int>(std::ceil(len / std::max(edgeStep, 1e-3))));
        for (int i = 0; i <= samples; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(samples);
            Vec2 s = lerp(a, b, t);
            const double d = img.sdf.sample(s);
            const double tolerance = (clearance <= 0.0) ? 0.75 : 0.0;
            if (d < clearance - tolerance) {
                out.badEdgeSamples.push_back({e, t});
                if (static_cast<int>(out.badEdgeSamples.size()) >= maxEdge) return out;
            }
        }
    }
    return out;
}

static Poly8 repairByBinarySearch(const Poly8& validBase, const Poly8& proposal, const ImageData& img, double clearance, double edgeStep) {
    if (isValidDense(proposal, img, clearance, edgeStep)) return proposal;
    double lo = 0.0;
    double hi = 1.0;
    Poly8 best = validBase;
    for (int it = 0; it < 36; ++it) {
        const double mid = 0.5 * (lo + hi);
        Poly8 cand = lerpPoly(validBase, proposal, mid);
        if (isValidDense(cand, img, clearance, edgeStep)) {
            lo = mid;
            best = cand;
        } else {
            hi = mid;
        }
    }
    return best;
}

struct ResidualContext {
    const ImageData* img = nullptr;
    std::vector<Vec2> activeObject;
    std::vector<EdgeSample> activeEdge;
    int edgeSamplesPerEdge = 40;
    double clearance = 0.0;
    double wArea = 1.0;
    double wContain = 1.0e6;
    double wEdge = 1.0e6;
    double wSelf = 1.0e5;
    double wMinEdge = 1.0e4;
    double minEdgeLength = 1.0;
};

static int residualCount(const ResidualContext& ctx) {
    const int nonAdjacentPairs = 20; // C(8,2) - 8 adjacent - 1 wrap? For this loop below it is 20.
    return 1 + static_cast<int>(ctx.activeObject.size()) +
           K * ctx.edgeSamplesPerEdge + static_cast<int>(ctx.activeEdge.size()) +
           nonAdjacentPairs + K;
}

static void evalResiduals(const ResidualContext& ctx, const double* x, double* f, int mExpected) {
    const Poly8 p = decodeRawXY(x);
    int k = 0;
    const double a = std::max(1e-12, area(p));
    f[k++] = std::sqrt(ctx.wArea) * std::sqrt(a);

    for (Vec2 q : ctx.activeObject) {
        const double sd = pointPolySignedDistance(q, p); // positive outside polygon
        f[k++] = std::sqrt(ctx.wContain) * softplus(sd, 3.0);
    }

    for (int e = 0; e < K; ++e) {
        for (int j = 0; j < ctx.edgeSamplesPerEdge; ++j) {
            const double t = static_cast<double>(j + 1) / static_cast<double>(ctx.edgeSamplesPerEdge + 1);
            Vec2 s = lerp(p.v[e], p.v[(e + 1) % K], t);
            const double d = ctx.img->sdf.sample(s);
            f[k++] = std::sqrt(ctx.wEdge) * softplus(ctx.clearance - d, 3.0);
        }
    }

    for (const EdgeSample& es : ctx.activeEdge) {
        const int e = ((es.edge % K) + K) % K;
        Vec2 s = lerp(p.v[e], p.v[(e + 1) % K], clamp(es.t, 0.0, 1.0));
        const double d = ctx.img->sdf.sample(s);
        f[k++] = std::sqrt(ctx.wEdge) * softplus(ctx.clearance - d, 3.0);
    }

    for (int i = 0; i < K; ++i) {
        Vec2 a0 = p.v[i];
        Vec2 a1 = p.v[(i + 1) % K];
        for (int j = i + 1; j < K; ++j) {
            if (std::abs(i - j) <= 1) continue;
            if (i == 0 && j == K - 1) continue;
            Vec2 b0 = p.v[j];
            Vec2 b1 = p.v[(j + 1) % K];
            const double d = segmentSegmentDistance(a0, a1, b0, b1);
            f[k++] = std::sqrt(ctx.wSelf) * softplus(0.25 - d, 5.0);
        }
    }

    for (int i = 0; i < K; ++i) {
        const double len = norm(p.v[(i + 1) % K] - p.v[i]);
        f[k++] = std::sqrt(ctx.wMinEdge) * softplus(ctx.minEdgeLength - len, 5.0);
    }

    if (k != mExpected) {
        // Keep MKL from seeing uninitialized memory if the bookkeeping is wrong.
        for (; k < mExpected; ++k) f[k] = 1.0e9;
    }
}

// Signature used by oneMKL djacobix examples is non-const pointers.
static void residualThunk(MKL_INT* m, MKL_INT* /*n*/, double* x, double* f, void* userData) {
    auto* ctx = static_cast<ResidualContext*>(userData);
    evalResiduals(*ctx, x, f, static_cast<int>(*m));
}

struct MklSolveResult {
    Poly8 poly;
    int request = 0;
    int stopCriterion = 0;
    int iterations = 0;
    double initialResidual = 0.0;
    double finalResidual = 0.0;
    bool ok = false;
};

static MklSolveResult runMklLocal(const Poly8& start, ResidualContext& ctx, const Options& opt) {
    constexpr int nRaw = 2 * K;
    MKL_INT n = nRaw;
    MKL_INT m = residualCount(ctx);
    std::vector<double> x = encodeRawXY(start);

    const double pad = 2.0 * std::max(ctx.img->w, ctx.img->h);
    std::vector<double> lw(nRaw), up(nRaw);
    for (int i = 0; i < K; ++i) {
        lw[2 * i + 0] = -pad;
        up[2 * i + 0] = static_cast<double>(ctx.img->w) + pad;
        lw[2 * i + 1] = -pad;
        up[2 * i + 1] = static_cast<double>(ctx.img->h) + pad;
    }

    std::vector<double> fvec(m, 0.0);
    std::vector<double> fjac(m * n, 0.0);
    double eps[6] = {1e-7, 1e-7, 1e-7, 1e-7, 1e-12, 1e-10};
    MKL_INT iter1 = opt.mklMaxIterations;
    MKL_INT iter2 = opt.mklMaxTrialStepIterations;
    double rs = 20.0;

    _TRNSPBC_HANDLE_t handle = nullptr;
    MKL_INT res = dtrnlspbc_init(&handle, &n, &m, x.data(), lw.data(), up.data(), eps, &iter1, &iter2, &rs);
    if (res != TR_SUCCESS) {
        throw std::runtime_error("dtrnlspbc_init failed");
    }

    evalResiduals(ctx, x.data(), fvec.data(), static_cast<int>(m));
    double jacEps = 1e-6;
    res = djacobix(residualThunk, &n, &m, fjac.data(), x.data(), &jacEps, &ctx);
    if (res != TR_SUCCESS) {
        dtrnlspbc_delete(&handle);
        throw std::runtime_error("djacobix initial call failed");
    }

    MKL_INT info[6] = {0, 0, 0, 0, 0, 0};
    res = dtrnlspbc_check(&handle, &n, &m, fjac.data(), fvec.data(), lw.data(), up.data(), eps, info);
    if (res != TR_SUCCESS) {
        dtrnlspbc_delete(&handle);
        throw std::runtime_error("dtrnlspbc_check failed");
    }
    for (int i = 0; i < 6; ++i) {
        if (info[i] != 0) {
            dtrnlspbc_delete(&handle);
            std::ostringstream os;
            os << "dtrnlspbc_check rejected input; info[" << i << "]=" << info[i];
            throw std::runtime_error(os.str());
        }
    }

    MKL_INT request = 0;
    int guard = 0;
    while (true) {
        res = dtrnlspbc_solve(&handle, fvec.data(), fjac.data(), &request);
        if (res != TR_SUCCESS) {
            dtrnlspbc_delete(&handle);
            throw std::runtime_error("dtrnlspbc_solve failed");
        }
        if (request == 1) {
            evalResiduals(ctx, x.data(), fvec.data(), static_cast<int>(m));
        } else if (request == 2) {
            res = djacobix(residualThunk, &n, &m, fjac.data(), x.data(), &jacEps, &ctx);
            if (res != TR_SUCCESS) {
                dtrnlspbc_delete(&handle);
                throw std::runtime_error("djacobix failed during solve");
            }
        } else if (request == 0) {
            // Successful iteration; keep going.
        } else {
            break;
        }
        if (++guard > 10000) break;
    }

    MKL_INT iter = 0;
    MKL_INT stCr = 0;
    double r1 = 0.0;
    double r2 = 0.0;
    dtrnlspbc_get(&handle, &iter, &stCr, &r1, &r2);
    dtrnlspbc_delete(&handle);

    MklSolveResult out;
    out.poly = ensureCCW(decodeRawXY(x.data()));
    out.request = static_cast<int>(request);
    out.stopCriterion = static_cast<int>(stCr);
    out.iterations = static_cast<int>(iter);
    out.initialResidual = r1;
    out.finalResidual = r2;
    out.ok = true;
    return out;
}

struct SupportResidualContext {
    const ImageData* img = nullptr;
    std::vector<Vec2> boundary;
    double clearance = 0.0;
    double wArea = 25.0;
    double wContain = 2.0e4;
    double wSlack = 35.0;
    double wCanvas = 8.0e4;
    double wOrder = 2.0e4;
    double wMinEdge = 1.0e4;
    double minAngleGap = 0.10;
    double minEdgeLength = 1.0;
};

static int supportResidualCount(const SupportResidualContext& ctx) {
    return 1 + static_cast<int>(ctx.boundary.size()) + K + 4 * K + K + K;
}

static void evalSupportResiduals(const SupportResidualContext& ctx, const double* x, double* f, int mExpected) {
    bool ok = true;
    Poly8 p = decodeSupportLines(x, &ok);
    int k = 0;

    const double imageArea = std::max(1.0, static_cast<double>(ctx.img->w * ctx.img->h));
    const double polyArea = ok ? std::max(1e-12, area(p)) : imageArea * 100.0;
    f[k++] = std::sqrt(ctx.wArea) * std::sqrt(polyArea / imageArea);

    std::array<Vec2, K> normals{};
    std::array<double, K> h{};
    for (int i = 0; i < K; ++i) {
        normals[i] = normalFromAngle(x[2 * i + 0]);
        h[i] = x[2 * i + 1];
    }

    for (Vec2 q : ctx.boundary) {
        double maxViolation = -INF;
        for (int i = 0; i < K; ++i) {
            maxViolation = std::max(maxViolation, dot(normals[i], q) - h[i]);
        }
        f[k++] = std::sqrt(ctx.wContain) * softplus(maxViolation, 5.0);
    }

    for (int i = 0; i < K; ++i) {
        double support = -INF;
        for (Vec2 q : ctx.boundary) {
            support = std::max(support, dot(normals[i], q));
        }
        const double slack = h[i] - support - ctx.clearance;
        f[k++] = std::sqrt(ctx.wSlack) * softplus(slack, 1.5);
    }

    const double maxX = static_cast<double>(ctx.img->w);
    const double maxY = static_cast<double>(ctx.img->h);
    for (int i = 0; i < K; ++i) {
        Vec2 v = p.v[i];
        f[k++] = std::sqrt(ctx.wCanvas) * softplus(-v.x, 5.0);
        f[k++] = std::sqrt(ctx.wCanvas) * softplus(v.x - maxX, 5.0);
        f[k++] = std::sqrt(ctx.wCanvas) * softplus(-v.y, 5.0);
        f[k++] = std::sqrt(ctx.wCanvas) * softplus(v.y - maxY, 5.0);
    }

    for (int i = 0; i < K; ++i) {
        const int j = (i + 1) % K;
        const double ai = x[2 * i + 0];
        const double aj = (j == 0) ? x[0] + 2.0 * PI : x[2 * j + 0];
        f[k++] = std::sqrt(ctx.wOrder) * softplus(ctx.minAngleGap - (aj - ai), 5.0);
    }

    for (int i = 0; i < K; ++i) {
        const double len = norm(p.v[(i + 1) % K] - p.v[i]);
        f[k++] = std::sqrt(ctx.wMinEdge) * softplus(ctx.minEdgeLength - len, 5.0);
    }

    if (!ok || !polygonIsSimple(p)) {
        for (int i = 0; i < k; ++i) f[i] += 1.0e4;
    }
    for (; k < mExpected; ++k) f[k] = 1.0e9;
}

static void supportResidualThunk(MKL_INT* m, MKL_INT* /*n*/, double* x, double* f, void* userData) {
    auto* ctx = static_cast<SupportResidualContext*>(userData);
    evalSupportResiduals(*ctx, x, f, static_cast<int>(*m));
}

static MklSolveResult runMklSupportLocal(const Poly8& start, SupportResidualContext& ctx, const Options& opt) {
    constexpr int nRaw = 2 * K;
    MKL_INT n = nRaw;
    MKL_INT m = supportResidualCount(ctx);
    std::vector<double> x = encodeSupportLines(start);
    const std::vector<double> x0 = x;

    std::vector<double> lw(nRaw), up(nRaw);
    const double maxDim = static_cast<double>(std::max(ctx.img->w, ctx.img->h));
    const double angleWindow = clamp(std::max(opt.angleVariance, 0.18), 0.08, 0.36);
    for (int i = 0; i < K; ++i) {
        lw[2 * i + 0] = x0[2 * i + 0] - angleWindow;
        up[2 * i + 0] = x0[2 * i + 0] + angleWindow;
        lw[2 * i + 1] = -2.0 * maxDim;
        up[2 * i + 1] = 3.0 * maxDim;
    }

    std::vector<double> fvec(m, 0.0);
    std::vector<double> fjac(m * n, 0.0);
    double eps[6] = {1e-6, 1e-6, 1e-6, 1e-6, 1e-12, 1e-10};
    MKL_INT iter1 = opt.mklMaxIterations;
    MKL_INT iter2 = opt.mklMaxTrialStepIterations;
    double rs = 5.0;

    _TRNSPBC_HANDLE_t handle = nullptr;
    MKL_INT res = dtrnlspbc_init(&handle, &n, &m, x.data(), lw.data(), up.data(), eps, &iter1, &iter2, &rs);
    if (res != TR_SUCCESS) throw std::runtime_error("dtrnlspbc_init failed");

    evalSupportResiduals(ctx, x.data(), fvec.data(), static_cast<int>(m));
    double jacEps = 1e-5;
    res = djacobix(supportResidualThunk, &n, &m, fjac.data(), x.data(), &jacEps, &ctx);
    if (res != TR_SUCCESS) {
        dtrnlspbc_delete(&handle);
        throw std::runtime_error("support djacobix initial call failed");
    }

    MKL_INT info[6] = {0, 0, 0, 0, 0, 0};
    res = dtrnlspbc_check(&handle, &n, &m, fjac.data(), fvec.data(), lw.data(), up.data(), eps, info);
    if (res != TR_SUCCESS) {
        dtrnlspbc_delete(&handle);
        throw std::runtime_error("support dtrnlspbc_check failed");
    }

    MKL_INT request = 0;
    int guard = 0;
    while (true) {
        res = dtrnlspbc_solve(&handle, fvec.data(), fjac.data(), &request);
        if (res != TR_SUCCESS) {
            dtrnlspbc_delete(&handle);
            throw std::runtime_error("support dtrnlspbc_solve failed");
        }
        if (request == 1) {
            evalSupportResiduals(ctx, x.data(), fvec.data(), static_cast<int>(m));
        } else if (request == 2) {
            res = djacobix(supportResidualThunk, &n, &m, fjac.data(), x.data(), &jacEps, &ctx);
            if (res != TR_SUCCESS) {
                dtrnlspbc_delete(&handle);
                throw std::runtime_error("support djacobix failed during solve");
            }
        } else if (request == 0) {
            // Continue iteration.
        } else {
            break;
        }
        if (++guard > 10000) break;
    }

    MKL_INT iter = 0;
    MKL_INT stCr = 0;
    double r1 = 0.0;
    double r2 = 0.0;
    dtrnlspbc_get(&handle, &iter, &stCr, &r1, &r2);
    dtrnlspbc_delete(&handle);

    MklSolveResult out;
    out.poly = decodeSupportLines(x.data());
    out.request = static_cast<int>(request);
    out.stopCriterion = static_cast<int>(stCr);
    out.iterations = static_cast<int>(iter);
    out.initialResidual = r1;
    out.finalResidual = r2;
    out.ok = true;
    return out;
}

static void addUniqueObjectSamples(std::vector<Vec2>& dst, const std::vector<Vec2>& src, int maxTotal) {
    for (Vec2 p : src) {
        if (static_cast<int>(dst.size()) >= maxTotal) break;
        bool exists = false;
        for (Vec2 q : dst) {
            if (norm2(p - q) < 1e-6) { exists = true; break; }
        }
        if (!exists) dst.push_back(p);
    }
}

static void addUniqueEdgeSamples(std::vector<EdgeSample>& dst, const std::vector<EdgeSample>& src, int maxTotal) {
    for (EdgeSample s : src) {
        if (static_cast<int>(dst.size()) >= maxTotal) break;
        bool exists = false;
        for (EdgeSample t : dst) {
            if (s.edge == t.edge && std::abs(s.t - t.t) < 1e-3) { exists = true; break; }
        }
        if (!exists) dst.push_back(s);
    }
}

static Poly8 deterministicPolish(Poly8 p, const ImageData& img, const Options& opt, double clearance) {
    if (!isValidDense(p, img, clearance, opt.denseEdgeStep)) return p;
    double bestArea = area(p);
    Vec2 c = centroidOfPoints(img.objectPixels);

    std::vector<double> steps = {8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.10};
    for (double step : steps) {
        bool improved = true;
        int sweep = 0;
        while (improved && sweep++ < 10) {
            improved = false;
            for (int i = 0; i < K; ++i) {
                const int im = (i + K - 1) % K;
                const int ip = (i + 1) % K;
                // Gradient of signed polygon area wrt vertex i.
                Vec2 grad{0.5 * (p.v[ip].y - p.v[im].y), 0.5 * (p.v[im].x - p.v[ip].x)};
                if (signedArea(p) < 0.0) grad = grad * -1.0;
                Vec2 inward = c - p.v[i];
                if (norm(inward) > 1e-12) inward = inward / norm(inward);
                std::array<Vec2, 3> dirs = {grad * -1.0, inward, (grad * -1.0 + inward)};
                for (Vec2 d : dirs) {
                    if (norm(d) < 1e-12) continue;
                    d = d / norm(d);
                    Poly8 proposal = p;
                    proposal.v[i] = proposal.v[i] + d * step;
                    proposal = ensureCCW(proposal);
                    Poly8 cand = proposal;
                    if (!isValidDense(cand, img, clearance, opt.denseEdgeStep)) {
                        cand = repairByBinarySearch(p, proposal, img, clearance, opt.denseEdgeStep);
                    }
                    const double ar = area(cand);
                    if (isValidDense(cand, img, clearance, opt.denseEdgeStep) && ar + 1e-6 < bestArea) {
                        p = cand;
                        bestArea = ar;
                        improved = true;
                    }
                }
            }
        }
    }
    return p;
}

static Poly8 conservativeChamferStart(const ImageData& img, double margin, double chamferFrac) {
    Bounds b = objectBounds(img);
    const double spanX = std::max(1.0, b.x1 - b.x0);
    const double spanY = std::max(1.0, b.y1 - b.y0);
    const double c = std::min(spanX, spanY) * chamferFrac;
    Poly8 p;
    p.v = {{
        {b.x0 + c, b.y0 - margin},
        {b.x1 - c, b.y0 - margin},
        {b.x1 + margin, b.y0 + c},
        {b.x1 + margin, b.y1 - c},
        {b.x1 - c, b.y1 + margin},
        {b.x0 + c, b.y1 + margin},
        {b.x0 - margin, b.y1 - c},
        {b.x0 - margin, b.y0 + c},
    }};
    return ensureCCW(p);
}

static std::vector<Poly8> generateStarts(const ImageData& img, const Options& opt) {
    std::vector<Poly8> starts;
    const int count = std::max(1, opt.starts);
    const double dim = static_cast<double>(std::max(img.w, img.h));
    const double baseMargin = std::max(2.0, dim * 0.043);
    starts.reserve(count);
    for (int i = 0; i < count; ++i) {
        const double t = count == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(count - 1);
        const double margin = baseMargin * (0.75 + 0.85 * t);
        const double chamfer = 0.06 + 0.12 * std::fmod(static_cast<double>(i) * 0.61803398875, 1.0);
        starts.push_back(conservativeChamferStart(img, margin, chamfer));
    }
    return starts;
}

static Poly8 makeValidStart(Poly8 p, const ImageData& img, const Options& opt, double clearance) {
    Vec2 c = centroidOfPoints(img.objectPixels);
    p = ensureCCW(p);
    double factor = 1.0;
    for (int it = 0; it < 40; ++it) {
        Poly8 q = scaleFromCenter(p, c, factor);
        q = ensureCCW(q);
        if (isValidDense(q, img, clearance, opt.denseEdgeStep)) return q;
        factor *= 1.08;
    }
    throw std::runtime_error("Could not expand start polygon to validity");
}

static Vec2 polygonCentroidAverage(const Poly8& p) {
    Vec2 c{0.0, 0.0};
    for (Vec2 v : p.v) c += v;
    return c / static_cast<double>(K);
}

static bool directAccept(Poly8& p, const Poly8& proposal, const ImageData& img, const Options& opt, double clearance) {
    Poly8 candidate = ensureCCW(proposal);
    if (area(candidate) + 1e-6 >= area(p)) return false;
    if (!isValidDense(candidate, img, clearance, opt.denseEdgeStep)) return false;
    p = candidate;
    return true;
}

static Poly8 greedyInwardShrink(Poly8 p, const ImageData& img, const Options& opt, double clearance, int sweeps = 80) {
    p = ensureCCW(p);
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        bool improved = false;
        Vec2 c = centroidOfPoints(img.objectPixels);
        for (int i = 0; i < K; ++i) {
            const Vec2 cur = p.v[i];
            Vec2 inward = c - cur;
            if (norm(inward) <= 1e-9) continue;
            static const std::array<double, 18> fracs = {
                0.24, 0.20, 0.16, 0.13, 0.10, 0.08, 0.06, 0.045, 0.033,
                0.024, 0.017, 0.012, 0.008, 0.0055, 0.0035, 0.0022, 0.0014, 0.0010
            };
            for (double frac : fracs) {
                Poly8 proposal = p;
                proposal.v[i] = cur + inward * frac;
                if (directAccept(p, proposal, img, opt, clearance)) {
                    improved = true;
                    break;
                }
            }
        }
        if (!improved) break;
    }
    return p;
}

static Poly8 stochasticSlidePolish(Poly8 p, const ImageData& img, const Options& opt, double clearance, int seed) {
    p = ensureCCW(p);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> vertexDist(0, K - 1);
    std::normal_distribution<double> normal(0.0, 1.0);
    Vec2 c = centroidOfPoints(img.objectPixels);
    const std::array<double, 8> steps = {10.0, 6.0, 3.5, 2.0, 1.0, 0.5, 0.25, 0.1};

    for (double step : steps) {
        int noImprove = 0;
        const int maxAttempts = opt.fastClearances ? 700 : 1800;
        const int maxNoImprove = opt.fastClearances ? 220 : 450;
        for (int attempt = 0; attempt < maxAttempts; ++attempt) {
            const int i = vertexDist(rng);
            const Vec2 cur = p.v[i];
            Vec2 inward = c - cur;
            if (norm(inward) <= 1e-9) continue;
            inward = inward / norm(inward);
            Vec2 tangent{-inward.y, inward.x};
            Vec2 proposed = cur + inward * std::abs(normal(rng) * step) + tangent * (normal(rng) * step * 1.4);
            Poly8 proposal = p;
            proposal.v[i] = proposed;
            if (directAccept(p, proposal, img, opt, clearance)) {
                noImprove = 0;
            } else {
                ++noImprove;
            }
            if (noImprove > maxNoImprove) break;
        }
    }
    return p;
}

static Poly8 growUntilClear(Poly8 p, const ImageData& img, const Options& opt, double targetClearance, double maxScale = 1.035) {
    Vec2 c = polygonCentroidAverage(p);
    for (int i = 0; i <= 200; ++i) {
        const double t = static_cast<double>(i) / 200.0;
        const double scale = 1.0 + (maxScale - 1.0) * t;
        Poly8 q = scaleFromCenter(p, c, scale);
        q = ensureCCW(q);
        Validation rep = validateDense(q, img, targetClearance, opt.denseEdgeStep);
        if (rep.simple && rep.contains && rep.edgeClear) return q;
    }
    return p;
}

static Poly8 sdfSafeFreeShrink(Poly8 start, const ImageData& img, const Options& opt, double clearance, int seed) {
    Poly8 p = ensureCCW(start);
    const double minGainPct = std::max(0.0, opt.minGainPct);
    double previousArea = area(p);
    int fruitlessRounds = 0;
    for (int round = 0; ; ++round) {
        Poly8 before = p;
        p = greedyInwardShrink(p, img, opt, clearance, opt.fastClearances ? 30 : 80);
        p = stochasticSlidePolish(p, img, opt, clearance, seed + round * 1009);
        p = growUntilClear(p, img, opt, clearance <= 0.0 ? -0.05 : clearance);
        if (!isValidDense(p, img, clearance, opt.denseEdgeStep)) {
            p = before;
            break;
        }
        const double newArea = area(p);
        const double gainPct = 100.0 * (previousArea - newArea) / std::max(previousArea, 1e-9);
        previousArea = newArea;
        if (gainPct < minGainPct) {
            ++fruitlessRounds;
        } else {
            fruitlessRounds = 0;
        }
        if (fruitlessRounds >= FRUITLESS_ROUND_PATIENCE) break;
    }
    return ensureCCW(p);
}

static PolyN toPolyN(const Poly8& p) {
    PolyN out;
    out.v.assign(p.v.begin(), p.v.end());
    return ensureCCW(out);
}

static Vec2 polygonCentroidAverage(const PolyN& p) {
    Vec2 c{0.0, 0.0};
    for (Vec2 v : p.v) c += v;
    return c / static_cast<double>(std::max<size_t>(1, p.v.size()));
}

static bool directAccept(PolyN& p, const PolyN& proposal, const ImageData& img, const Options& opt, double clearance) {
    PolyN candidate = ensureCCW(proposal);
    if (area(candidate) + 1e-6 >= area(p)) return false;
    if (!isValidDense(candidate, img, clearance, opt.denseEdgeStep)) return false;
    p = candidate;
    return true;
}

static PolyN greedyInwardShrink(PolyN p, const ImageData& img, const Options& opt, double clearance, int sweeps = 80) {
    p = ensureCCW(p);
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        bool improved = false;
        Vec2 c = centroidOfPoints(img.objectPixels);
        const int n = static_cast<int>(p.v.size());
        for (int i = 0; i < n; ++i) {
            const Vec2 cur = p.v[i];
            Vec2 inward = c - cur;
            if (norm(inward) <= 1e-9) continue;
            static const std::array<double, 18> fracs = {
                0.24, 0.20, 0.16, 0.13, 0.10, 0.08, 0.06, 0.045, 0.033,
                0.024, 0.017, 0.012, 0.008, 0.0055, 0.0035, 0.0022, 0.0014, 0.0010
            };
            for (double frac : fracs) {
                PolyN proposal = p;
                proposal.v[i] = cur + inward * frac;
                if (directAccept(p, proposal, img, opt, clearance)) {
                    improved = true;
                    break;
                }
            }
        }
        if (!improved) break;
    }
    return p;
}

static PolyN stochasticSlidePolish(
    PolyN p,
    const ImageData& img,
    const Options& opt,
    double clearance,
    int seed,
    int attemptOverride = -1,
    int noImproveOverride = -1,
    int stepLimit = 8
) {
    p = ensureCCW(p);
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Vec2 c = centroidOfPoints(img.objectPixels);
    const std::array<double, 8> steps = {10.0, 6.0, 3.5, 2.0, 1.0, 0.5, 0.25, 0.1};

    const int activeSteps = std::max(1, std::min(stepLimit, static_cast<int>(steps.size())));
    for (int stepIndex = 0; stepIndex < activeSteps; ++stepIndex) {
        const double step = steps[stepIndex];
        int noImprove = 0;
        const int maxAttempts = attemptOverride > 0 ? attemptOverride : (opt.fastClearances ? 700 : 1800);
        const int maxNoImprove = noImproveOverride > 0 ? noImproveOverride : (opt.fastClearances ? 220 : 450);
        for (int attempt = 0; attempt < maxAttempts; ++attempt) {
            std::uniform_int_distribution<int> vertexDist(0, static_cast<int>(p.v.size()) - 1);
            const int i = vertexDist(rng);
            const Vec2 cur = p.v[i];
            Vec2 inward = c - cur;
            if (norm(inward) <= 1e-9) continue;
            inward = inward / norm(inward);
            Vec2 tangent{-inward.y, inward.x};
            PolyN proposal = p;
            proposal.v[i] = cur + inward * std::abs(normal(rng) * step) + tangent * (normal(rng) * step * 1.4);
            if (directAccept(p, proposal, img, opt, clearance)) {
                noImprove = 0;
            } else {
                ++noImprove;
            }
            if (noImprove > maxNoImprove) break;
        }
    }
    return p;
}

static PolyN growUntilClear(PolyN p, const ImageData& img, const Options& opt, double targetClearance, double maxScale = 1.035);

static PolyN quickAddedPointShrink(PolyN start, const ImageData& img, const Options& opt, double clearance, int seed) {
    PolyN p = ensureCCW(start);
    for (int round = 0; round < 3; ++round) {
        PolyN before = p;
        p = greedyInwardShrink(p, img, opt, clearance, opt.fastClearances ? 8 : 14);
        p = stochasticSlidePolish(
            p,
            img,
            opt,
            clearance,
            seed + round * 1009,
            opt.fastClearances ? 120 : 220,
            opt.fastClearances ? 45 : 80,
            opt.fastClearances ? 5 : 6
        );
        p = growUntilClear(p, img, opt, clearance <= 0.0 ? -0.05 : clearance);
        if (!isValidDense(p, img, clearance, opt.denseEdgeStep)) return before;
        const double gainPct = 100.0 * (area(before) - area(p)) / std::max(area(before), 1e-9);
        if (gainPct < std::max(0.0, opt.minGainPct)) break;
    }
    return ensureCCW(p);
}

static PolyN growUntilClear(PolyN p, const ImageData& img, const Options& opt, double targetClearance, double maxScale) {
    Vec2 c = polygonCentroidAverage(p);
    for (int i = 0; i <= 200; ++i) {
        const double t = static_cast<double>(i) / 200.0;
        const double scale = 1.0 + (maxScale - 1.0) * t;
        PolyN q = p;
        for (auto& v : q.v) v = c + (v - c) * scale;
        q = ensureCCW(q);
        Validation rep = validateDense(q, img, targetClearance, opt.denseEdgeStep);
        if (rep.simple && rep.contains && rep.edgeClear) return q;
    }
    return p;
}

static PolyN sdfSafeFreeShrink(PolyN start, const ImageData& img, const Options& opt, double clearance, int seed) {
    PolyN p = ensureCCW(start);
    const double minGainPct = std::max(0.0, opt.minGainPct);
    double previousArea = area(p);
    int fruitlessRounds = 0;
    for (int round = 0; ; ++round) {
        PolyN before = p;
        p = greedyInwardShrink(p, img, opt, clearance, opt.fastClearances ? 30 : 80);
        p = stochasticSlidePolish(p, img, opt, clearance, seed + round * 1009);
        p = growUntilClear(p, img, opt, clearance <= 0.0 ? -0.05 : clearance);
        if (!isValidDense(p, img, clearance, opt.denseEdgeStep)) {
            p = before;
            break;
        }
        const double newArea = area(p);
        const double gainPct = 100.0 * (previousArea - newArea) / std::max(previousArea, 1e-9);
        previousArea = newArea;
        if (gainPct < minGainPct) ++fruitlessRounds;
        else fruitlessRounds = 0;
        if (fruitlessRounds >= FRUITLESS_ROUND_PATIENCE) break;
    }
    return ensureCCW(p);
}

static PolyN geneticOptimize(PolyN start, const ImageData& img, const Options& opt, double clearance, int seed) {
    PolyN best = ensureCCW(start);
    if (!isValidDense(best, img, clearance, opt.denseEdgeStep)) {
        best = growUntilClear(best, img, opt, clearance <= 0.0 ? -0.05 : clearance);
    }
    if (!isValidDense(best, img, clearance, opt.denseEdgeStep)) return ensureCCW(start);

    const int n = static_cast<int>(best.v.size());
    const int populationSize = std::max(8, opt.gaPopulation);
    const int generations = std::max(1, opt.gaGenerations);
    const double mutationBase = std::max(0.05, opt.gaMutation);
    const Vec2 objectCenter = centroidOfPoints(img.objectPixels);
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> vertexDist(0, n - 1);

    std::vector<PolyN> population;
    population.reserve(populationSize);
    population.push_back(best);

    auto validArea = [&](const PolyN& p) -> double {
        Validation rep = validateDense(p, img, clearance, opt.denseEdgeStep);
        if (!rep.simple || !rep.contains || !rep.edgeClear) return INF;
        return rep.area;
    };

    auto mutateFrom = [&](const PolyN& parent, int generation, int salt) {
        std::mt19937 localRng(seed + generation * 104729 + salt * 8191);
        std::normal_distribution<double> localNormal(0.0, 1.0);
        PolyN child = parent;
        const double cool = 1.0 - 0.82 * (static_cast<double>(generation) / std::max(1, generations - 1));
        const double sigma = mutationBase * std::max(0.18, cool);
        const int edits = 1 + (unit(localRng) < 0.35 ? 1 : 0) + (unit(localRng) < 0.12 ? 1 : 0);
        for (int edit = 0; edit < edits; ++edit) {
            const int i = vertexDist(localRng);
            const Vec2 cur = child.v[i];
            Vec2 inward = objectCenter - cur;
            if (norm(inward) <= 1e-9) inward = {localNormal(localRng), localNormal(localRng)};
            inward = inward / std::max(norm(inward), 1e-9);
            Vec2 tangent{-inward.y, inward.x};
            const double inwardScale = std::abs(localNormal(localRng)) * sigma;
            const double tangentScale = localNormal(localRng) * sigma * 1.8;
            child.v[i] = cur + inward * inwardScale + tangent * tangentScale;
        }
        child = ensureCCW(child);
        child = growUntilClear(child, img, opt, clearance <= 0.0 ? -0.05 : clearance, 1.02 + sigma * 0.004);
        child = quickAddedPointShrink(child, img, opt, clearance, seed + generation * 31337 + salt);
        return ensureCCW(child);
    };

    for (int i = 1; i < populationSize; ++i) {
        PolyN child = mutateFrom(best, 0, i);
        if (validArea(child) < INF) population.push_back(child);
    }

    double bestArea = validArea(best);
    int fruitless = 0;
    for (int generation = 0; generation < generations && fruitless < FRUITLESS_ROUND_PATIENCE; ++generation) {
        std::vector<std::pair<double, PolyN>> scored;
        scored.reserve(population.size());
        for (const PolyN& p : population) {
            const double a = validArea(p);
            if (a < INF) scored.push_back({a, p});
        }
        if (scored.empty()) break;
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        const double generationBestArea = scored.front().first;
        if (generationBestArea + 1e-6 < bestArea) {
            bestArea = generationBestArea;
            best = scored.front().second;
            fruitless = 0;
            if (opt.verbose) {
                std::cerr << "genetic generation " << generation
                          << " vertices=" << n
                          << " area=" << bestArea << "\n";
            }
        } else {
            ++fruitless;
        }

        std::vector<PolyN> next;
        const int eliteCount = std::min<int>(std::max(2, populationSize / 8), static_cast<int>(scored.size()));
        next.reserve(populationSize);
        for (int i = 0; i < eliteCount; ++i) next.push_back(scored[i].second);

        std::uniform_int_distribution<int> eliteDist(0, eliteCount - 1);
        while (static_cast<int>(next.size()) < populationSize) {
            const PolyN& a = scored[eliteDist(rng)].second;
            const PolyN& b = scored[eliteDist(rng)].second;
            PolyN child = a;
            for (int i = 0; i < n; ++i) {
                if (unit(rng) < 0.45) child.v[i] = b.v[i];
                if (unit(rng) < 0.20) child.v[i] = lerp(a.v[i], b.v[i], unit(rng));
            }
            child = mutateFrom(child, generation + 1, static_cast<int>(next.size()));
            if (validArea(child) < INF) next.push_back(child);
        }
        population.swap(next);
    }
    return ensureCCW(best);
}

static PolyN insertVertexOnEdge(const PolyN& p, int edge) {
    PolyN out;
    const int n = static_cast<int>(p.v.size());
    out.v.reserve(n + 1);
    for (int i = 0; i < n; ++i) {
        out.v.push_back(p.v[i]);
        if (i == edge) {
            out.v.push_back(lerp(p.v[i], p.v[(i + 1) % n], 0.5));
        }
    }
    return ensureCCW(out);
}

static PolyN addPointsUntilFruitless(PolyN start, const ImageData& img, const Options& opt, int seed) {
    PolyN best = sdfSafeFreeShrink(start, img, opt, opt.finalClearance, seed);
    if (opt.genetic) {
        best = geneticOptimize(best, img, opt, opt.finalClearance, seed + 17);
    }
    while (static_cast<int>(best.v.size()) < std::max(3, opt.maxPoints)) {
        const double beforeArea = area(best);
        PolyN bestCandidate = best;
        double bestArea = beforeArea;
        const int n = static_cast<int>(best.v.size());
        for (int e = 0; e < n; ++e) {
            PolyN trial = insertVertexOnEdge(best, e);
            trial = quickAddedPointShrink(trial, img, opt, opt.finalClearance, seed + n * 10007 + e * 101);
            Validation validation = validateDense(trial, img, opt.finalClearance, opt.denseEdgeStep);
            if (validation.simple && validation.contains && validation.edgeClear && validation.area < bestArea) {
                bestArea = validation.area;
                bestCandidate = trial;
            }
        }
        const double gainPct = 100.0 * (beforeArea - bestArea) / std::max(beforeArea, 1e-9);
        if (opt.verbose) {
            std::cerr << "auto-points candidate " << (n + 1)
                      << " area=" << bestArea
                      << " gain=" << gainPct << "%\n";
        }
        if (gainPct < opt.pointGainPct) break;
        best = sdfSafeFreeShrink(bestCandidate, img, opt, opt.finalClearance, seed + n * 31337);
        if (opt.genetic) {
            best = geneticOptimize(best, img, opt, opt.finalClearance, seed + n * 65537);
        }
    }
    return ensureCCW(best);
}

static double canvasPenalty(const Poly8& p, const ImageData& img) {
    const double maxX = static_cast<double>(img.w);
    const double maxY = static_cast<double>(img.h);
    double penalty = 0.0;
    for (Vec2 v : p.v) {
        const double lx = std::max(0.0, -v.x);
        const double rx = std::max(0.0, v.x - maxX);
        const double ty = std::max(0.0, -v.y);
        const double by = std::max(0.0, v.y - maxY);
        penalty += lx * lx + rx * rx + ty * ty + by * by;
    }
    return penalty;
}

static std::optional<Poly8> optimizeSupportAngles(
    const ImageData& img,
    const std::vector<Vec2>& supportPoints,
    const Options& opt,
    const Poly8& seed,
    double clearance
) {
    if (supportPoints.empty()) return std::nullopt;

    auto scorePoly = [&](const Poly8& p) {
        const double canvasScale = static_cast<double>(std::max(1, img.w * img.h));
        return area(p) + canvasPenalty(p, img) * canvasScale;
    };

    auto acceptCandidate = [&](const std::array<double, K>& angles, double& bestScore, Poly8& bestPoly, std::array<double, K>& bestAngles) {
        auto maybe = supportPolygonFromAngles(supportPoints, angles, clearance);
        if (!maybe) return false;
        Poly8 candidate = *maybe;
        if (canvasPenalty(candidate, img) > 1e-6) return false;
        Validation validation = validateDense(candidate, img, 0.0, opt.denseEdgeStep);
        if (!validation.simple || !validation.contains) return false;
        const double score = scorePoly(candidate);
        if (score + 1e-6 < bestScore) {
            bestScore = score;
            bestPoly = candidate;
            bestAngles = angles;
            return true;
        }
        return false;
    };

    Poly8 bestPoly = seed;
    std::array<double, K> bestAngles = supportAnglesFromPoly(seed);
    normalizeAngleCycle(bestAngles, 0.05);
    double bestScore = isValidDense(seed, img, clearance, opt.denseEdgeStep) ? scorePoly(seed) : INF;
    acceptCandidate(bestAngles, bestScore, bestPoly, bestAngles);

    const int rotationSamples = opt.fastClearances ? 192 : 384;
    for (int r = 0; r < rotationSamples; ++r) {
        std::array<double, K> angles{};
        const double base = (PI / 4.0) * static_cast<double>(r) / static_cast<double>(rotationSamples);
        for (int i = 0; i < K; ++i) angles[i] = base + static_cast<double>(i) * PI / 4.0;
        acceptCandidate(angles, bestScore, bestPoly, bestAngles);
    }

    double step = clamp(std::max(opt.angleVariance, 0.25), 0.10, 0.45);
    int stagnant = 0;
    while (step > 0.0008 && stagnant < 8) {
        bool improved = false;
        for (int i = 0; i < K; ++i) {
            for (double dir : {-1.0, 1.0}) {
                std::array<double, K> angles = bestAngles;
                angles[i] += dir * step;
                if (!normalizeAngleCycle(angles, 0.05)) continue;
                if (acceptCandidate(angles, bestScore, bestPoly, bestAngles)) improved = true;
            }
        }
        if (improved) {
            stagnant = 0;
        } else {
            ++stagnant;
            step *= 0.65;
        }
    }

    if (bestScore >= INF / 2.0) return std::nullopt;
    return bestPoly;
}

struct SolveResult {
    Poly8 poly;
    Validation validation;
    int startIndex = -1;
};

static cv::Mat makePaddedOverlayBase(const ImageData& img, const std::vector<Vec2>& vertices, cv::Point2d& offset) {
    (void)vertices;
    const int pad = std::max(16, static_cast<int>(std::ceil(std::max(img.w, img.h) * 0.25)));

    cv::Mat bgra;
    cv::cvtColor(img.rgba, bgra, cv::COLOR_RGBA2BGRA);
    cv::Mat imageBgr;
    cv::cvtColor(bgra, imageBgr, cv::COLOR_BGRA2BGR);

    cv::Mat out(img.h + pad * 2, img.w + pad * 2, CV_8UC3, cv::Scalar(245, 245, 245));
    imageBgr.copyTo(out(cv::Rect(pad, pad, img.w, img.h)));
    offset = cv::Point2d(static_cast<double>(pad), static_cast<double>(pad));
    return out;
}

static cv::Mat makeOverlay(const ImageData& img, const Poly8& p) {
    std::vector<Vec2> verts(p.v.begin(), p.v.end());
    cv::Point2d offset;
    cv::Mat out = makePaddedOverlayBase(img, verts, offset);

    // Boundary in green.
    for (Vec2 q : img.boundaryPixels) {
        int x = static_cast<int>(std::floor(q.x + offset.x));
        int y = static_cast<int>(std::floor(q.y + offset.y));
        if (0 <= x && x < out.cols && 0 <= y && y < out.rows) {
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
        }
    }

    std::vector<cv::Point> pts;
    pts.reserve(K);
    for (int i = 0; i < K; ++i) {
        pts.emplace_back(
            static_cast<int>(std::round(p.v[i].x + offset.x)),
            static_cast<int>(std::round(p.v[i].y + offset.y))
        );
    }
    const cv::Point* data = pts.data();
    int npts = static_cast<int>(pts.size());
    const int minDim = std::min(out.cols, out.rows);
    const int lineThickness = minDim >= 256 ? 1 : 1;
    const int pointRadius = minDim >= 256 ? 2 : 1;
    cv::polylines(out, &data, &npts, 1, true, cv::Scalar(0, 0, 255), lineThickness, cv::LINE_AA);

    for (const auto& pt : pts) cv::circle(out, pt, pointRadius, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
    return out;
}

static cv::Mat makeOverlay(const ImageData& img, const PolyN& p) {
    cv::Point2d offset;
    cv::Mat out = makePaddedOverlayBase(img, p.v, offset);

    for (Vec2 q : img.boundaryPixels) {
        int x = static_cast<int>(std::floor(q.x + offset.x));
        int y = static_cast<int>(std::floor(q.y + offset.y));
        if (0 <= x && x < out.cols && 0 <= y && y < out.rows) {
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
        }
    }

    std::vector<cv::Point> pts;
    pts.reserve(p.v.size());
    for (Vec2 v : p.v) {
        pts.emplace_back(
            static_cast<int>(std::round(v.x + offset.x)),
            static_cast<int>(std::round(v.y + offset.y))
        );
    }
    const cv::Point* data = pts.data();
    int npts = static_cast<int>(pts.size());
    cv::polylines(out, &data, &npts, 1, true, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    const int pointRadius = std::min(out.cols, out.rows) >= 256 ? 2 : 1;
    for (const auto& pt : pts) cv::circle(out, pt, pointRadius, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
    return out;
}

static cv::Mat makeBasePreview(const ImageData& img) {
    cv::Mat bgra;
    cv::cvtColor(img.rgba, bgra, cv::COLOR_RGBA2BGRA);
    cv::Mat out;
    cv::cvtColor(bgra, out, cv::COLOR_BGRA2BGR);
    return out;
}

struct ProgressSnapshot {
    cv::Mat image;
    std::string title = "mkl_sdf_8gon";
    std::vector<std::string> labels;
    double progress = 0.0;
};

class ProgressUi {
public:
    ProgressUi() = default;
    ~ProgressUi() { stop(); }

    bool start(const std::string& title) {
#ifdef _WIN32
        if (running_.exchange(true)) return true;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot_.title = title;
        }
        thread_ = std::thread([this]() { threadMain(); });
        return true;
#else
        (void)title;
        return false;
#endif
    }

    void stop() {
#ifdef _WIN32
        if (!running_.exchange(false)) return;
        HWND hwnd = hwnd_.load();
        if (hwnd) PostMessageA(hwnd, WM_CLOSE, 0, 0);
        if (thread_.joinable()) thread_.join();
#endif
    }

    void waitUntilClosed() {
#ifdef _WIN32
        if (thread_.joinable()) thread_.join();
#endif
    }

    void update(const cv::Mat& image, double progress, const std::string& title, std::vector<std::string> labels) {
#ifdef _WIN32
        if (!running_.load()) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot_.image = image.empty() ? cv::Mat() : image.clone();
            snapshot_.progress = clamp(progress, 0.0, 1.0);
            snapshot_.title = title;
            snapshot_.labels = std::move(labels);
        }
        HWND hwnd = hwnd_.load();
        if (hwnd) {
            InvalidateRect(hwnd, nullptr, FALSE);
        }
#else
        (void)image;
        (void)progress;
        (void)title;
        (void)labels;
#endif
    }

private:
#ifdef _WIN32
    static LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
        ProgressUi* self = reinterpret_cast<ProgressUi*>(GetWindowLongPtrA(hwnd, GWLP_USERDATA));
        if (msg == WM_NCCREATE) {
            auto* cs = reinterpret_cast<CREATESTRUCTA*>(lparam);
            self = static_cast<ProgressUi*>(cs->lpCreateParams);
            SetWindowLongPtrA(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        }
        if (!self) return DefWindowProcA(hwnd, msg, wparam, lparam);

        switch (msg) {
            case WM_TIMER:
                InvalidateRect(hwnd, nullptr, FALSE);
                return 0;
            case WM_PAINT:
                self->paint(hwnd);
                return 0;
            case WM_CLOSE:
                self->running_.store(false);
                DestroyWindow(hwnd);
                return 0;
            case WM_DESTROY:
                PostQuitMessage(0);
                return 0;
            default:
                return DefWindowProcA(hwnd, msg, wparam, lparam);
        }
    }

    void threadMain() {
        HINSTANCE instance = GetModuleHandleA(nullptr);
        const char* className = "MklSdf8gonProgressWindow";
        WNDCLASSEXA wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = ProgressUi::wndProc;
        wc.hInstance = instance;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        wc.lpszClassName = className;
        RegisterClassExA(&wc);

        std::string title;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            title = snapshot_.title;
        }
        HWND hwnd = CreateWindowExA(
            0,
            className,
            title.c_str(),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            1040,
            760,
            nullptr,
            nullptr,
            instance,
            this
        );
        if (!hwnd) {
            running_.store(false);
            return;
        }

        hwnd_.store(hwnd);
        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);
        SetTimer(hwnd, 1, 100, nullptr);

        MSG msg;
        while (GetMessageA(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
            if (!running_.load() && !IsWindow(hwnd)) break;
        }
        hwnd_.store(nullptr);
        running_.store(false);
    }

    void paint(HWND hwnd) {
        PAINTSTRUCT ps;
        HDC windowDc = BeginPaint(hwnd, &ps);

        RECT rc;
        GetClientRect(hwnd, &rc);
        const int backW = std::max(1, static_cast<int>(rc.right - rc.left));
        const int backH = std::max(1, static_cast<int>(rc.bottom - rc.top));
        HDC memDc = CreateCompatibleDC(windowDc);
        HBITMAP backBitmap = CreateCompatibleBitmap(windowDc, backW, backH);
        HGDIOBJ oldBitmap = SelectObject(memDc, backBitmap);
        HDC hdc = memDc;

        HBRUSH bg = CreateSolidBrush(RGB(248, 248, 248));
        FillRect(hdc, &rc, bg);
        DeleteObject(bg);

        ProgressSnapshot snap;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snap.title = snapshot_.title;
            snap.progress = snapshot_.progress;
            snap.labels = snapshot_.labels;
            snap.image = snapshot_.image.clone();
        }

        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, RGB(24, 24, 24));

        const int pad = 16;
        const int panelW = 330;
        const int clientW = static_cast<int>(rc.right - rc.left);
        const int clientH = static_cast<int>(rc.bottom - rc.top);
        const int imageAreaW = std::max(10, clientW - panelW - pad * 3);
        const int imageAreaH = std::max(10, clientH - pad * 2);

        if (!snap.image.empty()) {
            cv::Mat bgr = snap.image;
            if (!bgr.isContinuous()) bgr = bgr.clone();

            const double scale = std::min(
                static_cast<double>(imageAreaW) / static_cast<double>(bgr.cols),
                static_cast<double>(imageAreaH) / static_cast<double>(bgr.rows)
            );
            const int drawW = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
            const int drawH = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));
            const int drawX = pad + (imageAreaW - drawW) / 2;
            const int drawY = pad + (imageAreaH - drawH) / 2;

            BITMAPINFO bmi{};
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth = bgr.cols;
            bmi.bmiHeader.biHeight = -bgr.rows;
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 24;
            bmi.bmiHeader.biCompression = BI_RGB;
            StretchDIBits(
                hdc,
                drawX,
                drawY,
                drawW,
                drawH,
                0,
                0,
                bgr.cols,
                bgr.rows,
                bgr.data,
                &bmi,
                DIB_RGB_COLORS,
                SRCCOPY
            );
        } else {
            RECT emptyRc{pad, pad, pad + imageAreaW, pad + imageAreaH};
            DrawTextA(hdc, "Waiting for optimizer image...", -1, &emptyRc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        }

        const int panelX = pad * 2 + imageAreaW;
        RECT titleRc{panelX, pad, rc.right - pad, pad + 34};
        DrawTextA(hdc, snap.title.c_str(), -1, &titleRc, DT_LEFT | DT_TOP | DT_SINGLELINE | DT_END_ELLIPSIS);

        const int barY = pad + 48;
        RECT barOuter{panelX, barY, rc.right - pad, barY + 18};
        HBRUSH barBg = CreateSolidBrush(RGB(226, 226, 226));
        FillRect(hdc, &barOuter, barBg);
        DeleteObject(barBg);
        RECT barFill = barOuter;
        barFill.right = barFill.left + static_cast<LONG>((barOuter.right - barOuter.left) * snap.progress);
        HBRUSH barBrush = CreateSolidBrush(RGB(28, 112, 216));
        FillRect(hdc, &barFill, barBrush);
        DeleteObject(barBrush);

        std::ostringstream percent;
        percent << std::fixed << std::setprecision(1) << (snap.progress * 100.0) << "%";
        RECT percentRc{panelX, barY + 24, rc.right - pad, barY + 46};
        DrawTextA(hdc, percent.str().c_str(), -1, &percentRc, DT_LEFT | DT_TOP | DT_SINGLELINE);

        int y = barY + 58;
        for (const std::string& line : snap.labels) {
            RECT textRc{panelX, y, rc.right - pad, y + 22};
            DrawTextA(hdc, line.c_str(), -1, &textRc, DT_LEFT | DT_TOP | DT_SINGLELINE | DT_END_ELLIPSIS);
            y += 24;
            if (y > rc.bottom - pad) break;
        }

        BitBlt(windowDc, 0, 0, backW, backH, memDc, 0, 0, SRCCOPY);
        SelectObject(memDc, oldBitmap);
        DeleteObject(backBitmap);
        DeleteDC(memDc);
        EndPaint(hwnd, &ps);
    }

    std::atomic<bool> running_{false};
    std::atomic<HWND> hwnd_{nullptr};
    std::thread thread_;
    std::mutex mutex_;
    ProgressSnapshot snapshot_;
#endif
};

static std::string formatDouble(double value, int precision = 3) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << value;
    return os.str();
}

static std::vector<std::string> makeProgressLabels(
    const std::string& jobName,
    const Options& opt,
    int startIndex,
    int startCount,
    double clearance,
    int outer,
    const Validation& validation,
    int activeObject,
    int activeEdge,
    const MklSolveResult* mklResult
) {
    std::vector<std::string> labels;
    labels.push_back("Job: " + jobName);
    labels.push_back("Start: " + std::to_string(startIndex + 1) + " / " + std::to_string(startCount));
    labels.push_back("Clearance: " + formatDouble(clearance, 3));
    const int outerDisplay = std::max(1, std::min(opt.maxOuterPerClearance, outer));
    labels.push_back("Outer pass: " + std::to_string(outerDisplay) + " / " + std::to_string(opt.maxOuterPerClearance));
    labels.push_back("Area: " + formatDouble(validation.area, 3));
    labels.push_back("Min SDF on edges: " + formatDouble(validation.minSdfOnEdges, 3));
    labels.push_back("Outside pixels: " + std::to_string(validation.outsideObjectCount));
    labels.push_back("Active object samples: " + std::to_string(activeObject));
    labels.push_back("Active edge samples: " + std::to_string(activeEdge));
    labels.push_back("Starts requested: " + std::to_string(opt.starts));
    labels.push_back("Start variance: " + formatDouble(opt.startVariance, 2));
    labels.push_back("Angle variance: " + formatDouble(opt.angleVariance, 3));
    labels.push_back("Edge samples/edge: " + std::to_string(opt.edgeSamplesPerEdge));
    labels.push_back("MKL max iterations: " + std::to_string(opt.mklMaxIterations));
    labels.push_back("Dense edge step: " + formatDouble(opt.denseEdgeStep, 2));
    labels.push_back("Clearances: " + std::string(opt.fastClearances ? "fast" : "full"));
    labels.push_back("Polish: " + std::string(opt.polish ? "on" : "off"));
    labels.push_back("Min gain pct: " + formatDouble(opt.minGainPct, 3));
    labels.push_back("Alpha threshold: " + std::to_string(opt.alphaThreshold));
    labels.push_back("Seed: " + std::to_string(opt.seed));
    if (mklResult) {
        labels.push_back("MKL iterations: " + std::to_string(mklResult->iterations));
        labels.push_back("MKL stop: " + std::to_string(mklResult->stopCriterion));
        labels.push_back("MKL residual: " + formatDouble(mklResult->finalResidual, 6));
    }
    return labels;
}

static void updateProgressUi(
    ProgressUi* ui,
    const ImageData& img,
    const Poly8& p,
    const std::string& jobName,
    const Options& opt,
    int startIndex,
    int startCount,
    int clearanceIndex,
    int clearanceCount,
    double clearance,
    int outer,
    int activeObject,
    int activeEdge,
    const MklSolveResult* mklResult,
    const std::string& phase
) {
    if (!ui) return;
    const int unitsPerStart = clearanceCount * std::max(1, opt.maxOuterPerClearance);
    const int doneUnits = startIndex * unitsPerStart + clearanceIndex * opt.maxOuterPerClearance + std::max(0, outer);
    const int totalUnits = std::max(1, startCount * unitsPerStart);
    const double progress = std::min(0.99, static_cast<double>(doneUnits) / static_cast<double>(totalUnits));
    Validation validation = validateDense(p, img, clearance, opt.denseEdgeStep);
    std::vector<std::string> labels = makeProgressLabels(
        jobName,
        opt,
        startIndex,
        startCount,
        clearance,
        outer,
        validation,
        activeObject,
        activeEdge,
        mklResult
    );
    labels.insert(labels.begin(), "Phase: " + phase);
    ui->update(makeOverlay(img, p), progress, "8-gon optimization progress", std::move(labels));
}

static SolveResult solve8gon(const ImageData& img, const Options& opt, ProgressUi* ui, const std::string& jobName) {
    const std::vector<double> clearances = opt.fastClearances
        ? std::vector<double>{1.0, opt.finalClearance}
        : std::vector<double>{4.0, 2.0, 1.0, 0.5, opt.finalClearance};
    std::vector<Poly8> starts = generateStarts(img, opt);
    std::vector<Vec2> baseBoundary = deterministicSample(img.boundaryPixels, std::min(opt.maxActiveObject, 800), opt.seed);

    std::optional<SolveResult> globalBest;
    if (ui) {
        ui->update(
            makeBasePreview(img),
            0.0,
            "8-gon optimization progress",
            {
                "Phase: loaded image",
                "Job: " + jobName,
                "Image: " + std::to_string(img.w) + "x" + std::to_string(img.h),
                "Object pixels: " + std::to_string(img.objectPixels.size()),
                "Boundary pixels: " + std::to_string(img.boundaryPixels.size()),
                "Starts requested: " + std::to_string(opt.starts),
                "Start variance: " + formatDouble(opt.startVariance, 2),
                "Angle variance: " + formatDouble(opt.angleVariance, 3),
                "Clearances: " + std::string(opt.fastClearances ? "fast" : "full"),
                "Polish: " + std::string(opt.polish ? "on" : "off"),
                "Min gain pct: " + formatDouble(opt.minGainPct, 3),
                "Alpha threshold: " + std::to_string(opt.alphaThreshold),
                "Seed: " + std::to_string(opt.seed)
            }
        );
    }

    for (int si = 0; si < static_cast<int>(starts.size()); ++si) {
        Poly8 p;
        try {
            p = makeValidStart(starts[si], img, opt, clearances.front());
        } catch (const std::exception& e) {
            if (opt.verbose) std::cerr << "start " << si << " failed: " << e.what() << "\n";
            continue;
        }

        if (opt.verbose) {
            std::cerr << "start " << si << " initial area=" << area(p) << "\n";
        }

        std::vector<Vec2> activeObject = baseBoundary;
        std::vector<EdgeSample> activeEdge;

        for (int ci = 0; ci < static_cast<int>(clearances.size()); ++ci) {
            const double clearance = clearances[ci];
            Poly8 direct = sdfSafeFreeShrink(p, img, opt, clearance, opt.seed + si * 7919 + ci * 104729);
            if (isValidDense(direct, img, clearance, opt.denseEdgeStep) && area(direct) + 1e-6 < area(p)) {
                p = direct;
                if (opt.verbose) {
                    std::cerr << "  sdf-safe direct area=" << area(p) << " clr=" << clearance << "\n";
                }
            }
            ViolationSet vio = collectViolations(p, img, clearance, opt.denseEdgeStep, 256, 256);
            addUniqueObjectSamples(activeObject, vio.outsidePoints, opt.maxActiveObject);
            addUniqueEdgeSamples(activeEdge, vio.badEdgeSamples, 1200);

            updateProgressUi(
                ui,
                img,
                p,
                jobName,
                opt,
                si,
                static_cast<int>(starts.size()),
                ci,
                static_cast<int>(clearances.size()),
                clearance,
                1,
                static_cast<int>(activeObject.size()),
                static_cast<int>(activeEdge.size()),
                nullptr,
                "sdf-safe direct shrink"
            );

            if (opt.verbose) {
                Validation v = validateDense(p, img, clearance, opt.denseEdgeStep);
                std::cerr << "  clr=" << clearance
                          << " area=" << area(p)
                          << " minSdf=" << v.minSdfOnEdges
                          << " outside=" << v.outsideObjectCount
                          << " activeObj=" << activeObject.size()
                          << " activeEdge=" << activeEdge.size()
                          << "\n";
            }
            if (opt.polish) {
                p = deterministicPolish(p, img, opt, clearance);
            }
            updateProgressUi(
                ui,
                img,
                p,
                jobName,
                opt,
                si,
                static_cast<int>(starts.size()),
                ci,
                static_cast<int>(clearances.size()),
                clearance,
                opt.maxOuterPerClearance,
                static_cast<int>(activeObject.size()),
                static_cast<int>(activeEdge.size()),
                nullptr,
                "polished clearance stage"
            );
        }

        Validation finalVal = validateDense(p, img, opt.finalClearance, opt.denseEdgeStep);
        if (finalVal.simple && finalVal.contains && finalVal.edgeClear) {
            if (!globalBest || finalVal.area < globalBest->validation.area) {
                globalBest = SolveResult{p, finalVal, si};
            }
        }
    }

    if (!globalBest) throw std::runtime_error("No valid 8-gon found; increase starts/margins or relax clearance");
    int finalFruitless = 0;
    double previousBestArea = globalBest->validation.area;
    for (int round = 0; finalFruitless < FRUITLESS_ROUND_PATIENCE; ++round) {
        Poly8 candidate = sdfSafeFreeShrink(
            globalBest->poly,
            img,
            opt,
            opt.finalClearance,
            opt.seed + 1000003 + round * 9176
        );
        Validation candidateVal = validateDense(candidate, img, opt.finalClearance, opt.denseEdgeStep);
        bool improved = false;
        double gainPct = 0.0;
        if (candidateVal.simple && candidateVal.contains && candidateVal.edgeClear && candidateVal.area + 1e-6 < globalBest->validation.area) {
            gainPct = 100.0 * (globalBest->validation.area - candidateVal.area) / std::max(globalBest->validation.area, 1e-9);
            globalBest = SolveResult{candidate, candidateVal, globalBest->startIndex};
            improved = gainPct >= opt.minGainPct;
            previousBestArea = candidateVal.area;
        }

        if (improved) {
            finalFruitless = 0;
        } else {
            ++finalFruitless;
        }

        if (ui) {
            std::vector<std::string> labels = {
                "Phase: final convergence",
                "Job: " + jobName,
                "Round: " + std::to_string(round + 1),
                "Fruitless rounds: " + std::to_string(finalFruitless) + " / " + std::to_string(FRUITLESS_ROUND_PATIENCE),
                "Best area: " + formatDouble(globalBest->validation.area, 3),
                "Last gain pct: " + formatDouble(gainPct, 4),
                "Min gain pct: " + formatDouble(opt.minGainPct, 3),
                "Min SDF on edges: " + formatDouble(globalBest->validation.minSdfOnEdges, 3),
                "Outside pixels: " + std::to_string(globalBest->validation.outsideObjectCount)
            };
            ui->update(makeOverlay(img, globalBest->poly), 0.99, "8-gon optimization progress", std::move(labels));
        }

        if (opt.verbose) {
            std::cerr << "  final round=" << round
                      << " area=" << globalBest->validation.area
                      << " gainPct=" << gainPct
                      << " fruitless=" << finalFruitless
                      << "\n";
        }
        (void)previousBestArea;
    }
    return *globalBest;
}

static void printUsage(const char* argv0) {
    std::cerr << "Usage:\n"
              << "  " << argv0 << " input.png overlay.png [--starts N] [--alpha T] [--seed S]\n"
              << "Options:\n"
              << "  --fast            use cheaper preview settings for much faster iteration\n"
              << "  --starts N        number of rotated support-octagon starts; default 48\n"
              << "  --alpha T         alpha threshold; default 10\n"
              << "  --seed S          deterministic sampling seed; default 1\n"
              << "  --max-active N    maximum active object samples; default 1400\n"
              << "  --edge-samples N  residual edge samples per edge; default 40\n"
              << "  --outer N         outer cut-generation passes per clearance; default 6\n"
              << "  --mkl-iter N      MKL solve iterations per local solve; default 160\n"
              << "  --trial-iter N    MKL trial-step iterations per local solve; default 80\n"
              << "  --dense-step V    dense validation edge step in pixels; default 0.35\n"
              << "  --min-gain-pct V  stop shrink rounds when area gain falls below V percent; default 0.05\n"
              << "  --auto-points     keep adding polygon points while area improves enough\n"
              << "  --point-gain-pct V stop adding points when best added-point gain is below V percent; default 0.1\n"
              << "  --max-points N    maximum auto-points polygon vertices; default 24\n"
              << "  --point-visibility use contour-point visibility graph / DP solver\n"
              << "  --point-candidates N contour samples for point visibility; default 128\n"
              << "  --min-points N    first point count tested by visibility solver; default 3\n"
              << "  --point-patience N stop visibility point growth after N low-gain counts; default 5\n"
              << "  --no-cuda         disable CUDA visibility graph offload\n"
              << "  --cpu-threads N   CPU worker threads; default 2 with CUDA, all cores without CUDA\n"
              << "  --genetic         run genetic valid-polygon search after local SDF-safe shrink\n"
              << "  --ga-pop N        genetic population size; default 96\n"
              << "  --ga-gen N        genetic generations; default 80\n"
              << "  --ga-mutation V   genetic mutation scale in pixels; default 3\n"
              << "  --fast-clearances use only 1.0px and final clearance stages\n"
              << "  --no-polish       skip deterministic vertex polish sweeps\n"
              << "  --start-variance V random support-plane/margin spread in pixels; default 32\n"
              << "  --angle-variance V random start-angle spread in radians; default 0.18\n"
              << "  --no-ui           disable the Win32 progress window\n"
              << "  --quiet           reduce progress logging\n";
}

static void applyFastPreset(Options& opt) {
    opt.starts = 16;
    opt.maxActiveObject = 450;
    opt.edgeSamplesPerEdge = 14;
    opt.maxOuterPerClearance = 2;
    opt.mklMaxIterations = 45;
    opt.mklMaxTrialStepIterations = 20;
    opt.denseEdgeStep = 1.25;
    opt.startVariance = std::max(opt.startVariance, 80.0);
    opt.angleVariance = std::max(opt.angleVariance, 0.35);
    opt.gaPopulation = 48;
    opt.gaGenerations = 36;
    opt.gaMutation = std::max(opt.gaMutation, 3.5);
    opt.pointCandidates = std::min(opt.pointCandidates, 96);
    opt.fastClearances = true;
    opt.polish = false;
}

static Options parseOptions(int argc, char** argv, std::string& inPath, std::string& outPath) {
    if (argc < 3) {
        printUsage(argv[0]);
        std::exit(2);
    }
    inPath = argv[1];
    outPath = argv[2];
    Options opt;
    bool explicitStarts = false;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto needValue = [&](const std::string& name) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return std::string(argv[++i]);
        };
        if (a == "--fast") applyFastPreset(opt);
        else if (a == "--starts") {
            opt.starts = std::stoi(needValue(a));
            explicitStarts = true;
        }
        else if (a == "--alpha") opt.alphaThreshold = std::stoi(needValue(a));
        else if (a == "--seed") opt.seed = std::stoi(needValue(a));
        else if (a == "--max-active") opt.maxActiveObject = std::stoi(needValue(a));
        else if (a == "--edge-samples") opt.edgeSamplesPerEdge = std::stoi(needValue(a));
        else if (a == "--outer") opt.maxOuterPerClearance = std::stoi(needValue(a));
        else if (a == "--mkl-iter") opt.mklMaxIterations = std::stoi(needValue(a));
        else if (a == "--trial-iter") opt.mklMaxTrialStepIterations = std::stoi(needValue(a));
        else if (a == "--dense-step") opt.denseEdgeStep = std::stod(needValue(a));
        else if (a == "--min-gain-pct") opt.minGainPct = std::stod(needValue(a));
        else if (a == "--auto-points") opt.autoPoints = true;
        else if (a == "--point-gain-pct") opt.pointGainPct = std::stod(needValue(a));
        else if (a == "--max-points") opt.maxPoints = std::stoi(needValue(a));
        else if (a == "--point-visibility") opt.pointVisibility = true;
        else if (a == "--point-candidates") opt.pointCandidates = std::stoi(needValue(a));
        else if (a == "--min-points") opt.minPoints = std::stoi(needValue(a));
        else if (a == "--point-patience") opt.pointPatience = std::stoi(needValue(a));
        else if (a == "--no-cuda") opt.useCuda = false;
        else if (a == "--cpu-threads") opt.cpuThreads = std::stoi(needValue(a));
        else if (a == "--genetic") opt.genetic = true;
        else if (a == "--ga-pop") opt.gaPopulation = std::stoi(needValue(a));
        else if (a == "--ga-gen") opt.gaGenerations = std::stoi(needValue(a));
        else if (a == "--ga-mutation") opt.gaMutation = std::stod(needValue(a));
        else if (a == "--fast-clearances") opt.fastClearances = true;
        else if (a == "--no-polish") opt.polish = false;
        else if (a == "--start-variance") opt.startVariance = std::stod(needValue(a));
        else if (a == "--angle-variance") opt.angleVariance = std::stod(needValue(a));
        else if (a == "--no-ui") opt.showUi = false;
        else if (a == "--quiet") opt.verbose = false;
        else throw std::runtime_error("Unknown option: " + a);
    }
    if (explicitStarts) {
        for (int i = 3; i < argc; ++i) {
            if (std::string(argv[i]) == "--starts" && i + 1 < argc) {
                opt.starts = std::stoi(argv[i + 1]);
            }
        }
    }
    return opt;
}

} // namespace mkl8gon

int main(int argc, char** argv) {
    try {
        std::string inputPath;
        std::string outputPath;
        mkl8gon::Options opt = mkl8gon::parseOptions(argc, argv, inputPath, outputPath);

        int workerThreads = opt.cpuThreads;
        if (workerThreads <= 0) {
            workerThreads = opt.useCuda ? 2 : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        }
        workerThreads = std::max(1, workerThreads);
        mkl_set_num_threads_local(workerThreads);
        #ifdef _OPENMP
        omp_set_num_threads(workerThreads);
        #endif

        mkl8gon::ProgressUi progressUi;
        mkl8gon::ProgressUi* progressUiPtr = nullptr;
        if (opt.showUi && progressUi.start("polygon optimization progress")) {
            progressUiPtr = &progressUi;
        }

        if (std::filesystem::is_directory(inputPath)) {
            std::filesystem::create_directories(outputPath);
            std::vector<std::filesystem::path> sprites;
            for (const auto& entry : std::filesystem::directory_iterator(inputPath)) {
                if (!entry.is_regular_file()) continue;
                std::filesystem::path path = entry.path();
                if (path.extension() != ".png") continue;
                const std::string stem = path.stem().string();
                if (stem.find("overlay") != std::string::npos ||
                    stem.find("pointvis") != std::string::npos ||
                    stem.find("converge") != std::string::npos ||
                    stem.find("autopoints") != std::string::npos ||
                    stem.find("sdfsafe") != std::string::npos ||
                    stem.find("50starts") != std::string::npos ||
                    stem.find("points-") != std::string::npos) {
                    continue;
                }
                sprites.push_back(path);
            }
            std::sort(sprites.begin(), sprites.end());

            int ok = 0;
            int fail = 0;
            for (int i = 0; i < static_cast<int>(sprites.size()); ++i) {
                const std::filesystem::path& spritePath = sprites[i];
                const std::string stem = spritePath.stem().string();
                const std::filesystem::path overlayPath = std::filesystem::path(outputPath) / (stem + ".png");
                const std::filesystem::path jsonPath = std::filesystem::path(outputPath) / (stem + ".json");
                try {
                    mkl8gon::ImageData batchImg = mkl8gon::loadImage(spritePath.string(), opt.alphaThreshold);
                    auto onFeasible = [&](int vertices, const mkl8gon::PolyN& candidate, double area, double bestArea) {
                        if (!progressUiPtr) return;
                        cv::Mat overlay = mkl8gon::makeOverlay(batchImg, candidate);
                        const double local = static_cast<double>(vertices - std::max(3, opt.minPoints) + 1) /
                            static_cast<double>(std::max(std::max(3, opt.minPoints), opt.maxPoints) - std::max(3, opt.minPoints) + 1);
                        const double progress = (static_cast<double>(i) + 0.85 * local) / std::max(1.0, static_cast<double>(sprites.size()));
                        progressUi.update(
                            overlay,
                            progress,
                            "batch point visibility optimization",
                            {
                                "Phase: solving sprite",
                                "Sprite: " + spritePath.filename().string(),
                                "Output: " + overlayPath.string(),
                                "Batch: " + std::to_string(i + 1) + " / " + std::to_string(sprites.size()),
                                "Vertices: " + std::to_string(vertices),
                                "Area: " + mkl8gon::formatDouble(area, 3),
                                "Best area: " + mkl8gon::formatDouble(bestArea, 3),
                                "Succeeded: " + std::to_string(ok),
                                "Failed: " + std::to_string(fail)
                            }
                        );
                    };

                    mkl8gon::PolyN poly = opt.pointVisibility
                        ? mkl8gon::solveVisibilityPointPolygon(batchImg, opt, onFeasible)
                        : mkl8gon::toPolyN(mkl8gon::solve8gon(batchImg, opt, nullptr, spritePath.string()).poly);
                    mkl8gon::Validation validation = mkl8gon::validateDense(poly, batchImg, opt.finalClearance, opt.denseEdgeStep);
                    cv::Mat overlay = mkl8gon::makeOverlay(batchImg, poly);
                    cv::imwrite(overlayPath.string(), overlay);

                    std::ofstream js(jsonPath);
                    js << std::fixed << std::setprecision(6);
                    js << "{\n";
                    js << "  \"area\": " << validation.area << ",\n";
                    js << "  \"vertices\": [\n";
                    for (size_t vi = 0; vi < poly.v.size(); ++vi) {
                        js << "    " << mkl8gon::fmt(poly.v[vi]) << (vi + 1 == poly.v.size() ? "\n" : ",\n");
                    }
                    js << "  ]\n";
                    js << "}\n";
                    ++ok;

                    if (progressUiPtr) {
                        progressUi.update(
                            overlay,
                            static_cast<double>(i + 1) / std::max(1.0, static_cast<double>(sprites.size())),
                            "batch point visibility optimization",
                            {
                                "Phase: sprite complete",
                                "Sprite: " + spritePath.filename().string(),
                                "Output: " + overlayPath.string(),
                                "Batch: " + std::to_string(i + 1) + " / " + std::to_string(sprites.size()),
                                "Area: " + mkl8gon::formatDouble(validation.area, 3),
                                "Vertices: " + std::to_string(poly.v.size()),
                                "Succeeded: " + std::to_string(ok),
                                "Failed: " + std::to_string(fail)
                            }
                        );
                    }
                } catch (const std::exception& e) {
                    ++fail;
                    std::ofstream log(std::filesystem::path(outputPath) / (stem + ".error.txt"));
                    log << e.what() << "\n";
                }
            }
            std::cerr << "Batch complete ok=" << ok << " fail=" << fail << " out=" << outputPath << "\n";
            if (progressUiPtr) {
                progressUi.waitUntilClosed();
            }
            return fail == 0 ? 0 : 1;
        }

        mkl8gon::ImageData img = mkl8gon::loadImage(inputPath, opt.alphaThreshold);
        std::cerr << "Loaded " << img.w << "x" << img.h
                  << " objectPixels=" << img.objectPixels.size()
                  << " boundaryPixels=" << img.boundaryPixels.size() << "\n";

        if (opt.pointVisibility) {
            const int minPoints = std::max(3, opt.minPoints);
            const int maxPoints = std::max(minPoints, opt.maxPoints);
            auto onFeasible = [&](int vertices, const mkl8gon::PolyN& candidate, double area, double bestArea) {
                cv::Mat intermediateOverlay = mkl8gon::makeOverlay(img, candidate);
                const std::string intermediatePath = mkl8gon::numberedOutputPath(outputPath, vertices);
                cv::imwrite(intermediatePath, intermediateOverlay);
                if (progressUiPtr) {
                    progressUi.update(
                        intermediateOverlay,
                        static_cast<double>(vertices - minPoints + 1) / static_cast<double>(maxPoints - minPoints + 1),
                        "point visibility optimization",
                        {
                            "Phase: evaluating point count",
                            "Job: " + inputPath,
                            "Output: " + intermediatePath,
                            "Solver: point visibility DP",
                            "Vertices: " + std::to_string(vertices),
                            "Area: " + mkl8gon::formatDouble(area, 3),
                            "Best area: " + mkl8gon::formatDouble(bestArea, 3),
                            "Candidates: " + std::to_string(opt.pointCandidates),
                            "Point gain pct: " + mkl8gon::formatDouble(opt.pointGainPct, 3),
                            "Point patience: " + std::to_string(opt.pointPatience)
                        }
                    );
                }
            };
            mkl8gon::PolyN poly = mkl8gon::solveVisibilityPointPolygon(img, opt, onFeasible);
            mkl8gon::Validation validation = mkl8gon::validateDense(poly, img, opt.finalClearance, opt.denseEdgeStep);
            cv::Mat overlay = mkl8gon::makeOverlay(img, poly);
            if (!cv::imwrite(outputPath, overlay)) {
                throw std::runtime_error("Could not write overlay: " + outputPath);
            }
            if (progressUiPtr) {
                progressUi.update(
                    overlay,
                    1.0,
                    "point visibility optimization complete",
                    {
                        "Phase: complete",
                        "Job: " + inputPath,
                        "Output: " + outputPath,
                        "Solver: point visibility DP",
                        "Vertices: " + std::to_string(poly.v.size()),
                        "Area: " + mkl8gon::formatDouble(validation.area, 3),
                        "Point gain pct: " + mkl8gon::formatDouble(opt.pointGainPct, 3),
                        "Point patience: " + std::to_string(opt.pointPatience),
                        "Candidates: " + std::to_string(opt.pointCandidates),
                        "Min SDF on edges: " + mkl8gon::formatDouble(validation.minSdfOnEdges, 3),
                        "Contains object: " + std::string(validation.contains ? "true" : "false"),
                        "Edge clear: " + std::string(validation.edgeClear ? "true" : "false")
                    }
                );
            }
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "{\n";
            std::cout << "  \"area\": " << validation.area << ",\n";
            std::cout << "  \"vertices\": [\n";
            for (size_t i = 0; i < poly.v.size(); ++i) {
                std::cout << "    " << mkl8gon::fmt(poly.v[i]) << (i + 1 == poly.v.size() ? "\n" : ",\n");
            }
            std::cout << "  ]\n";
            std::cout << "}\n";
            std::cerr << "Wrote overlay: " << outputPath << "\n";
            if (progressUiPtr) {
                std::cerr << "Debug UI will stay open until you close its window.\n";
                progressUi.waitUntilClosed();
            }
            return 0;
        }

        if (opt.autoPoints) {
            mkl8gon::SolveResult base = mkl8gon::solve8gon(img, opt, progressUiPtr, inputPath);
            mkl8gon::PolyN poly = mkl8gon::addPointsUntilFruitless(mkl8gon::toPolyN(base.poly), img, opt, opt.seed + 424242);
            mkl8gon::Validation validation = mkl8gon::validateDense(poly, img, opt.finalClearance, opt.denseEdgeStep);
            cv::Mat overlay = mkl8gon::makeOverlay(img, poly);
            if (!cv::imwrite(outputPath, overlay)) {
                throw std::runtime_error("Could not write overlay: " + outputPath);
            }
            if (progressUiPtr) {
                progressUi.update(
                    overlay,
                    1.0,
                    "polygon optimization complete",
                    {
                        "Phase: complete",
                        "Job: " + inputPath,
                        "Output: " + outputPath,
                        "Vertices: " + std::to_string(poly.v.size()),
                        "Area: " + mkl8gon::formatDouble(validation.area, 3),
                        "Point gain pct: " + mkl8gon::formatDouble(opt.pointGainPct, 3),
                        "Genetic: " + std::string(opt.genetic ? "true" : "false"),
                        "Min SDF on edges: " + mkl8gon::formatDouble(validation.minSdfOnEdges, 3),
                        "Contains object: " + std::string(validation.contains ? "true" : "false"),
                        "Edge clear: " + std::string(validation.edgeClear ? "true" : "false")
                    }
                );
            }
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "{\n";
            std::cout << "  \"area\": " << validation.area << ",\n";
            std::cout << "  \"vertices\": [\n";
            for (size_t i = 0; i < poly.v.size(); ++i) {
                std::cout << "    " << mkl8gon::fmt(poly.v[i]) << (i + 1 == poly.v.size() ? "\n" : ",\n");
            }
            std::cout << "  ]\n";
            std::cout << "}\n";
            std::cerr << "Wrote overlay: " << outputPath << "\n";
            if (progressUiPtr) {
                std::cerr << "Debug UI will stay open until you close its window.\n";
                progressUi.waitUntilClosed();
            }
            return 0;
        }

        mkl8gon::SolveResult result = mkl8gon::solve8gon(img, opt, progressUiPtr, inputPath);
        cv::Mat overlay = mkl8gon::makeOverlay(img, result.poly);
        if (!cv::imwrite(outputPath, overlay)) {
            throw std::runtime_error("Could not write overlay: " + outputPath);
        }
        if (progressUiPtr) {
            progressUi.update(
                overlay,
                1.0,
                "8-gon optimization complete",
                {
                    "Phase: complete",
                    "Job: " + inputPath,
                    "Output: " + outputPath,
                    "Best start: " + std::to_string(result.startIndex + 1),
                    "Area: " + mkl8gon::formatDouble(result.validation.area, 3),
                    "Min SDF on edges: " + mkl8gon::formatDouble(result.validation.minSdfOnEdges, 3),
                    "Contains object: " + std::string(result.validation.contains ? "true" : "false"),
                    "Edge clear: " + std::string(result.validation.edgeClear ? "true" : "false")
                }
            );
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\n";
        std::cout << "  \"area\": " << result.validation.area << ",\n";
        std::cout << "  \"startIndex\": " << result.startIndex << ",\n";
        std::cout << "  \"simple\": " << (result.validation.simple ? "true" : "false") << ",\n";
        std::cout << "  \"contains\": " << (result.validation.contains ? "true" : "false") << ",\n";
        std::cout << "  \"edgeClear\": " << (result.validation.edgeClear ? "true" : "false") << ",\n";
        std::cout << "  \"minSdfOnEdges\": " << result.validation.minSdfOnEdges << ",\n";
        std::cout << "  \"vertices\": [\n";
        for (int i = 0; i < mkl8gon::K; ++i) {
            std::cout << "    " << mkl8gon::fmt(result.poly.v[i]) << (i + 1 == mkl8gon::K ? "\n" : ",\n");
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
        std::cerr << "Wrote overlay: " << outputPath << "\n";
        if (progressUiPtr) {
            std::cerr << "Debug UI will stay open until you close its window.\n";
            progressUi.waitUntilClosed();
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
