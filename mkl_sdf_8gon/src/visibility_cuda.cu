#include "visibility_cuda.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

static __device__ __forceinline__ double clampd(double x, double lo, double hi) {
    return fmin(fmax(x, lo), hi);
}

static __device__ __forceinline__ float sample_sdf(const float* sdf, int w, int h, double px, double py) {
    const double x = clampd(px, 0.0, static_cast<double>(w - 1));
    const double y = clampd(py, 0.0, static_cast<double>(h - 1));
    const int x0 = static_cast<int>(floor(x));
    const int y0 = static_cast<int>(floor(y));
    const int x1 = min(w - 1, x0 + 1);
    const int y1 = min(h - 1, y0 + 1);
    const double dx = x - x0;
    const double dy = y - y0;
    const float v00 = sdf[y0 * w + x0];
    const float v10 = sdf[y0 * w + x1];
    const float v01 = sdf[y1 * w + x0];
    const float v11 = sdf[y1 * w + x1];
    return static_cast<float>(
        (1.0 - dx) * (1.0 - dy) * v00 +
        dx * (1.0 - dy) * v10 +
        (1.0 - dx) * dy * v01 +
        dx * dy * v11
    );
}

static __global__ void visibility_kernel(
    const double* xy,
    int n,
    const float* sdf,
    int w,
    int h,
    double clearance,
    double step,
    unsigned char* out_visible
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * n;
    if (id >= total) return;

    const int i = id / n;
    const int j = id - i * n;
    if (i == j) {
        out_visible[id] = 0;
        return;
    }

    const double ax = xy[i * 2 + 0];
    const double ay = xy[i * 2 + 1];
    const double bx = xy[j * 2 + 0];
    const double by = xy[j * 2 + 1];
    const double dx = bx - ax;
    const double dy = by - ay;
    const double len = sqrt(dx * dx + dy * dy);
    const int samples = max(2, static_cast<int>(ceil(len / fmax(step, 0.25))));
    const double tolerance = clearance <= 0.0 ? 1.25 : 1.0e-6;

    for (int s = 0; s <= samples; ++s) {
        const double t = static_cast<double>(s) / static_cast<double>(samples);
        const double px = ax + dx * t;
        const double py = ay + dy * t;
        if (sample_sdf(sdf, w, h, px, py) < clearance - tolerance) {
            out_visible[id] = 0;
            return;
        }
    }
    out_visible[id] = 1;
}

extern "C" int mkl8gon_build_visibility_cuda(
    const double* xy,
    int n,
    const float* sdf,
    int w,
    int h,
    double clearance,
    double step,
    uint8_t* out_visible
) {
    if (!xy || !sdf || !out_visible || n <= 0 || w <= 0 || h <= 0) return 1;

    double* d_xy = nullptr;
    float* d_sdf = nullptr;
    unsigned char* d_visible = nullptr;
    const size_t xy_bytes = static_cast<size_t>(n) * 2u * sizeof(double);
    const size_t sdf_bytes = static_cast<size_t>(w) * static_cast<size_t>(h) * sizeof(float);
    const size_t visible_bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(unsigned char);

    cudaError_t err = cudaMalloc(&d_xy, xy_bytes);
    if (err != cudaSuccess) return 2;
    err = cudaMalloc(&d_sdf, sdf_bytes);
    if (err != cudaSuccess) { cudaFree(d_xy); return 3; }
    err = cudaMalloc(&d_visible, visible_bytes);
    if (err != cudaSuccess) { cudaFree(d_xy); cudaFree(d_sdf); return 4; }

    err = cudaMemcpy(d_xy, xy, xy_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(d_sdf, sdf, sdf_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_xy);
        cudaFree(d_sdf);
        cudaFree(d_visible);
        return 5;
    }

    const int total = n * n;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    visibility_kernel<<<grid, block>>>(d_xy, n, d_sdf, w, h, clearance, step, d_visible);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(out_visible, d_visible, visible_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_xy);
    cudaFree(d_sdf);
    cudaFree(d_visible);
    return err == cudaSuccess ? 0 : 6;
}

static __device__ __forceinline__ double cross_xy(const double* xy, int a, int b) {
    return xy[a * 2 + 0] * xy[b * 2 + 1] - xy[a * 2 + 1] * xy[b * 2 + 0];
}

static __global__ void dp_anchor_kernel(
    const double* xy,
    const unsigned char* visible,
    int n,
    int vertices,
    int anchor_step,
    double* anchor_areas,
    int* anchor_paths,
    int anchor_count
) {
    constexpr double INF_D = 1.0e100;
    constexpr int MAX_N = 160;
    constexpr int MAX_VERTICES = 16;
    constexpr int MAX_STRIDE = MAX_N * 2 + 1;

    const int anchor_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor_index >= anchor_count || n > MAX_N || vertices > MAX_VERTICES) return;

    const int anchor = anchor_index * anchor_step;
    const int limit = anchor + n;
    const int stride = limit + 1;
    if (stride > MAX_STRIDE) return;

    double dp[MAX_VERTICES + 1][MAX_STRIDE];
    int prev[MAX_VERTICES + 1][MAX_STRIDE];
    for (int u = 0; u <= vertices; ++u) {
        for (int i = 0; i < stride; ++i) {
            dp[u][i] = INF_D;
            prev[u][i] = -1;
        }
    }
    dp[1][anchor] = 0.0;

    for (int used = 1; used < vertices; ++used) {
        for (int i = anchor; i < limit; ++i) {
            const double base_cost = dp[used][i];
            if (base_cost >= INF_D * 0.5) continue;
            const int ii = i % n;
            const int min_j = i + 1;
            const int max_j = limit - (vertices - used - 1);
            for (int j = min_j; j <= max_j; ++j) {
                const int jj = j % n;
                if (!visible[ii * n + jj]) continue;
                const double cost = base_cost + cross_xy(xy, ii, jj);
                if (cost < dp[used + 1][j]) {
                    dp[used + 1][j] = cost;
                    prev[used + 1][j] = i;
                }
            }
        }
    }

    double best = INF_D;
    int best_end = -1;
    for (int end = anchor + vertices - 1; end < limit; ++end) {
        const double path_cost = dp[vertices][end];
        if (path_cost >= INF_D * 0.5) continue;
        const int ee = end % n;
        if (!visible[ee * n + anchor]) continue;
        const double twice_area = path_cost + cross_xy(xy, ee, anchor);
        if (twice_area > 0.0 && twice_area < best) {
            best = twice_area;
            best_end = end;
        }
    }

    anchor_areas[anchor_index] = best;
    int* path = anchor_paths + anchor_index * vertices;
    for (int i = 0; i < vertices; ++i) path[i] = -1;
    if (best_end < 0) return;

    int cur = best_end;
    for (int used = vertices; used >= 1; --used) {
        path[used - 1] = cur % n;
        cur = prev[used][cur];
    }
}

extern "C" int mkl8gon_solve_visibility_dp_cuda(
    const double* xy,
    const uint8_t* visible,
    int n,
    int vertices,
    int anchor_step,
    int max_out,
    double* out_xy,
    double* out_twice_area,
    int* out_count
) {
    if (!xy || !visible || !out_xy || !out_twice_area || !out_count || n <= 0 || vertices < 3 || anchor_step <= 0 || max_out <= 0) return 1;
    if (n > 160 || vertices > 16) return 2;

    const int anchor_count = (n + anchor_step - 1) / anchor_step;
    const size_t xy_bytes = static_cast<size_t>(n) * 2u * sizeof(double);
    const size_t visible_bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(uint8_t);
    const size_t area_bytes = static_cast<size_t>(anchor_count) * sizeof(double);
    const size_t path_bytes = static_cast<size_t>(anchor_count) * static_cast<size_t>(vertices) * sizeof(int);

    double* d_xy = nullptr;
    unsigned char* d_visible = nullptr;
    double* d_areas = nullptr;
    int* d_paths = nullptr;
    cudaError_t err = cudaMalloc(&d_xy, xy_bytes);
    if (err != cudaSuccess) return 3;
    err = cudaMalloc(&d_visible, visible_bytes);
    if (err != cudaSuccess) { cudaFree(d_xy); return 4; }
    err = cudaMalloc(&d_areas, area_bytes);
    if (err != cudaSuccess) { cudaFree(d_xy); cudaFree(d_visible); return 5; }
    err = cudaMalloc(&d_paths, path_bytes);
    if (err != cudaSuccess) { cudaFree(d_xy); cudaFree(d_visible); cudaFree(d_areas); return 6; }

    err = cudaMemcpy(d_xy, xy, xy_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(d_visible, visible, visible_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_xy); cudaFree(d_visible); cudaFree(d_areas); cudaFree(d_paths);
        return 7;
    }

    const int block = 128;
    const int grid = (anchor_count + block - 1) / block;
    dp_anchor_kernel<<<grid, block>>>(d_xy, d_visible, n, vertices, anchor_step, d_areas, d_paths, anchor_count);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_xy); cudaFree(d_visible); cudaFree(d_areas); cudaFree(d_paths);
        return 8;
    }

    double* areas = new double[anchor_count];
    int* paths = new int[anchor_count * vertices];
    err = cudaMemcpy(areas, d_areas, area_bytes, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(paths, d_paths, path_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_xy); cudaFree(d_visible); cudaFree(d_areas); cudaFree(d_paths);
    if (err != cudaSuccess) {
        delete[] areas; delete[] paths;
        return 9;
    }

    int count = 0;
    std::vector<unsigned char> used(static_cast<size_t>(anchor_count), 0);
    for (; count < max_out; ++count) {
        int best_anchor = -1;
        double best_area = 1.0e100;
        for (int i = 0; i < anchor_count; ++i) {
            if (!used[static_cast<size_t>(i)] && areas[i] < best_area) {
                best_area = areas[i];
                best_anchor = i;
            }
        }
        if (best_anchor < 0 || best_area >= 1.0e99) break;
        used[static_cast<size_t>(best_anchor)] = 1;
        out_twice_area[count] = best_area;
        for (int i = 0; i < vertices; ++i) {
            const int id = paths[best_anchor * vertices + i];
            if (id < 0 || id >= n) {
                out_twice_area[count] = 1.0e100;
                continue;
            }
            out_xy[(static_cast<size_t>(count) * static_cast<size_t>(vertices) + static_cast<size_t>(i)) * 2u + 0u] = xy[id * 2 + 0];
            out_xy[(static_cast<size_t>(count) * static_cast<size_t>(vertices) + static_cast<size_t>(i)) * 2u + 1u] = xy[id * 2 + 1];
        }
    }
    *out_count = count;
    delete[] areas; delete[] paths;
    return count > 0 ? 0 : 10;
}
