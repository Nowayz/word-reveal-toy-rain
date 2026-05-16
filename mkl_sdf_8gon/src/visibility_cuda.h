#pragma once

#include <cstdint>

extern "C" int mkl8gon_build_visibility_cuda(
    const double* xy,
    int n,
    const float* sdf,
    int w,
    int h,
    double clearance,
    double step,
    uint8_t* out_visible
);

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
);
