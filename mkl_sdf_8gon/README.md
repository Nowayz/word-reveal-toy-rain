# MKL + SDF minimum-area 8-gon fitter

This is a C++17 reference implementation for fitting a low-area 8-vertex polygon around a PNG alpha mask.

Pipeline:

1. Load PNG with OpenCV, using the alpha channel if available.
2. Build a signed distance field with a CPU Jump Flood Algorithm approximation.
3. Generate multiple guaranteed-valid convex 8-gon starts using support half-planes.
4. Optimize free 8-gon vertices with Intel oneMKL `dtrnlspbc_*` nonlinear least-squares.
5. Enforce geometry with an external validity oracle: polygon simplicity, mask containment, and SDF edge clearance.
6. Repair invalid MKL proposals by binary-searching from the last valid polygon.
7. Add active violated samples and repeat.
8. Run deterministic local polish.

## Dependencies

- Intel oneMKL 2025.x or compatible oneAPI MKL installation.
- OpenCV 4.x.
- CMake 3.20+.

## Build

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -DMKL_DIR=$MKLROOT/lib/cmake/mkl
cmake --build build -j
```

## Run

```bash
./build/mkl_sdf_8gon input.png overlay.png --starts 48 --alpha 10 --seed 1
```

The program prints area, validation diagnostics, and the final 8 vertices to stdout.

## Notes

This is a serious robust solver, not a formal global-optimality proof. The bitmap itself makes the target discrete/sampled. To push closer to a global answer, increase `--starts`, decrease final clearance, and run several seeds.
