cmake_minimum_required(VERSION 3.20)
project(mkl_sdf_8gon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Run Intel oneAPI environment first, e.g.:
#   source /opt/intel/oneapi/setvars.sh
# Then configure with:
#   cmake -S . -B build -DMKL_DIR=$MKLROOT/lib/cmake/mkl
if (DEFINED ENV{MKLROOT})
    file(TO_CMAKE_PATH "$ENV{MKLROOT}" MKLROOT_CMAKE)
    find_package(MKL CONFIG REQUIRED PATHS "${MKLROOT_CMAKE}" "${MKLROOT_CMAKE}/lib/cmake/mkl")
else()
    find_package(MKL CONFIG REQUIRED)
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(OpenMP)

add_executable(mkl_sdf_8gon src/main.cpp)
target_link_libraries(mkl_sdf_8gon PRIVATE MKL::MKL ${OpenCV_LIBS})
target_include_directories(mkl_sdf_8gon PRIVATE ${OpenCV_INCLUDE_DIRS})
if (OpenMP_CXX_FOUND)
    target_link_libraries(mkl_sdf_8gon PRIVATE OpenMP::OpenMP_CXX)
endif()

if (MSVC)
    # Tune for current workstation: safe, high-optimization defaults.
    # Keep /O2 and LTCG because this app is numeric/geometry-heavy.
        target_compile_options(
            mkl_sdf_8gon
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/W4>
                $<$<COMPILE_LANGUAGE:CXX>:/permissive->
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/Ob2>
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/GL>
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/MP>
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/DNDEBUG>
        )
    set_target_properties(mkl_sdf_8gon PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
    )
    target_link_options(
        mkl_sdf_8gon
        PRIVATE
            $<$<CONFIG:Release>:/LTCG>
    )
    target_link_libraries(mkl_sdf_8gon PRIVATE user32 gdi32)
else()
    target_compile_options(
        mkl_sdf_8gon
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-march=native>
            $<$<CONFIG:Release>:-mtune=native>
            $<$<CONFIG:Release>:-ffast-math>
            $<$<CONFIG:Release>:-funroll-loops>
    )
    set_target_properties(mkl_sdf_8gon PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
    )
    target_link_options(
        mkl_sdf_8gon
        PRIVATE
            $<$<CONFIG:Release>:-flto>
    )
endif()
