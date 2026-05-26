/**
 * @file
 * @brief A collection of all of ThunderKittens prototypes, that can be filled in to easily build full kernels.
 */

#pragma once

#include "../include/kittens.cuh"

#include "common/common.cuh"
#ifdef LC3F
    #include "lc3f/lcf.cuh"
#else
    #include "lcf/lcf.cuh"
#endif
#include "lcsc/lcsc.cuh"
#include "lcsf/lcsf.cuh"
#include "interpreter/interpreter.cuh"
