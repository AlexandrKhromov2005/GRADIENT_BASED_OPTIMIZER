#pragma once
#include <cstddef>
#define VEC_SIZE_DEFAULT 22
#define POP_SIZE 30
#define ITERATIONS 40
#define TH 10.0
#define M_PI 3.14159265358979323846
#define SIGN(x) ((x) >= 0.0? 1.0 : -1.0)
#define PR 0.5
#define WM_SIZE 1024
//#define EPS 0.05

// Global dynamic vector size (initialized in embedding_schemes.cpp)
extern size_t CURRENT_VEC_SIZE;