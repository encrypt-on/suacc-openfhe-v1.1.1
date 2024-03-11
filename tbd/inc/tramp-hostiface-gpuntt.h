#ifndef _TRAMP_HOSTIFACE_GPUNTT
#define _TRAMP_HOSTIFACE_GPUNTT

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <cassert> 

std::vector<uint64_t> iface_GPUNTT_ComputeForward(
                         const std::vector<uint64_t>& fInp, 
                         uint32_t len,
                         uint64_t modulus,
                         uint64_t psi,
                         uint64_t* elT=nullptr
);
std::vector<uint64_t> iface_GPUNTT_ComputeInverse(
                         const std::vector<uint64_t>& iInp, 
                         uint32_t len,
                         uint64_t modulus,
                         uint64_t psi,
                         uint64_t* elT=nullptr
);

#ifdef ENB_ELAPSED_TIME_MEASUREMENT
    #define MEASURE_ELAPSEDTIME(func, elT) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        func; \
        auto stop = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
        elT = (uint64_t)duration.count(); \
} while(0)
#else
    #define MEASURE_ELAPSEDTIME(func, elT) \
    do { \
        func; \
        elT = 0; \
    } while(0)
#endif

#ifdef ENB_COUTPRINT
  #define SUDEBUG_ONLY(...) do { __VA_ARGS__ } while(0)
#else
  #define SUDEBUG_ONLY(...) do {} while(0)
#endif

#endif