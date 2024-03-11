
// #define PROFILE  // need to define in order to turn on timing

#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <vector>
// #include "openfhecore.h"
// #include "time.h"
// #include "math/math-hal.h"
#include <cuda_runtime.h>
#include "ntt.cuh" //issue with curand kernel

#define BARRETT_64 // TODO: ajaveed, fix this stupid requirement for reduction definition

//using namespace lbcrypto;


// main()   need this for Kurts' makefile to ignore this.
int main(int argc, char* argv[]) {
    std::cout<<" -- SU GPU NTT Demo!!! -- "<<std::endl;
    ntt_configuration nttcfg;
    nttcfg.n_power = 2;
    std::cout << "-- alisah nttcfg.n_power: " << nttcfg.n_power << std::endl;
    NTTParameters parameters(12, ModularReductionType::BARRET, ReductionPolynomial::X_N_minus);
    // ^__ TODO: uncomment and still problematic figure it out

        int deviceCount;

    // Initialize the CUDA runtime
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found on this system." << std::endl;
        return 0;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        // You can print more information about the device using other properties of cudaDeviceProp.
    }


   
    return 0;
}

// function to compare two BigVectors and print differing indicies
// void vec_diff(BigVector& a, BigVector& b) {
//     for (usint i = 0; i < a.GetLength(); ++i) {
//         if (a.at(i) != b.at(i)) {
//             std::cout << "i: " << i << std::endl;
//             std::cout << "first vector " << std::endl;
//             std::cout << a.at(i);
//             std::cout << std::endl;
//             std::cout << "second vector " << std::endl;
//             std::cout << b.at(i);
//             std::cout << std::endl;
//         }
//     }
// }

// // function to compare two Poly and print differing values
// bool clonetest(Poly& a, Poly& b, std::string name) {
//     if (a != b) {
//         std::cout << name << " FAILED " << std::endl;
//         return true;
//     }
//     else {
//         return false;
//     }
// }

