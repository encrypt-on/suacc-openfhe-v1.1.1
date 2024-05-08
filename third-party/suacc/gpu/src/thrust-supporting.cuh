// //704::added
// #ifdef _TH_SUPPORTED
// #define _TH_SUPPORTED

// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
// #include <thrust/transform.h>
// #include <thrust/for_each.h>
// #include <iostream>

// // Functor to sum pairwise elements and store the result in the preceding index
// struct SumPairwiseFunctor {
//     uint64_t* data; // Pointer to the device vector data

//     // Constructor to initialize the data pointer
//     SumPairwiseFunctor(uint64_t* data) : data(data) {}

//     // Functor operator
//     __host__ __device__
//     void operator()(size_t index) const {
//         if (index > 0) {
//             // Sum pairwise elements and store the result in the preceding index
//             data[index - 1] += data[index];
//         }
//     }
// };

// #endif