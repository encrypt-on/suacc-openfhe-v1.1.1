#include "inc/tramp-hostiface-gpuntt.h"

#ifdef ENB_SUGPUNTT
    #include <cuda_runtime.h>
    #include "ntt.cuh"
    #include "common.cuh"
    #include <cstdlib> 
    #include <random>
    using namespace std;
    #pragma GCC diagnostic ignored "-Wsign-compare"
    #include <chrono>
    #include <exception>
    #include <fstream>
    #include <iostream>
    #include <vector>
   #include <thread>

#endif //ENB_SUGPUNTT


std::vector<uint64_t> iface_GPUNTT_ComputeForward(
                         const std::vector<uint64_t>& fInp, 
                         uint32_t len,
                         uint64_t modulus,
                         uint64_t psi,
                         uint64_t* elT
){
    std::vector<uint64_t> fOup(len, 0);
    std::vector<std::pair<std::string, uint32_t>> vtimeMeas(1, std::make_pair("", 0));
    #ifdef ENB_SUGPUNTT

        NTTFactors factor((Modulus)modulus, 1, psi); // w=1 is placeholder, not needed for X_N_plus
        NTTParameters parameters(log2(len), factor, ReductionPolynomial::X_N_plus, XFORMDIR::FWD, true);

        // safety checks
        assert(fInp.size() == len); 
        assert(modulus == factor.modulus.value);
        assert(psi == factor.psi);

        //==== 704:: cache the rou tables to prevent re-generation ====
        static std::map<std::tuple<uint64_t, uint64_t, uint64_t>, std::vector<Data>> cached_wtable;
        static std::map<std::tuple<uint64_t, uint64_t, uint64_t>, Root*> cached_fotd;
        static std::map<uint64_t, Data*> cached_IOD;
        std::tuple<uint64_t, uint64_t, uint64_t> k { fInp.size(), factor.modulus.value, factor.psi };

        if (cached_wtable.count(k)) {
            parameters.forward_root_of_unity_table = cached_wtable[k] ;
        } else {
            parameters.forward_root_of_unity_table_generator();
            cached_wtable[k] = parameters.forward_root_of_unity_table;
        }
        
        Data* InOut_Datas;
        if(cached_IOD.count(len)){
                InOut_Datas = cached_IOD[len]; 
        }else{

        THROW_IF_CUDA_ERROR(
                cudaMalloc(&InOut_Datas, 1 * parameters.n * sizeof(Data)));
            cached_IOD[len] = InOut_Datas;  
        }
        THROW_IF_CUDA_ERROR(
            cudaMemcpy(InOut_Datas, fInp.data(), parameters.n * sizeof(Data), cudaMemcpyHostToDevice));
    

        Root* Forward_Omega_Table_Device;
        if (cached_fotd.count(k)) {
            Forward_Omega_Table_Device = cached_fotd[k] ;
        } else {
        
         vector<Root_> forward_omega_table;
         forward_omega_table = parameters.gpu_root_of_unity_table_generator(parameters.forward_root_of_unity_table);        
        
         THROW_IF_CUDA_ERROR(
            cudaMalloc(&Forward_Omega_Table_Device, parameters.root_of_unity_size * sizeof(Root)));
        
         THROW_IF_CUDA_ERROR(
            cudaMemcpy(Forward_Omega_Table_Device, forward_omega_table.data(),
                            parameters.root_of_unity_size * sizeof(Root), cudaMemcpyHostToDevice));
            cached_fotd[k] = Forward_Omega_Table_Device;
        }

        ntt_configuration cfg_ntt = {
                .n_power = parameters.logn,
                .ntt_type = FORWARD,
                .reduction_poly = ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = 0
        };
        uint64_t elTime = 0;
        vtimeMeas[0].first = "GFN[k]";
        MEASURE_ELAPSEDTIME(
            GPU_NTT_Inplace(InOut_Datas, Forward_Omega_Table_Device, parameters.modulus, cfg_ntt, 1);,
            elTime
        );
        if(elT) {
            *elT = elTime;
            vtimeMeas[0].second = elTime;
        }
        
        
        #ifndef MULTITHREADED_GPUMEMCPY
            // copy result to host memory
            vtimeMeas.push_back(std::make_pair("IO_DtoH",0));
            MEASURE_ELAPSEDTIME(
            THROW_IF_CUDA_ERROR(
                cudaMemcpy(fOup.data(), InOut_Datas, parameters.n * sizeof(Data), cudaMemcpyDeviceToHost)
            );
            ,
            vtimeMeas.back().second
            );
        #else //MULTITHREADED_GPUMEMCPY - performs poor so just keep it for experimentation
            const int numThreads = std::thread::hardware_concurrency();
            const int chunkSize = len / numThreads;
            //231     567  599.6 w one thread
            std::vector<std::thread> threads;
            for (int i = 0; i < numThreads; i += 1) {
                int startIdx = i * chunkSize;
                int endIdx = (i == numThreads - 1) ? fOup.size() : (i + 1) * chunkSize;

                threads.emplace_back([=]() {
                    THROW_IF_CUDA_ERROR(
                        cudaMemcpy((Data*)fOup.data(), InOut_Datas, 
                            // parameters.n * sizeof(Data),
                            chunkSize * sizeof(Data), 
                            cudaMemcpyDeviceToHost)
                    );
                });
                threads.back().join();
            }
        #endif
        
        //zeroize cudaMallocd
        cudaMemset(InOut_Datas, 0, parameters.n * sizeof(Data));
    
    #endif //ENB_SUGPUNTT
    return fOup;
}

std::vector<uint64_t> iface_GPUNTT_ComputeInverse(
                         const std::vector<uint64_t>& iInp, 
                         uint32_t len,
                         uint64_t modulus,
                         uint64_t psi,
                         uint64_t* elT
){
    std::vector<uint64_t> iOup(len, 0);
    #ifdef ENB_SUGPUNTT
        
        NTTFactors factor((Modulus)modulus, 1, psi); // w=1 is placeholder, not needed for X_N_plus
        NTTParameters parameters(log2(len), factor, ReductionPolynomial::X_N_plus 
         , XFORMDIR::REV
        , true
        );

        assert(iInp.size() == len); // safety check
        assert(modulus == factor.modulus.value);
        assert(psi == factor.psi);

        //==== 704:: cache the rou tables to prevent re-generation =====
        static std::map<std::tuple<uint64_t, uint64_t, uint64_t>, std::vector<Data>> cached_wtable;
        static std::map<std::tuple<uint64_t, uint64_t, uint64_t>, Root*> cached_iotd;
        static std::map<uint64_t, Data*> cached_IOD; 
        std::tuple<uint64_t, uint64_t, uint64_t> k { iInp.size(), factor.modulus.value, factor.psi };

        if (cached_wtable.count(k)) {  //mod, rou, size
            parameters.inverse_root_of_unity_table = cached_wtable[k] ;
        } else {
            parameters.inverse_root_of_unity_table_generator();
            cached_wtable[k] = parameters.inverse_root_of_unity_table;
        }

        Data* InOut_Datas;

        if(cached_IOD.count(len)){
            InOut_Datas = cached_IOD[len];
        }else{
            THROW_IF_CUDA_ERROR(
                cudaMalloc(&InOut_Datas, 1 * parameters.n * sizeof(Data)));
            cached_IOD[len] = InOut_Datas; 
        }
        THROW_IF_CUDA_ERROR(
                cudaMemcpy(InOut_Datas, iInp.data(), parameters.n * sizeof(Data), cudaMemcpyHostToDevice));
        Root* Inverse_Omega_Table_Device;

        if (cached_iotd.count(k)) {
            Inverse_Omega_Table_Device = cached_iotd[k] ;
        } else {
            vector<Root_> inverse_omega_table;
            inverse_omega_table = parameters.gpu_root_of_unity_table_generator(
                    parameters.inverse_root_of_unity_table);

            THROW_IF_CUDA_ERROR(
                cudaMalloc(&Inverse_Omega_Table_Device,
                        parameters.root_of_unity_size * sizeof(Root)));

            THROW_IF_CUDA_ERROR(cudaMemcpy(
                Inverse_Omega_Table_Device, inverse_omega_table.data(),
                parameters.root_of_unity_size * sizeof(Root), cudaMemcpyHostToDevice));
            cached_iotd[k] = Inverse_Omega_Table_Device;
        }
        
        ntt_configuration cfg_intt = {
            .n_power = parameters.logn,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = parameters.n_inv,
            .stream = 0};

        
        uint64_t elTime = 0;
        MEASURE_ELAPSEDTIME (
            GPU_NTT_Inplace(InOut_Datas, Inverse_Omega_Table_Device, parameters.modulus, cfg_intt, 1);,
            elTime
        );
        if(elT) {
            *elT = elTime;
        }

        THROW_IF_CUDA_ERROR(
            cudaMemcpy(iOup.data(), InOut_Datas, parameters.n * sizeof(Data), cudaMemcpyDeviceToHost)
        );
        cudaMemset(InOut_Datas, 0, parameters.n * sizeof(Data));
    #endif //ENB_SUGPUNTT
    return iOup;
}

