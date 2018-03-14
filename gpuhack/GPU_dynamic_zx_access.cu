//
// Created by cheesema on 09.03.18.
//
#include <algorithm>
#include <vector>
#include <array>
#include <iostream>
#include <cassert>
#include <limits>
#include <chrono>
#include <iomanip>

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/ExtraParticleData.hpp"
#include "data_structures/Mesh/MeshData.hpp"
#include "io/TiffUtils.hpp"

#include "thrust/device_vector.h"
#include "thrust/tuple.h"
#include "thrust/copy.h"
#include "../src/misc/APRTimer.hpp"
#include "../src/data_structures/APR/ExtraParticleData.hpp"
#include "../../../../../Developer/NVIDIA/CUDA-9.1/include/thrust/tuple.h"

#include "GPUAPRAccess.hpp"





struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
};

bool command_option_exists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv) {
    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"Example_apr_neighbour_access -i input_apr_file -d directory\"" << std::endl;
        exit(1);
    }
    if(command_option_exists(argv, argv + argc, "-i")) {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }
    if(command_option_exists(argv, argv + argc, "-d")) {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }
    if(command_option_exists(argv, argv + argc, "-o")) {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    return result;
}




__global__ void load_balance_xzl(const thrust::tuple<std::size_t,std::size_t>* row_info,std::size_t*  _chunk_index_end,
                                 std::size_t total_number_chunks,std::float_t parts_per_block,std::size_t total_number_rows);

__global__ void test_dynamic_balance(const GPUAccessPtrs* gpuAccessPtrs,std::uint16_t* particle_data_output);

__global__ void test_dynamic_balance_XZYL(const GPUAccessPtrs* gpuAccessPtrs,std::uint16_t* particle_data_output);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    // Read provided APR file
    cmdLineOptions options = read_command_line_options(argc, argv);
    const int reps = 100;

    std::string fileName = options.directory + options.input;
    APR<uint16_t> apr;
    apr.read_apr(fileName);

    // Get dense representation of APR
    APRIterator<uint16_t> aprIt(apr);

    APRTimer timer;
    timer.verbose_flag = true;


    /*
     * Set up the GPU Access data structures
     *
     */

    GPUAPRAccess gpuaprAccess(apr);

    /*
     *  Now launch the kernels across all the chunks determiend by the load balancing
     *
     */

    ExtraParticleData<uint16_t> iteration_check_particles(apr);
    iteration_check_particles.init_gpu(apr.total_number_particles());

    int number_reps = 40;

    timer.start_timer("iterate over all particles");

    dim3 threads_dyn(32);
    dim3 blocks_dyn((gpuaprAccess.actual_number_chunks + threads_dyn.x - 1)/threads_dyn.x);

    for (int rep = 0; rep < number_reps; ++rep) {

        test_dynamic_balance << < blocks_dyn, threads_dyn >> > (gpuaprAccess.gpu_access_ptr, iteration_check_particles.gpu_pointer);
        cudaDeviceSynchronize();
    }

    timer.stop_timer();

    float gpu_iterate_time = timer.timings.back();

    /*
     *  Off-load the particle data from the GPU
     *
     */

    timer.start_timer("output transfer from GPU");

    iteration_check_particles.copy_data_to_host();

    timer.stop_timer();

    /*
    *  Test the x,y,z,level information is correct
    *
    */

    ExtraParticleData<uint16_t> spatial_info_test(apr);
    spatial_info_test.init_gpu(apr.total_number_particles());

    timer.start_timer("summing the sptial informatino for each partilce on the GPU");
    for (int rep = 0; rep < number_reps; ++rep) {

        test_dynamic_balance_XZYL << < blocks_dyn, threads_dyn >> >
                                                   (gpuaprAccess.gpu_access_ptr, spatial_info_test.gpu_pointer);

        cudaDeviceSynchronize();
    }

    timer.stop_timer();

    float gpu_iterate_time_si = timer.timings.back();
    //copy data back from gpu
    spatial_info_test.copy_data_to_host();


    /*
    *  Performance comparison with CPU
    *
    */

    ExtraParticleData<uint16_t> test_cpu(apr);

    timer.start_timer("Performance comparison on CPU serial");
    for (int rep = 0; rep < number_reps; ++rep) {
        for (uint64_t particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            aprIt.set_iterator_to_particle_by_number(particle_number);

            test_cpu[aprIt] += 1;

        }
    }

    timer.stop_timer();

    float cpu_iterate_time = timer.timings.back();



    timer.start_timer("Performance comparison on CPU access sum"); //not working
    for (int rep = 0; rep < number_reps; ++rep) {

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(aprIt)
        for (uint64_t particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            aprIt.set_iterator_to_particle_by_number(particle_number);

            test_cpu[aprIt] = aprIt.x() + aprIt.y() + aprIt.z() + aprIt.level();

        }
    }

    timer.stop_timer();

    float cpu_iterate_time_si = timer.timings.back();

    std::cout << "SPEEDUP GPU vs. CPU iterate= " << cpu_iterate_time/gpu_iterate_time << std::endl;
    std::cout << "SPEEDUP GPU vs. CPU iterate (Spatial Info)= " << cpu_iterate_time_si/gpu_iterate_time_si << std::endl;

    //////////////////////////
    ///
    /// Now check the data
    ///
    ////////////////////////////


    bool success = true;

    uint64_t c_fail= 0;
    uint64_t c_pass= 0;

    for (uint64_t particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        aprIt.set_iterator_to_particle_by_number(particle_number);
        if(iteration_check_particles[aprIt]==number_reps){
            c_pass++;
        } else {
            c_fail++;
            success = false;
            //std::cout << test_access_data[particle_number] << " Level: " < aprIt.level() << std::endl;
        }
    }

    if(success){
        std::cout << "Iteration Check, PASS" << std::endl;
    } else {
        std::cout << "Iteration Check, FAIL Total: " << c_fail << " Pass Total:  " << c_pass << std::endl;
    }


    /*
     *  Check the spatial data, by comparing x+y+z+level for every particle
     *
     */

    c_pass = 0;
    c_fail = 0;
    success=true;


    for (uint64_t particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        aprIt.set_iterator_to_particle_by_number(particle_number);
        if(spatial_info_test[aprIt]==(aprIt.x() + aprIt.y() + aprIt.z() + aprIt.level())){
            c_pass++;
        } else {
            c_fail++;
            success = false;
            //std::cout << test_access_data[particle_number] << " Level: " < aprIt.level() << std::endl;
        }
    }

    if(success){
        std::cout << "Spatial information Check, PASS" << std::endl;
    } else {
        std::cout << "Spatial information Check, FAIL Total: " << c_fail << " Pass Total:  " << c_pass << std::endl;
    }

}

__global__ void test_dynamic_balance(const GPUAccessPtrs* gpuAccessPtrs,std::uint16_t* particle_data_output){
    //
    //  This kernel checks that every particle is only visited once in the iteration
    //

    int chunk_index = blockDim.x * blockIdx.x + threadIdx.x; // the input to each kernel is its chunk index for which it should iterate over

    if(chunk_index >= gpuAccessPtrs->total_number_chunks){
        return; //out of bounds
    }

    //load in the begin and end row indexs
    std::size_t row_begin;
    std::size_t row_end;

    if(chunk_index==0){
        row_begin = 0;
    } else {
        row_begin = gpuAccessPtrs->_chunk_index_end[chunk_index-1] + 1; //This chunk starts the row after the last one finished.
    }

    row_end = gpuAccessPtrs->_chunk_index_end[chunk_index];

    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    std::size_t current_row_key;

    for (std::size_t current_row = row_begin; current_row <= row_end; ++current_row) {
        current_row_key = thrust::get<0>(gpuAccessPtrs->row_info[current_row]);
        if(current_row_key&1) { //checks if there any particles in the row

            particle_global_index_end = thrust::get<1>(gpuAccessPtrs->row_info[current_row]);

            if (current_row == 0) {
                particle_global_index_begin = 0;
            } else {
                particle_global_index_begin = thrust::get<1>(gpuAccessPtrs->row_info[current_row-1]);
            }

            //loop over the particles in the row
            for (std::size_t particle_global_index = particle_global_index_begin; particle_global_index < particle_global_index_end; ++particle_global_index) {

                particle_data_output[particle_global_index]+=1;
            }
        }
    }


}

__global__ void test_dynamic_balance_XZYL(const GPUAccessPtrs* gpuAccessPtrs,std::uint16_t* particle_data_output){
    //
    //  This kernel loops over particles and saves the sum of x+z+y+l as a test for the access
    //

    int chunk_index = blockDim.x * blockIdx.x + threadIdx.x; // the input to each kernel is its chunk index for which it should iterate over

    if(chunk_index >= gpuAccessPtrs->total_number_chunks){
        return; //out of bounds
    }

    std::uint16_t local_kernel[3][3][3];

    //load in the begin and end row indexs
    std::size_t row_begin;
    std::size_t row_end;

    if(chunk_index==0){
        row_begin = 0;
    } else {
        row_begin = gpuAccessPtrs->_chunk_index_end[chunk_index-1] + 1; //This chunk starts the row after the last one finished.
    }

    row_end = gpuAccessPtrs->_chunk_index_end[chunk_index];

    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    std::size_t current_row_key;

    for (std::size_t current_row = row_begin; current_row <= row_end; ++current_row) {
        current_row_key = thrust::get<0>(gpuAccessPtrs->row_info[current_row]);
        if(current_row_key&1) { //checks if there any particles in the row

            particle_global_index_end = thrust::get<1>(gpuAccessPtrs->row_info[current_row]);

            if (current_row == 0) {
                particle_global_index_begin = 0;
            } else {
                particle_global_index_begin = thrust::get<1>(gpuAccessPtrs->row_info[current_row-1]);
            }

            std::uint16_t x;
            std::uint16_t z;
            std::uint8_t level;

            //decode the key
            x = (current_row_key & KEY_X_MASK) >> KEY_X_SHIFT;
            z = (current_row_key & KEY_Z_MASK) >> KEY_Z_SHIFT;
            level = (current_row_key & KEY_LEVEL_MASK) >> KEY_LEVEL_SHIFT;

            //loop over the particles in the row
            for (std::size_t particle_global_index = particle_global_index_begin; particle_global_index < particle_global_index_end; ++particle_global_index) {
                uint16_t current_y = gpuAccessPtrs->y_part_coord[particle_global_index];
                particle_data_output[particle_global_index]=current_y+x+z+level;

                local_kernel[1][1][1] = current_y;
            }

        }

    }


}






