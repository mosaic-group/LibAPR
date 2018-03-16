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

#include "GPUAPRAccess.hpp"
#include "../src/data_structures/APR/APR.hpp"

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



__global__ void shared_update(const thrust::tuple <std::size_t, std::size_t> *row_info,
                              const std::size_t *_chunk_index_end,
                              const std::uint16_t *particle_y,
                              const std::uint16_t *particle_data_input,
                              std::uint16_t *particle_data_output,
                              std::size_t offset,
                              std::size_t x_num,
                              std::size_t z_num,
                              std::size_t y_num,
                              std::size_t level);

__global__ void shared_update_conv(const thrust::tuple <std::size_t, std::size_t> *row_info,
                                   const std::size_t *_chunk_index_end,
                                   const std::uint16_t *particle_y,
                                   const std::uint16_t *particle_data_input,
                                   std::uint16_t *particle_data_output,
                                   const std::size_t offset,
                                   const std::size_t x_num,
                                   const std::size_t z_num,
                                   const std::size_t y_num,
                                   const std::size_t level);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    // Read provided APR file
    cmdLineOptions options = read_command_line_options(argc, argv);

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

    int number_reps = 500;

    timer.start_timer("iterate over all particles");


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

    apr.particles_intensities.copy_data_to_gpu();


    timer.start_timer("summing the sptial informatino for each partilce on the GPU");
    for (int rep = 0; rep < number_reps; ++rep) {

        for (int level = apr.level_min(); level <= aprIt.level_max(); ++level) {

            std::size_t number_rows_l = apr.spatial_index_x_max(level) * apr.spatial_index_z_max(level);
            std::size_t offset = gpuaprAccess.h_level_offset[level];

            std::size_t x_num = apr.spatial_index_x_max(level);
            std::size_t z_num = apr.spatial_index_z_max(level);
            std::size_t y_num = apr.spatial_index_y_max(level);


            dim3 threads_l(10, 1, 10);

            int x_blocks = (x_num + 8 - 1) / 8;
            int z_blocks = (z_num + 8 - 1) / 8;

            //std::cout << "xb: " << x_blocks << " zb: " << z_blocks << std::endl;

            dim3 blocks_l(x_blocks, 1, z_blocks);

            shared_update <<< blocks_l, threads_l >>>
                                     (gpuaprAccess.gpu_access.row_info, gpuaprAccess.gpu_access._chunk_index_end, gpuaprAccess.gpu_access.y_part_coord, apr.particles_intensities.gpu_pointer,spatial_info_test.gpu_pointer, offset,x_num,z_num,y_num,level);

            cudaDeviceSynchronize();
        }
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

            test_cpu[aprIt] = apr.particles_intensities[aprIt];

        }
    }

    timer.stop_timer();

    float cpu_iterate_time = timer.timings.back();


    std::cout << "SPEEDUP GPU level (2D) vs. CPU iterate (Insert Intensity)= " << cpu_iterate_time/gpu_iterate_time_si << std::endl;

    ExtraParticleData<uint16_t> spatial_info_test2(apr);
    spatial_info_test2.init_gpu(apr.total_number_particles());


    timer.start_timer("summing the sptial informatino for each partilce on the GPU");
    for (int rep = 0; rep < number_reps; ++rep) {

        for (int level = apr.level_min(); level <= apr.level_max(); ++level) {

            std::size_t number_rows_l = apr.spatial_index_x_max(level) * apr.spatial_index_z_max(level);
            std::size_t offset = gpuaprAccess.h_level_offset[level];

            std::size_t x_num = apr.spatial_index_x_max(level);
            std::size_t z_num = apr.spatial_index_z_max(level);
            std::size_t y_num = apr.spatial_index_y_max(level);

            dim3 threads_l(10, 1, 10);

            int x_blocks = (x_num + 8 - 1) / 8;
            int z_blocks = (z_num + 8 - 1) / 8;

            dim3 blocks_l(x_blocks, 1, z_blocks);

            shared_update_conv <<< blocks_l, threads_l >>>
                                        (gpuaprAccess.gpu_access.row_info, gpuaprAccess.gpu_access._chunk_index_end, gpuaprAccess.gpu_access.y_part_coord, apr.particles_intensities.gpu_pointer,spatial_info_test2.gpu_pointer, offset,x_num,z_num,y_num,level);

            cudaDeviceSynchronize();
        }
    }

    timer.stop_timer();

    float gpu_iterate_time_si2 = timer.timings.back();
    //copy data back from gpu
    spatial_info_test2.copy_data_to_host();


    std::cout << "SPEEDUP GPU level (2D) + CONV vs. CPU iterate (Insert Intensities)= " << cpu_iterate_time/gpu_iterate_time_si2 << std::endl;


    std::cout << "Average time for loop conv: " << (gpu_iterate_time_si2/(number_reps*1.0f))*1000 << " ms" << std::endl;
    std::cout << "Average time for loop insert: " << (gpu_iterate_time_si/(number_reps*1.0f))*1000 << " ms" << std::endl;

    //////////////////////////
    ///
    /// Now check the data
    ///
    ////////////////////////////


    bool success = true;

    uint64_t c_fail= 0;
    uint64_t c_pass= 0;


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
        //if(spatial_info_test[aprIt]==(aprIt.x() + aprIt.y() + aprIt.z() + aprIt.level())){
        if(spatial_info_test[aprIt]==apr.particles_intensities[aprIt]){
            c_pass++;
        } else {
            c_fail++;
            success = false;
            if(aprIt.level() == (aprIt.level_min()+1)) {
                std::cout << spatial_info_test[aprIt] << " Level: " << aprIt.level() << " x: " << aprIt.x() << " z: " << aprIt.z() << std::endl;
            }
        }
    }

    if(success){
        std::cout << "Spatial information Check, PASS" << std::endl;
    } else {
        std::cout << "Spatial information Check, FAIL Total: " << c_fail << " Pass Total:  " << c_pass << std::endl;
    }

}


    //
    //  This kernel checks that every particle is only visited once in the iteration
    //


__global__ void shared_update_conv(const thrust::tuple <std::size_t, std::size_t> *row_info,
                              const std::size_t *_chunk_index_end,
                              const std::uint16_t *particle_y,
                              const std::uint16_t *particle_data_input,
                              std::uint16_t *particle_data_output,
                              const std::size_t offset,
                              const std::size_t x_num,
                              const std::size_t z_num,
                              const std::size_t y_num,
                              const std::size_t level) {

    const unsigned int N = 1; //1 + 7 seems optimal, removes bank conflicts.
    const unsigned int Nx = 8;
    const unsigned int Nz = 8;


    __shared__ int local_patch[Nz+2][Nx+2][N+7]; // This is block wise shared memory this is assuming an 8*8 block with pad()

    uint16_t y_cache[N]={0}; // These are local register/private caches
    uint16_t index_cache[N]={0}; // These are local register/private caches

    int x_index = (8 * blockIdx.x + threadIdx.x - 1);
    int z_index = (8 * blockIdx.z + threadIdx.z - 1);


    if(x_index >= x_num || x_index < 0){
        return; //out of bounds
    }

    if(z_index >= z_num || z_index < 0){
        return; //out of bounds
    }

    if(threadIdx.x >= 10){
        return;
    }

    if(threadIdx.z >= 10){
        return;
    }

    int current_row = offset + (x_index) + (z_index)*x_num; // the input to each kernel is its chunk index for which it should iterate over

    bool not_ghost=false;

    if((threadIdx.x > 0) && (threadIdx.x < 9) && (threadIdx.z > 0) && (threadIdx.z < 9)){
        not_ghost = true;
    }


    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    particle_global_index_end = thrust::get<1>(row_info[current_row]);

    if (current_row == 0) {
        particle_global_index_begin = 0;
    } else {
        particle_global_index_begin = thrust::get<1>(row_info[current_row-1]);
    }

    std::size_t y_block = 1;
    std::size_t y_counter = 0;

    double neighbour_sum = 0;

    std::size_t particle_global_index = particle_global_index_begin;
    while( particle_global_index < particle_global_index_end){

        uint16_t current_y = particle_y[particle_global_index];

        while(current_y >= y_block*N){
            //threads need to wait for there progression
            __syncthreads();
            y_block++;

            //Do the cached loop

            for (int i = 0; i < y_counter; ++i) {
                //T->P to

                if (not_ghost) {
                    int lower_bound = (1);

                    for (int q = -(lower_bound); q < (lower_bound + 1); ++q) {     // z stencil
                        for (int l = -(lower_bound); l < (lower_bound + 1); ++l) {   // x stencil
                            for (int w = -(lower_bound); w < (lower_bound + 1); ++w) {    // y stencil
                                neighbour_sum += local_patch[threadIdx.z + q][threadIdx.x + l][
                                        (y_cache[i]) % N + 1 + w];
                            }
                        }
                    }

                    particle_data_output[particle_global_index_begin + index_cache[i]] = std::round(
                            neighbour_sum / 27.0f);
                }
            }

            y_counter=0;
        }


        //P->T
        local_patch[threadIdx.z ][threadIdx.x ][current_y % N +1 ] = particle_data_input[particle_global_index];

        //caching for update loop
        index_cache[y_counter]=(particle_global_index-particle_global_index_begin);
        y_cache[y_counter]=current_y;
        y_counter++;

        //global index update
        particle_global_index++;
    }


    //do i need a last exit loop?
    __syncthreads();

    for (int i = 0; i < y_counter; ++i) {

        if(not_ghost) {
            int lower_bound = (1);

            for (int q = -(lower_bound); q < (lower_bound + 1); ++q) {     // z stencil
                for (int l = -(lower_bound); l < (lower_bound + 1); ++l) {   // x stencil
                    for (int w = -(lower_bound); w < (lower_bound + 1); ++w) {    // y stencil
                        neighbour_sum += local_patch[threadIdx.z + q][threadIdx.x + l][
                                (y_cache[i]) % N + 1 + w];
                    }
                }
            }

            particle_data_output[particle_global_index_begin + index_cache[i]] = std::round(
                    neighbour_sum / 27.0f);
        }
    }



}

__global__ void shared_update(const thrust::tuple <std::size_t, std::size_t> *row_info,
                              const std::size_t *_chunk_index_end,
                              const std::uint16_t *particle_y,
                              const std::uint16_t *particle_data_input,
                              std::uint16_t *particle_data_output,
                              std::size_t offset,
                              std::size_t x_num,
                              std::size_t z_num,
                              std::size_t y_num,
                              std::size_t level) {

    const unsigned int N = 1;
    const unsigned int N_t = N+2;

    __shared__ int local_patch[10][10][N+7]; // This is block wise shared memory this is assuming an 8*8 block with pad()

    uint16_t y_cache[N]={0}; // These are local register/private caches
    uint16_t index_cache[N]={0}; // These are local register/private caches

    if(threadIdx.x >= 10){
        return;
    }
    if(threadIdx.z >= 10){
        return;
    }


    int x_index = (8 * blockIdx.x + threadIdx.x - 1);
    int z_index = (8 * blockIdx.z + threadIdx.z - 1);

    bool not_ghost=false;

    if((threadIdx.x > 0) && (threadIdx.x < 9) && (threadIdx.z > 0) && (threadIdx.z < 9)){
        not_ghost = true;
    }


    if((x_index >= x_num) || (x_index < 0)){
        return; //out of bounds
    }

    if((z_index >= z_num) || (z_index < 0)){
        return; //out of bounds
    }

    int current_row = offset + (x_index) + (z_index)*x_num; // the input to each kernel is its chunk index for which it should iterate over


    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    particle_global_index_end = thrust::get<1>(row_info[current_row]);

    if (current_row == 0) {
        particle_global_index_begin = 0;
    } else {
        particle_global_index_begin = thrust::get<1>(row_info[current_row-1]);
    }

    std::size_t y_block = 1;
    std::size_t y_counter = 0;

    std::size_t particle_global_index = particle_global_index_begin;
    while( particle_global_index < particle_global_index_end){

        uint16_t current_y = particle_y[particle_global_index];

        while(current_y >= y_block*N){
            //threads need to wait for there progression
            __syncthreads();
            y_block++;

            //Do the cached loop
            //T->P to

            for (int i = 0; i < y_counter; ++i) {
                if(not_ghost) {
                    particle_data_output[particle_global_index_begin +
                                         index_cache[i]] = local_patch[threadIdx.z][threadIdx.x][(y_cache[i]) % N];
                }
            }

            y_counter=0;
        }


        //P->T
        local_patch[threadIdx.z][threadIdx.x][current_y % N ] = particle_data_input[particle_global_index];

        //caching for update loop
        index_cache[y_counter]=(particle_global_index-particle_global_index_begin);
        y_cache[y_counter]=current_y;
        y_counter++;

        //global index update
        particle_global_index++;

    }


        //do i need a last exit loop?
        __syncthreads();
    for (int i = 0; i < y_counter; ++i) {
        if(not_ghost) {
            particle_data_output[particle_global_index_begin +
                                 index_cache[i]] = local_patch[threadIdx.z][threadIdx.x][
                    (y_cache[i]) % N];
        }
    }

}






