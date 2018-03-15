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
#include "../../../../../Developer/NVIDIA/CUDA-9.1/include/host_defines.h"
#include "../../../../../Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/cstddef"
#include "../../../../../Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/cstdint"
#include "../../../../../usr/local/Cellar/llvm/3.8.1/include/c++/v1/cstdint"
#include "../../../../../Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/cstring"
#include "../../../../../usr/local/Cellar/llvm/3.8.1/include/c++/v1/cstddef"
#include "../../../../../usr/local/Cellar/llvm/3.8.1/include/c++/v1/cstdio"
#include "../../../../../usr/local/Cellar/llvm/3.8.1/include/c++/v1/cstdlib"
#include "../../../../../usr/local/Cellar/llvm/3.8.1/include/c++/v1/cstring"
#include "../../../../../usr/local/Cellar/llvm/3.8.1/include/c++/v1/ctime"
#include "../../../../../usr/local/Cellar/llvm/3.8.1/include/c++/v1/cwchar"
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
                              std::size_t level);


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

        for (int level = apr.level_min(); level <= apr.level_max(); ++level) {

            std::size_t number_rows_l = apr.spatial_index_x_max(level) * apr.spatial_index_z_max(level);
            std::size_t offset = gpuaprAccess.h_level_offset[level];

            std::size_t x_num = apr.spatial_index_x_max(level);
            std::size_t z_num = apr.spatial_index_z_max(level);

            dim3 threads_l(8, 1, 8);
            dim3 blocks_l((x_num + threads_l.x - 1) / threads_l.x, 1, (z_num + threads_l.z - 1) / threads_l.z);

            shared_update <<< blocks_l, threads_l >>>
                                     (gpuaprAccess.gpu_access.row_info, gpuaprAccess.gpu_access._chunk_index_end, gpuaprAccess.gpu_access.y_part_coord, apr.particles_intensities.gpu_pointer,spatial_info_test.gpu_pointer, offset,x_num,z_num,level);

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


    std::cout << "SPEEDUP GPU level (2D) vs. CPU iterate (Spatial Info)= " << cpu_iterate_time/gpu_iterate_time_si << std::endl;

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
            //std::cout << spatial_info_test[aprIt] << " Level: " << aprIt.level() << std::endl;
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




__global__ void shared_update(const thrust::tuple <std::size_t, std::size_t> *row_info,
                              const std::size_t *_chunk_index_end,
                              const std::uint16_t *particle_y,
                              const std::uint16_t *particle_data_input,
                              std::uint16_t *particle_data_output,
                              std::size_t offset,
                              std::size_t x_num,
                              std::size_t z_num,
                              std::size_t level) {

    __shared__ int local_patch[10][10][10];

    int x_index = (blockDim.x * blockIdx.x + threadIdx.x);
    int z_index = (blockDim.z * blockIdx.z + threadIdx.z);


    if(x_index >= x_num){
        return; //out of bounds
    }

    if(z_index >= z_num){
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

    //loop over the particles in the row
    for (std::size_t particle_global_index = particle_global_index_begin; particle_global_index < particle_global_index_end; ++particle_global_index) {
        uint16_t current_y = particle_y[particle_global_index];


        local_patch[threadIdx.z+1][threadIdx.x+1][current_y%10]=particle_data_input[particle_global_index];


        particle_data_output[particle_global_index]=particle_data_input[particle_global_index];

        //__syncthreads();
    }




}






