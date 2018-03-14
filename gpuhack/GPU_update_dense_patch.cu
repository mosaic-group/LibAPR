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


__global__ void update_dense_patch(const thrust::tuple<std::size_t,std::size_t>* row_info,const std::size_t*  _chunk_index_end,
                                   std::size_t total_number_chunks,const std::uint16_t* particle_y,const std::uint16_t* particles_input,std::uint16_t* particles_output);

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

    int number_reps = 1;

    timer.start_timer("iterate over all particles");

    dim3 threads_dyn(32);
    dim3 blocks_dyn((gpuaprAccess.actual_number_chunks + threads_dyn.x - 1)/threads_dyn.x);

    /*
    *  Test the x,y,z,level information is correct
    *
    */

    ExtraParticleData<uint16_t> dense_patch_output(apr);

    apr.particles_intensities.copy_data_to_gpu();

    timer.start_timer("summing the sptial informatino for each partilce on the GPU");
    for (int rep = 0; rep < number_reps; ++rep) {

        update_dense_patch <<< blocks_dyn, threads_dyn >>> (gpuaprAccess.row_info,gpuaprAccess._chunk_index_end,gpuaprAccess.actual_number_chunks,gpuaprAccess.y_part_coord,apr.particles_intensities.gpu_pointer,dense_patch_output.gpu_pointer);

        cudaDeviceSynchronize();
    }

    timer.stop_timer();

    float gpu_iterate_time_si = timer.timings.back();
    //copy data back from gpu
    dense_patch_output.copy_data_to_host();


}




__global__ void update_dense_patch(const thrust::tuple<std::size_t,std::size_t>* row_info,const std::size_t*  _chunk_index_end,
                                          std::size_t total_number_chunks,const std::uint16_t* particle_y,const std::uint16_t* particles_input,std::uint16_t* particles_output){


    int chunk_index = blockDim.x * blockIdx.x + threadIdx.x; // the input to each kernel is its chunk index for which it should iterate over

    if(chunk_index >=total_number_chunks){
        return; //out of bounds
    }

    std::uint16_t local_patch[3][3][3];

    std::uint16_t local_y[3][3];

    //load in the begin and end row indexs
    std::size_t row_begin;
    std::size_t row_end;

    if(chunk_index==0){
        row_begin = 0;
    } else {
        row_begin = _chunk_index_end[chunk_index-1] + 1; //This chunk starts the row after the last one finished.
    }

    row_end = _chunk_index_end[chunk_index];

    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    std::size_t current_row_key;

    for (std::size_t current_row = row_begin; current_row <= row_end; ++current_row) {
        current_row_key = thrust::get<0>(row_info[current_row]);
        if(current_row_key&1) { //checks if there any particles in the row

            particle_global_index_end = thrust::get<1>(row_info[current_row]);

            if (current_row == 0) {
                particle_global_index_begin = 0;
            } else {
                particle_global_index_begin = thrust::get<1>(row_info[current_row-1]);
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
                uint16_t current_y = particle_y[particle_global_index];

                //local_patch[1][1][1] = particles_input[particle_global_index] + current_y;

                uint16_t test = particles_output[particle_global_index];
                particles_output[particle_global_index] = 0;

            }

        }

    }


}






