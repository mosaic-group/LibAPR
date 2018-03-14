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
                                   std::size_t total_number_chunks,const std::uint16_t* particle_y,const std::size_t* level_offsets,const std::uint16_t* level_y_num,const std::uint16_t* level_x_num,const std::uint16_t* level_z_num, std::uint16_t* particles_input,std::uint16_t* particles_output);

__global__ void update_dense_insert(
        const thrust::tuple<std::size_t,std::size_t>* row_info,
        const std::size_t*  _chunk_index_end,
        std::size_t total_number_chunks,
        const std::uint16_t* particle_y,
        const std::size_t* level_offsets,
        const std::uint16_t* level_y_num,
        const std::uint16_t* level_x_num,
        const std::uint16_t* level_z_num,
        std::uint16_t* particles_input,
        std::uint16_t* particles_output);

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

    GPUAPRAccess gpuaprAccess(apr,1000);

    int number_reps = 10;

    timer.start_timer("iterate over all particles");

    dim3 threads_dyn(32);
    dim3 blocks_dyn((gpuaprAccess.actual_number_chunks + threads_dyn.x - 1)/threads_dyn.x);

    /*
    *  Test the x,y,z,level information is correct
    *
    */

    ExtraParticleData<uint16_t> dense_patch_output(apr);
    dense_patch_output.copy_data_to_gpu();

    apr.particles_intensities.copy_data_to_gpu();


    timer.start_timer("summing the sptial informatino for each partilce on the GPU");
    for (int rep = 0; rep < number_reps; ++rep) {

        update_dense_insert <<< blocks_dyn, threads_dyn >>> (
                                                               gpuaprAccess.gpu_access.row_info,
                                                               gpuaprAccess.gpu_access._chunk_index_end,
                                                               gpuaprAccess.actual_number_chunks,
                                                               gpuaprAccess.gpu_access.y_part_coord,
                                                               gpuaprAccess.gpu_access.level_offsets,
                                                               gpuaprAccess.gpu_access.level_y_num,
                                                               gpuaprAccess.gpu_access.level_x_num,
                                                               gpuaprAccess.gpu_access.level_z_num,
                                                               apr.particles_intensities.gpu_pointer,
                                                               dense_patch_output.gpu_pointer);

        cudaDeviceSynchronize();
    }

    timer.stop_timer();

    float gpu_iterate_time_si = timer.timings.back();
    //copy data back from gpu
    dense_patch_output.copy_data_to_host();

    std::cout << gpu_iterate_time_si/(number_reps*1.0f) << std::endl;

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
        if(dense_patch_output[aprIt]==apr.particles_intensities[aprIt]){
            c_pass++;
        } else {
            c_fail++;
            success = false;
            if(c_fail < 5) {
                std::cout << dense_patch_output[aprIt] << " Expected: " <<  apr.particles_intensities[aprIt] << " Level: " << aprIt.level() << " y: " << aprIt.y() <<std::endl;
            }
        }
    }

    if(success){
        std::cout << "Fill Dense Check, PASS" << std::endl;
    } else {
        std::cout << "Fill Dense Check, FAIL Total: " << c_fail << " Pass Total:  " << c_pass << std::endl;
    }







}


__device__ std::size_t compute_row_index(
        const std::uint16_t _x,
        const std::uint16_t _z,
        const std::uint8_t _level,
        const std::size_t* level_offsets,
        const std::uint16_t* level_y_num,
        const std::uint16_t* level_x_num,
        const std::uint16_t* level_z_num
){
    std::size_t level_zx_offset = level_offsets[_level] + level_x_num[_level] * _z + _x;

    return level_zx_offset;


}



__global__ void update_dense_insert(
        const thrust::tuple<std::size_t,std::size_t>* row_info,
        const std::size_t*  _chunk_index_end,
        std::size_t total_number_chunks,
        const std::uint16_t* particle_y,
        const std::size_t* level_offsets,
        const std::uint16_t* level_y_num,
        const std::uint16_t* level_x_num,
        const std::uint16_t* level_z_num,
        std::uint16_t* particles_input,
        std::uint16_t* particles_output)
{


    int chunk_index = blockDim.x * blockIdx.x + threadIdx.x; // the input to each kernel is its chunk index for which it should iterate over

    if(chunk_index >=total_number_chunks){
        return; //out of bounds
    }

    std::uint16_t local_row[120] ={0};



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


            std::uint16_t x;
            std::uint16_t z;
            std::uint8_t level;

            //decode the key
            x = (current_row_key & KEY_X_MASK) >> KEY_X_SHIFT;
            z = (current_row_key & KEY_Z_MASK) >> KEY_Z_SHIFT;
            level = (current_row_key & KEY_LEVEL_MASK) >> KEY_LEVEL_SHIFT;

            /*
             * Need to initiazlie the update structures
             */

            for (int z_d = -1; z_d < 2; ++z_d) {
                for (int x_d = -1; x_d < 2; ++x_d) {

                    if(((x+x_d) >=0) && ((x+x_d) < level_x_num[level])){
                        if(((z+z_d) >=0) && (z+z_d< level_z_num[level])) {

                            std::size_t row_index = compute_row_index(x+x_d,z+z_d,level,level_offsets,level_y_num,level_x_num,level_z_num);
                            std::size_t global_end = thrust::get<1>(row_info[row_index]);
                            std::size_t global_begin = 0;

                            if(row_index>0) {
                                global_begin = thrust::get<1>(row_info[row_index-1]);
                            } else {
                                global_begin=0;
                            }

                            for (std::size_t particle_global_index = global_begin; particle_global_index < global_end; ++particle_global_index) {
                                uint16_t current_y = particle_y[particle_global_index];

                                local_row[current_y]+=(1.0/10.0f)*particles_input[particle_global_index];

                            }


                        } else {
                            //this is out of bounds, section --> would need to be updated to handle boundary conditions

                        }
                    }

                }
            }


            //Particle Row Loop
            particle_global_index_end = thrust::get<1>(row_info[current_row]);

            if (current_row == 0) {
                particle_global_index_begin = 0;
            } else {
                particle_global_index_begin = thrust::get<1>(row_info[current_row-1]);
            }

            //loop over the particles in the row
            for (std::size_t particle_global_index = particle_global_index_begin; particle_global_index < particle_global_index_end; ++particle_global_index) {
                uint16_t current_y = particle_y[particle_global_index];


                //local_patch[1][1][current_y%3] = particles_input[particle_global_index];

                float local_sum = 0;


                particles_output[particle_global_index] = local_row[current_y];
                //particles_output[particle_global_index] = particles_input[particle_global_index];

            }

        }

    }


}




__global__ void update_dense_patch(
        const thrust::tuple<std::size_t,std::size_t>* row_info,
        const std::size_t*  _chunk_index_end,
        std::size_t total_number_chunks,
        const std::uint16_t* particle_y,
        const std::size_t* level_offsets,
        const std::uint16_t* level_y_num,
        const std::uint16_t* level_x_num,
        const std::uint16_t* level_z_num,
        std::uint16_t* particles_input,
        std::uint16_t* particles_output)
{


    int chunk_index = blockDim.x * blockIdx.x + threadIdx.x; // the input to each kernel is its chunk index for which it should iterate over

    if(chunk_index >=total_number_chunks){
        return; //out of bounds
    }

    std::uint16_t local_patch[3][3][3] ={0};

    std::size_t global_end[3][3]={0};
    std::size_t global_index[3][3]={0};


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


            std::uint16_t x;
            std::uint16_t z;
            std::uint8_t level;

            //decode the key
            x = (current_row_key & KEY_X_MASK) >> KEY_X_SHIFT;
            z = (current_row_key & KEY_Z_MASK) >> KEY_Z_SHIFT;
            level = (current_row_key & KEY_LEVEL_MASK) >> KEY_LEVEL_SHIFT;

            /*
             * Need to initiazlie the update structures
             */

            for (int z_d = -1; z_d < 2; ++z_d) {
                for (int x_d = -1; x_d < 2; ++x_d) {

                    if(((x+x_d) >=0) && ((x+x_d) < level_x_num[level])){
                        if(((z+z_d) >=0) && (z+z_d< level_z_num[level])) {

                            std::size_t row_index = compute_row_index(x+x_d,z+z_d,level,level_offsets,level_y_num,level_x_num,level_z_num);
                            global_end[z_d+1][x_d+1] = thrust::get<1>(row_info[row_index]);
                            if(row_index>0) {
                                global_index[z_d + 1][x_d + 1] = thrust::get<1>(row_info[row_index-1]);
                            } else {
                                global_index[z_d + 1][x_d + 1]=0;
                            }


                        } else {
                            //this is out of bounds, section --> would need to be updated to handle boundary conditions
                            global_end[z_d+1][x_d+1]=0;
                            global_index[z_d+1][x_d+1]=1;
                        }
                    }

                }
            }


            //Particle Row Loop
            particle_global_index_end = thrust::get<1>(row_info[current_row]);

            if (current_row == 0) {
                particle_global_index_begin = 0;
            } else {
                particle_global_index_begin = thrust::get<1>(row_info[current_row-1]);
            }

            //initialize first value
            for (int z_d = 0; z_d < 3; ++z_d) {
                for (int x_d = 0; x_d < 3; ++x_d) {
                    local_patch[z_d][x_d][particle_y[global_index[z_d][x_d]]%3] = particles_input[global_index[z_d][x_d]];
                }
            }


            //loop over the particles in the row
            for (std::size_t particle_global_index = particle_global_index_begin; particle_global_index < particle_global_index_end; ++particle_global_index) {
                uint16_t current_y = particle_y[particle_global_index];

                //update patch
                for (int z_d = 0; z_d < 3; ++z_d) {
                    for (int x_d = 0; x_d < 3; ++x_d) {


                        //iterates over and updates the local patch
                        while((global_index[z_d][x_d]+1) < (global_end[z_d+1][x_d+1]) && (particle_y[global_index[z_d][x_d]+1] <= (current_y+1))){
                            global_index[z_d][x_d]++;
                            local_patch[z_d][x_d][particle_y[global_index[z_d][x_d]]%3] = particles_input[global_index[z_d][x_d]];
                        }
                        //local_patch[z_d][x_d][current_y%3] = particles_input[global_index[z_d][x_d]];
                        //local_patch[1][1][1] = particles_input[particle_global_index];
                    }
                }

                //local_patch[1][1][current_y%3] = particles_input[particle_global_index];

                float local_sum = 0;

                for (int z_d = 0; z_d < 3; ++z_d) {
                    for (int x_d = 0; x_d < 3; ++x_d) {
                        for (int y_d = 0; y_d < 3; ++y_d) {
                            local_sum += local_patch[z_d][x_d][y_d];
                        }
                    }
                }


                particles_output[particle_global_index] = local_sum;
                //particles_output[particle_global_index] = particles_input[particle_global_index];

            }

        }

    }


}






