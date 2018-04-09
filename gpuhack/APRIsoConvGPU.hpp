//
// Created by cheesema on 09.04.18.
//

#ifndef LIBAPR_APRGPUISOCONV_HPP
#define LIBAPR_APRGPUISOCONV_HPP

#define LOCALPATCHCONV333(particle_output,index,z,x,y,neighbour_sum)\
neighbour_sum=0;\
if (not_ghost) {\
    for (int q = 0; q < 3; ++q) {\
        neighbour_sum += local_patch[z + q - 1][x + 0 - 1][(y+N)%N]\
                 + local_patch[z + q - 1][x + 0 - 1][(y+N-1)%N]\
                 + local_patch[z + q - 1][x + 0 - 1][(y+N+1)%N]\
                 + local_patch[z + q - 1][x + 1 - 1][(y+N)%N]\
                 + local_patch[z + q - 1][x + 1 - 1][(y+N-1)%N]\
                 + local_patch[z + q - 1][x + 1 - 1][(y+N+1)%N]\
                 + local_patch[z + q - 1][x + 2 - 1][(y+N)%N]\
                 + local_patch[z + q - 1][x + 2 - 1][(y+N-1)%N]\
                 + local_patch[z + q - 1][x + 2 - 1][(y+N+1)%N];\
    }\
    particle_output[index] = std::roundf(neighbour_sum / 27.0f);\
}\

#define LOCALPATCHCONV555(particle_output,index,z,x,y,neighbour_sum)\
neighbour_sum=0;\
if (not_ghost) {\
for (int q = 0; q < 5; ++q) {\
            neighbour_sum +=\
                local_patch[z + q - 2][x + 0 - 2][(y+N-2)%N]\
                 + local_patch[z + q - 2][x + 0 - 2][(y+N-1)%N]\
                 + local_patch[z + q - 2][x + 0 - 2][(y+N)%N]\
                 + local_patch[z + q - 2][x + 0 - 2][(y+N+1)%N]\
                 + local_patch[z + q - 2][x + 0 - 2][(y+N+2)%N]\
                 + local_patch[z + q - 2][x + 1 - 2][(y+N-2)%N]\
                 + local_patch[z + q - 2][x + 1 - 2][(y+N-1)%N]\
                 + local_patch[z + q - 2][x + 1 - 2][(y+N)%N]\
                 + local_patch[z + q - 2][x + 1 - 2][(y+N+1)%N]\
                 + local_patch[z + q - 2][x + 1 - 2][(y+N+2)%N]\
                + local_patch[z + q - 2][x + 2 - 2][(y+N-2)%N]\
                 + local_patch[z + q - 2][x + 2 - 2][(y+N-1)%N]\
                 + local_patch[z + q - 2][x + 2 - 2][(y+N)%N]\
                 + local_patch[z + q - 2][x + 2 - 2][(y+N+1)%N]\
                 + local_patch[z + q - 2][x + 2 - 2][(y+N+2)%N]\
                + local_patch[z + q - 2][x + 3 - 2][(y+N-2)%N]\
                 + local_patch[z + q - 2][x + 3 - 2][(y+N-1)%N]\
                 + local_patch[z + q - 2][x + 3 - 2][(y+N)%N]\
                 + local_patch[z + q - 2][x + 3 - 2][(y+N+1)%N]\
                 + local_patch[z + q - 2][x + 3 - 2][(y+N+2)%N]\
                + local_patch[z + q - 2][x + 4 - 2][(y+N-2)%N]\
                 + local_patch[z + q - 2][x + 4 - 2][(y+N-1)%N]\
                 + local_patch[z + q - 2][x + 4 - 2][(y+N)%N]\
                 + local_patch[z + q - 2][x + 4 - 2][(y+N+1)%N]\
                 + local_patch[z + q - 2][x + 4 - 2][(y+N+2)%N];\
}\
particle_output[index] = std::roundf(neighbour_sum / 125.0f);\
}\

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/APRTree.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/ExtraParticleData.hpp"
#include "misc/APRTimer.hpp"

#include "thrust/device_vector.h"
#include "thrust/tuple.h"
#include "thrust/copy.h"

#include "GPUAPRAccess.hpp"
#include "../src/data_structures/APR/APR.hpp"
#include "APRDownsampleGPU.hpp"






__global__ void conv_max_333(const std::size_t *row_info,
                             const std::uint16_t *particle_y,
                             const std::uint16_t *particle_data_input,
                             std::uint16_t *particle_data_output,
                             const std::size_t *level_offset,
                             const std::uint16_t *level_x_num,
                             const std::uint16_t *level_z_num,
                             const std::uint16_t *level_y_num,
                             const std::size_t level);
template<typename treeType>
__global__ void conv_interior_333(const std::size_t *row_info,
                                  const std::uint16_t *particle_y,
                                  const std::size_t *level_offset,
                                  const std::uint16_t *particle_data_input,
                                  const std::size_t *row_info_child,
                                  const std::uint16_t *particle_y_child,
                                  const std::size_t *level_offset_child,
                                  const treeType *particle_data_input_child,
                                  std::uint16_t *particle_data_output,
                                  const std::uint16_t *level_x_num,
                                  const std::uint16_t *level_z_num,
                                  const std::uint16_t *level_y_num,
                                  const std::size_t level);

template<typename treeType>
__global__ void conv_min_333(const std::size_t *row_info,
                             const std::uint16_t *particle_y,
                             const std::size_t *level_offset,
                             const std::uint16_t *particle_data_input,
                             const std::size_t *row_info_child,
                             const std::uint16_t *particle_y_child,
                             const std::size_t *level_offset_child,
                             const treeType *particle_data_input_child,
                             std::uint16_t *particle_data_output,
                             const std::uint16_t *level_x_num,
                             const std::uint16_t *level_z_num,
                             const std::uint16_t *level_y_num,
                             const std::size_t level);

__global__ void conv_max_555(const std::size_t *row_info,
                             const std::uint16_t *particle_y,
                             const std::uint16_t *particle_data_input,
                             std::uint16_t *particle_data_output,
                             const std::size_t *level_offset,
                             const std::uint16_t *level_x_num,
                             const std::uint16_t *level_z_num,
                             const std::uint16_t *level_y_num,
                             const std::size_t level);

__global__ void conv_interior_555(const std::size_t *row_info,
                                  const std::uint16_t *particle_y,
                                  const std::size_t *level_offset,
                                  const std::uint16_t *particle_data_input,
                                  const std::size_t *row_info_child,
                                  const std::uint16_t *particle_y_child,
                                  const std::size_t *level_offset_child,
                                  const std::float_t *particle_data_input_child,
                                  std::uint16_t *particle_data_output,
                                  const std::uint16_t *level_x_num,
                                  const std::uint16_t *level_z_num,
                                  const std::uint16_t *level_y_num,
                                  const std::size_t level);

__global__ void conv_min_555(const std::size_t *row_info,
                             const std::uint16_t *particle_y,
                             const std::size_t *level_offset,
                             const std::uint16_t *particle_data_input,
                             const std::size_t *row_info_child,
                             const std::uint16_t *particle_y_child,
                             const std::size_t *level_offset_child,
                             const std::float_t *particle_data_input_child,
                             std::uint16_t *particle_data_output,
                             const std::uint16_t *level_x_num,
                             const std::uint16_t *level_z_num,
                             const std::uint16_t *level_y_num,
                             const std::size_t level);

class APRIsoConvGPU {

public:
    GPUAPRAccess gpuaprAccess;
    GPUAPRAccess gpuaprAccessTree;

    unsigned int number_particles;
    unsigned int number_interior_particle_cells;

    template<typename T>
    APRIsoConvGPU(APR<T>& apr,APRTree<T>& aprTree){

        //intiitalize the access data on the GPU
        APRIterator<T> aprIterator(apr);
        gpuaprAccess.initialize_gpu_access_alternate(aprIterator);

        number_particles = aprIterator.total_number_particles();

        APRTreeIterator<T> aprTreeIterator(aprTree);
        gpuaprAccessTree.initialize_gpu_access_alternate(aprTreeIterator);
        number_interior_particle_cells = aprTreeIterator.total_number_particles();
    }

    template<typename T>
    void isotropic_convolve_333(APR<T>& apr,ExtraParticleData<uint16_t>& input_particles,
                                ExtraParticleData<uint16_t>& output_particles,
                                std::vector<float>& conv_stencil,
                                ExtraParticleData<uint16_t>& tree_temp){
        /*
         *  Perform APR Isotropic Convolution Operation on the GPU with a 3x3x3 kernel
         *
         */

        APRIterator<uint16_t> aprIt(apr);

        if(tree_temp.gpu_data.size()!=number_interior_particle_cells) {
            tree_temp.init_gpu(number_interior_particle_cells);
        }

        /*
        *  Test the x,y,z,level information is correct
        *
        */
        if(input_particles.gpu_data.size()!=number_particles) {
            input_particles.copy_data_to_gpu();
        }

        if(output_particles.gpu_data.size()!=number_particles) {
            output_particles.init_gpu(number_particles);
        }

        cudaDeviceSynchronize();

        for (int level = apr.level_max(); level > aprIt.level_min(); --level) {

            std::size_t number_rows_l = apr.spatial_index_x_max(level) * apr.spatial_index_z_max(level);
            std::size_t offset = gpuaprAccess.h_level_offset[level];

            std::size_t x_num = apr.spatial_index_x_max(level);
            std::size_t z_num = apr.spatial_index_z_max(level);
            std::size_t y_num = apr.spatial_index_y_max(level);

            dim3 threads_l(128, 1, 1);

            int x_blocks = (x_num + 2 - 1) / 2;
            int z_blocks = (z_num + 2 - 1) / 2;

            dim3 blocks_l(x_blocks, 1, z_blocks);

            if(level==apr.level_max()) {

                down_sample_avg <<< blocks_l, threads_l >>>
                                               (gpuaprAccess.gpu_access.row_global_index,
                                                       gpuaprAccess.gpu_access.y_part_coord,
                                                       gpuaprAccess.gpu_access.level_offsets,
                                                       input_particles.gpu_pointer,
                                                       gpuaprAccessTree.gpu_access.row_global_index,
                                                       gpuaprAccessTree.gpu_access.y_part_coord,
                                                       gpuaprAccessTree.gpu_access.level_offsets,
                                                       tree_temp.gpu_pointer,
                                                       gpuaprAccess.gpu_access.level_x_num,
                                                       gpuaprAccess.gpu_access.level_z_num,
                                                       gpuaprAccess.gpu_access.level_y_num,
                                                       level);


            } else {

                down_sample_avg_interior<< < blocks_l, threads_l >> >
                                                       (gpuaprAccess.gpu_access.row_global_index,
                                                               gpuaprAccess.gpu_access.y_part_coord,
                                                               gpuaprAccess.gpu_access.level_offsets,
                                                               input_particles.gpu_pointer,
                                                               gpuaprAccessTree.gpu_access.row_global_index,
                                                               gpuaprAccessTree.gpu_access.y_part_coord,
                                                               gpuaprAccessTree.gpu_access.level_offsets,
                                                               tree_temp.gpu_pointer,
                                                               gpuaprAccess.gpu_access.level_x_num,
                                                               gpuaprAccess.gpu_access.level_z_num,
                                                               gpuaprAccess.gpu_access.level_y_num,
                                                               level);
            }
        }

        cudaDeviceSynchronize();

        for (int level = apr.level_max(); level >= aprIt.level_min(); --level) {

            std::size_t number_rows_l = apr.spatial_index_x_max(level) * apr.spatial_index_z_max(level);
            std::size_t offset = gpuaprAccess.h_level_offset[level];

            std::size_t x_num = apr.spatial_index_x_max(level);
            std::size_t z_num = apr.spatial_index_z_max(level);
            std::size_t y_num = apr.spatial_index_y_max(level);

            dim3 threads_l(10, 1, 10);

            int x_blocks = (x_num + 8 - 1) / 8;
            int z_blocks = (z_num + 8 - 1) / 8;

            dim3 blocks_l(x_blocks, 1, z_blocks);

            if (level == apr.level_min()) {
                conv_min_333 << < blocks_l, threads_l >> >
                                            (gpuaprAccess.gpu_access.row_global_index,
                                                    gpuaprAccess.gpu_access.y_part_coord,
                                                    gpuaprAccess.gpu_access.level_offsets,
                                                    input_particles.gpu_pointer,
                                                    gpuaprAccessTree.gpu_access.row_global_index,
                                                    gpuaprAccessTree.gpu_access.y_part_coord,
                                                    gpuaprAccessTree.gpu_access.level_offsets,
                                                    tree_temp.gpu_pointer,
                                                    output_particles.gpu_pointer,
                                                    gpuaprAccess.gpu_access.level_x_num,
                                                    gpuaprAccess.gpu_access.level_z_num,
                                                    gpuaprAccess.gpu_access.level_y_num,
                                                    level);

            } else if (level == apr.level_max()) {
                conv_max_333 << < blocks_l, threads_l >> >
                                            (gpuaprAccess.gpu_access.row_global_index,
                                                    gpuaprAccess.gpu_access.y_part_coord,
                                                    input_particles.gpu_pointer,
                                                    output_particles.gpu_pointer,
                                                    gpuaprAccess.gpu_access.level_offsets,
                                                    gpuaprAccess.gpu_access.level_x_num,
                                                    gpuaprAccess.gpu_access.level_z_num,
                                                    gpuaprAccess.gpu_access.level_y_num,
                                                    level);

            } else {
                conv_interior_333 << < blocks_l, threads_l >> >
                                                 (gpuaprAccess.gpu_access.row_global_index,
                                                         gpuaprAccess.gpu_access.y_part_coord,
                                                         gpuaprAccess.gpu_access.level_offsets,
                                                         apr.particles_intensities.gpu_pointer,
                                                         gpuaprAccessTree.gpu_access.row_global_index,
                                                         gpuaprAccessTree.gpu_access.y_part_coord,
                                                         gpuaprAccessTree.gpu_access.level_offsets,
                                                         tree_temp.gpu_pointer,
                                                         output_particles.gpu_pointer,
                                                         gpuaprAccess.gpu_access.level_x_num,
                                                         gpuaprAccess.gpu_access.level_z_num,
                                                         gpuaprAccess.gpu_access.level_y_num,
                                                         level);
            }
            cudaDeviceSynchronize();

        }

        cudaDeviceSynchronize();



    }





};
__global__ void conv_max_333(const std::size_t *row_info,
                             const std::uint16_t *particle_y,
                             const std::uint16_t *particle_data_input,
                             std::uint16_t *particle_data_output,
                             const std::size_t *level_offset,
                             const std::uint16_t *level_x_num,
                             const std::uint16_t *level_z_num,
                             const std::uint16_t *level_y_num,
                             const std::size_t level) {

    /*
     *
     *  Here we introduce updating Particle Cells at a level below.
     *
     */

    const int x_num = level_x_num[level];

    const int z_num = level_z_num[level];

    const int x_num_p = level_x_num[level - 1];
    const int y_num_p = level_y_num[level - 1];
    const int z_num_p = level_z_num[level - 1];



    // This is block wise shared memory this is assuming an 8*8 block with pad()


    if (threadIdx.x >= 10) {
        return;
    }
    if (threadIdx.z >= 10) {
        return;
    }


    bool not_ghost = false;

    if ((threadIdx.x > 0) && (threadIdx.x < 9) && (threadIdx.z > 0) && (threadIdx.z < 9)) {
        not_ghost = true;
    }

    int x_index = (8 * blockIdx.x + threadIdx.x - 1);
    int z_index = (8 * blockIdx.z + threadIdx.z - 1);

    const unsigned int N = 4;
    __shared__
    std::uint16_t local_patch[10][10][6];

    if ((x_index >= x_num) || (x_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        return; //out of bounds
    }


    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    std::size_t particle_global_index_begin_p;
    std::size_t particle_global_index_end_p;

    // current level
    std::size_t current_row = level_offset[level] + (x_index) + (z_index) *
                                                                x_num; // the input to each kernel is its chunk index for which it should iterate over
    get_row_begin_end(&particle_global_index_begin, &particle_global_index_end, current_row, row_info);
    std::size_t particle_index_l = particle_global_index_begin;
    std::uint16_t y_l = particle_y[particle_index_l];
    std::uint16_t f_l = particle_data_input[particle_index_l];

    int x_index_p = (x_index) / 2;
    int z_index_p = (z_index) / 2;
    std::size_t current_row_p = level_offset[level - 1] + (x_index_p) + (z_index_p) *
                                                                        x_num_p; // the input to each kernel is its chunk index for which it should iterate over
    // parent level, level - 1, one resolution lower (coarser)
    get_row_begin_end(&particle_global_index_begin_p, &particle_global_index_end_p, current_row_p, row_info);

    //parent level variables
    std::size_t particle_index_p = particle_global_index_begin_p;
    std::uint16_t y_p = particle_y[particle_index_p];
    std::uint16_t f_p = particle_data_input[particle_index_p];



    //current level variables


    const int y_num = level_y_num[level];
    if (particle_global_index_begin_p == particle_global_index_end_p) {
        y_p = y_num + 1;//no particles don't do anything
    }

    if (particle_global_index_begin == particle_global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }


    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)

    const int filter_offset = 1;

    __shared__
    std::uint16_t y_update_flag[10][10][2];
    __shared__
    std::uint16_t y_update_index[10][10][2];

    y_update_flag[threadIdx.z][threadIdx.x][0] = 0;
    y_update_flag[threadIdx.z][threadIdx.x][1] = 0;


    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T
        __syncthreads();
        //Check if its time to update the parent level


        if (j == (2 * y_p)) {
            local_patch[threadIdx.z][threadIdx.x][(j) % N] = f_p; //initial update
            local_patch[threadIdx.z][threadIdx.x][(j + 1) % N] = f_p;
        }


        //Check if its time to update current level
        if (j == y_l) {
            local_patch[threadIdx.z][threadIdx.x][j % N] = f_l; //initial update
            y_update_flag[threadIdx.z][threadIdx.x][j % 2] = 1;
            y_update_index[threadIdx.z][threadIdx.x][j % 2] = particle_index_l - particle_global_index_begin;
        } else {
            y_update_flag[threadIdx.z][threadIdx.x][j % 2] = 0;
        }

        //update at current level
        if ((y_l <= j) && ((particle_index_l + 1) < particle_global_index_end)) {
            particle_index_l++;
            y_l = particle_y[particle_index_l];
            f_l = particle_data_input[particle_index_l];
        }

        //parent update loop
        if ((2 * y_p <= j) && ((particle_index_p + 1) < particle_global_index_end_p)) {
            particle_index_p++;
            y_p = particle_y[particle_index_p];
            f_p = particle_data_input[particle_index_p];
        }

        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if (y_update_flag[threadIdx.z][threadIdx.x][(j - filter_offset + 2) % 2] == 1) {

            float neighbour_sum = 0;
            LOCALPATCHCONV333(particle_data_output, particle_global_index_begin +
                                                    y_update_index[threadIdx.z][threadIdx.x][
                                                            (j + 2 - filter_offset) % 2], threadIdx.z, threadIdx.x,
                              j - 1, neighbour_sum);

        }

    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    if (y_update_flag[threadIdx.z][threadIdx.x][(y_num - 1) % 2] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;
        LOCALPATCHCONV333(particle_data_output, particle_index_l, threadIdx.z, threadIdx.x, y_num - 1,
                          neighbour_sum);


    }


}
template<typename treeType>
__global__ void conv_interior_333(const std::size_t *row_info,
                                  const std::uint16_t *particle_y,
                                  const std::size_t *level_offset,
                                  const std::uint16_t *particle_data_input,
                                  const std::size_t *row_info_child,
                                  const std::uint16_t *particle_y_child,
                                  const std::size_t *level_offset_child,
                                  const treeType *particle_data_input_child,
                                  std::uint16_t *particle_data_output,
                                  const std::uint16_t *level_x_num,
                                  const std::uint16_t *level_z_num,
                                  const std::uint16_t *level_y_num,
                                  const std::size_t level) {
    /*
     *
     *  Here we update both those Particle Cells at a level below and above.
     *
     */

    const int x_num = level_x_num[level];

    const int z_num = level_z_num[level];

    const int x_num_p = level_x_num[level - 1];
    const int y_num_p = level_y_num[level - 1];
    const int z_num_p = level_z_num[level - 1];

    const unsigned int N = 4;


    if (threadIdx.x >= 10) {
        return;
    }
    if (threadIdx.z >= 10) {
        return;
    }

    int x_index = (8 * blockIdx.x + threadIdx.x - 1);
    int z_index = (8 * blockIdx.z + threadIdx.z - 1);


    bool not_ghost = false;

    if ((threadIdx.x > 0) && (threadIdx.x < 9) && (threadIdx.z > 0) && (threadIdx.z < 9)) {
        not_ghost = true;
    }

    __shared__
    std::uint16_t local_patch[10][10][6]; // This is block wise shared memory this is assuming an 8*8 block with pad()
    if ((x_index >= x_num) || (x_index < 0)) {
        //set the whole buffer to the boundary condition
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        //set the whole buffer to the zero boundary condition
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        return; //out of bounds
    }

    int x_index_p = (8 * blockIdx.x + threadIdx.x - 1) / 2;
    int z_index_p = (8 * blockIdx.z + threadIdx.z - 1) / 2;


    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    std::size_t particle_global_index_begin_p;
    std::size_t particle_global_index_end_p;

    /*
    * Current level variable initialization,
    */

    // current level
    std::size_t current_row = level_offset[level] + (x_index) + (z_index) *
                                                                x_num; // the input to each kernel is its chunk index for which it should iterate over
    get_row_begin_end(&particle_global_index_begin, &particle_global_index_end, current_row, row_info);

    std::size_t particle_index_l = particle_global_index_begin;
    std::uint16_t y_l = particle_y[particle_index_l];
    //std::uint16_t f_l = particle_data_input[particle_index_l];

    // parent level, level - 1, one resolution lower (coarser)
    std::size_t current_row_p = level_offset[level - 1] + (x_index_p) + (z_index_p) *
                                                                        x_num_p; // the input to each kernel is its chunk index for which it should iterate over
    get_row_begin_end(&particle_global_index_begin_p, &particle_global_index_end_p, current_row_p, row_info);


    //parent level variables
    std::size_t particle_index_p = particle_global_index_begin_p;
    std::uint16_t y_p = particle_y[particle_index_p];
    std::uint16_t f_p = particle_data_input[particle_index_p];


    /*
    * Child level variable initialization, using 'Tree'
    * This is the same row as the current level
    */

    std::size_t current_row_child = level_offset_child[level] + (x_index) + (z_index) *
                                                                            x_num; // the input to each kernel is its chunk index for which it should iterate over

    std::size_t particle_global_index_begin_child;
    std::size_t particle_global_index_end_child;

    get_row_begin_end(&particle_global_index_begin_child, &particle_global_index_end_child, current_row_child,
                      row_info_child);

    std::size_t particle_index_child = particle_global_index_begin_child;
    std::uint16_t y_child = particle_y_child[particle_index_child];
    std::uint16_t f_child = particle_data_input_child[particle_index_child];


    const int y_num = level_y_num[level];

    if (particle_global_index_begin_child == particle_global_index_end_child) {
        y_child = y_num + 1;//no particles don't do anything
    }

    if (particle_global_index_begin_p == particle_global_index_end_p) {
        y_p = y_num + 1;//no particles don't do anything
    }

    if (particle_global_index_begin == particle_global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }

    __shared__
    std::uint16_t y_update_flag[10][10][2];
    __shared__
    std::uint16_t y_update_index[10][10][2];

    __shared__
    std::uint16_t f_l[10][10];
    f_l[threadIdx.z][threadIdx.x] = particle_data_input[particle_index_l];

    y_update_flag[threadIdx.z][threadIdx.x][0] = 0;
    y_update_flag[threadIdx.z][threadIdx.x][1] = 0;

    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)

    const int filter_offset = 1;


    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T

        //Check if its time to update the parent level
        if (j == (2 * y_p)) {
            local_patch[threadIdx.z][threadIdx.x][(j) % N] = f_p; //initial update
            local_patch[threadIdx.z][threadIdx.x][(j + 1) % N] = f_p;
        }

        //Check if its time to update child level
        if (j == y_child) {
            local_patch[threadIdx.z][threadIdx.x][y_child % N] = f_child; //initial update
        }

        //Check if its time to update current level
        if (j == y_l) {
            local_patch[threadIdx.z][threadIdx.x][y_l % N] = f_l[threadIdx.z][threadIdx.x]; //initial update
            y_update_flag[threadIdx.z][threadIdx.x][j % 2] = 1;
            y_update_index[threadIdx.z][threadIdx.x][j % 2] = particle_index_l - particle_global_index_begin;
        } else {
            y_update_flag[threadIdx.z][threadIdx.x][j % 2] = 0;
        }


        //update at current level
        if ((y_l <= j) && ((particle_index_l + 1) < particle_global_index_end)) {
            particle_index_l++;
            y_l = particle_y[particle_index_l];
            f_l[threadIdx.z][threadIdx.x] = particle_data_input[particle_index_l];
        }

        //parent update loop
        if ((2 * y_p <= j) && ((particle_index_p + 1) < particle_global_index_end_p)) {
            particle_index_p++;
            y_p = particle_y[particle_index_p];
            f_p = particle_data_input[particle_index_p];
        }


        //update at child level
        if ((y_child <= j) && ((particle_index_child + 1) < particle_global_index_end_child)) {
            particle_index_child++;
            y_child = particle_y_child[particle_index_child];
            f_child = particle_data_input_child[particle_index_child];
        }


        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if (y_update_flag[threadIdx.z][threadIdx.x][(j - filter_offset + 2) % 2] == 1) {
            float neighbour_sum = 0;

            LOCALPATCHCONV333(particle_data_output, particle_global_index_begin +
                                                    y_update_index[threadIdx.z][threadIdx.x][
                                                            (j + 2 - filter_offset) % 2], threadIdx.z, threadIdx.x,
                              j - 1, neighbour_sum);
        }
        __syncthreads();

    }

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();
    //set the boundary condition (zeros in this case)

    if (y_update_flag[threadIdx.z][threadIdx.x][(y_num - 1) % 2] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;

        LOCALPATCHCONV333(particle_data_output, particle_index_l, threadIdx.z, threadIdx.x, y_num - 1,
                          neighbour_sum);

    }


}

template<typename treeType>
__global__ void conv_min_333(const std::size_t *row_info,
                             const std::uint16_t *particle_y,
                             const std::size_t *level_offset,
                             const std::uint16_t *particle_data_input,
                             const std::size_t *row_info_child,
                             const std::uint16_t *particle_y_child,
                             const std::size_t *level_offset_child,
                             const treeType *particle_data_input_child,
                             std::uint16_t *particle_data_output,
                             const std::uint16_t *level_x_num,
                             const std::uint16_t *level_z_num,
                             const std::uint16_t *level_y_num,
                             const std::size_t level) {

    /*
     *
     *  Here we introduce updating Particle Cells at a level below.
     *
     */

    const int x_num = level_x_num[level];
    const int y_num = level_y_num[level];
    const int z_num = level_z_num[level];

    const unsigned int N = 4;

    __shared__
    std::float_t local_patch[10][10][4]; // This is block wise shared memory this is assuming an 8*8 block with pad()

    uint16_t y_cache[N] = {0}; // These are local register/private caches
    uint16_t index_cache[N] = {0}; // These are local register/private caches


    if (threadIdx.x >= 10) {
        return;
    }
    if (threadIdx.z >= 10) {
        return;
    }


    int x_index = (8 * blockIdx.x + threadIdx.x - 1);
    int z_index = (8 * blockIdx.z + threadIdx.z - 1);


    bool not_ghost = false;

    if ((threadIdx.x > 0) && (threadIdx.x < 9) && (threadIdx.z > 0) && (threadIdx.z < 9)) {
        not_ghost = true;
    }


    if ((x_index >= x_num) || (x_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;

        return; //out of bounds
    }


    /*
     * Current level variable initialization
     *
     */

    std::size_t current_row = level_offset[level] + (x_index) + (z_index) *
                                                                x_num; // the input to each kernel is its chunk index for which it should iterate over
    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    // current level
    get_row_begin_end(&particle_global_index_begin, &particle_global_index_end, current_row, row_info);

    std::size_t y_block = 1;
    std::uint16_t y_update_flag[2] = {0};
    std::size_t y_update_index[2] = {0};

    //current level variables
    std::size_t particle_index_l = particle_global_index_begin;
    std::uint16_t y_l = particle_y[particle_index_l];
    std::uint16_t f_l = particle_data_input[particle_index_l];

    /*
    * Child level variable initialization, using 'Tree'
    * This is the same row as the current level
    */

    std::size_t current_row_child = level_offset_child[level] + (x_index) + (z_index) *
                                                                            x_num; // the input to each kernel is its chunk index for which it should iterate over

    std::size_t particle_global_index_begin_child;
    std::size_t particle_global_index_end_child;

    get_row_begin_end(&particle_global_index_begin_child, &particle_global_index_end_child, current_row_child,
                      row_info_child);

    std::size_t particle_index_child = particle_global_index_begin_child;
    std::uint16_t y_child = particle_y_child[particle_index_child];
    std::float_t f_child = particle_data_input_child[particle_index_child];


    if (particle_global_index_begin_child == particle_global_index_end_child) {
        y_child = y_num + 1;//no particles don't do anything
    }

    if (particle_global_index_begin == particle_global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }


    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)

    const int filter_offset = 1;


    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T

        /*
         *
         * Current Level Update
         *
         */

        __syncthreads();

        //Check if its time to update current level
        if (j == y_l) {
            local_patch[threadIdx.z][threadIdx.x][y_l % N] = f_l; //initial update
            y_update_flag[j % 2] = 1;
            y_update_index[j % 2] = particle_index_l;
        } else {
            y_update_flag[j % 2] = 0;
        }

        //update at current level
        if ((y_l <= j) && ((particle_index_l + 1) < particle_global_index_end)) {
            particle_index_l++;
            y_l = particle_y[particle_index_l];
            f_l = particle_data_input[particle_index_l];
        }

        /*
         *
         * Child Level Update
         *
         */


        //Check if its time to update current level
        if (j == y_child) {
            local_patch[threadIdx.z][threadIdx.x][y_child % N] = f_child; //initial update
        }

        //update at current level
        if ((y_child <= j) && ((particle_index_child + 1) < particle_global_index_end_child)) {
            particle_index_child++;
            y_child = particle_y_child[particle_index_child];
            f_child = particle_data_input_child[particle_index_child];
        }


        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if (y_update_flag[(j - filter_offset + 2) % 2] == 1) {
            float neighbour_sum = 0;

            LOCALPATCHCONV333(particle_data_output, y_update_index[(j + 2 - filter_offset) % 2], threadIdx.z,
                              threadIdx.x, j - 1, neighbour_sum);
        }

    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num - 1) % 2] == 1) { //the last particle (if it exists)

        float neighbour_sum = 0;

        LOCALPATCHCONV333(particle_data_output, particle_index_l, threadIdx.z, threadIdx.x, y_num - 1,
                          neighbour_sum);
    }

}


__global__ void conv_max_555(const std::size_t *row_info,
                             const std::uint16_t *particle_y,
                             const std::uint16_t *particle_data_input,
                             std::uint16_t *particle_data_output,
                             const std::size_t *level_offset,
                             const std::uint16_t *level_x_num,
                             const std::uint16_t *level_z_num,
                             const std::uint16_t *level_y_num,
                             const std::size_t level) {

    /*
     *
     *  Here we introduce updating Particle Cells at a level below.
     *
     */

    const int x_num = level_x_num[level];
    const int y_num = level_y_num[level];
    const int z_num = level_z_num[level];

    const int x_num_p = level_x_num[level - 1];
    const int y_num_p = level_y_num[level - 1];
    const int z_num_p = level_z_num[level - 1];

    const unsigned int N = 6;

    __shared__
    std::uint16_t local_patch[12][12][6]; // This is block wise shared memory this is assuming an 8*8 block with pad()


    if (threadIdx.x >= 12) {
        return;
    }
    if (threadIdx.z >= 12) {
        return;
    }


    int x_index = (8 * blockIdx.x + threadIdx.x - 2);
    int z_index = (8 * blockIdx.z + threadIdx.z - 2);


    bool not_ghost = false;

    if ((threadIdx.x > 1) && (threadIdx.x < 10) && (threadIdx.z > 1) && (threadIdx.z < 10)) {
        not_ghost = true;
    }


    if ((x_index >= x_num) || (x_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;
        return; //out of bounds
    }

    int x_index_p = (x_index) / 2;
    int z_index_p = (z_index) / 2;

    std::size_t current_row = level_offset[level] + (x_index) + (z_index) *
                                                                x_num; // the input to each kernel is its chunk index for which it should iterate over
    std::size_t current_row_p = level_offset[level - 1] + (x_index_p) + (z_index_p) *
                                                                        x_num_p; // the input to each kernel is its chunk index for which it should iterate over

    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    std::size_t particle_global_index_begin_p;
    std::size_t particle_global_index_end_p;

    // current level
    get_row_begin_end(&particle_global_index_begin, &particle_global_index_end, current_row, row_info);
    // parent level, level - 1, one resolution lower (coarser)
    get_row_begin_end(&particle_global_index_begin_p, &particle_global_index_end_p, current_row_p, row_info);

    std::size_t y_block = 1;
    std::uint16_t y_update_flag[3] = {0};
    std::size_t y_update_index[3] = {0};

    //current level variables
    std::size_t particle_index_l = particle_global_index_begin;
    std::uint16_t y_l = particle_y[particle_index_l];
    std::uint16_t f_l = particle_data_input[particle_index_l];


    //parent level variables
    std::size_t particle_index_p = particle_global_index_begin_p;
    std::uint16_t y_p = particle_y[particle_index_p];
    std::uint16_t f_p = particle_data_input[particle_index_p];


    if (particle_global_index_begin_p == particle_global_index_end_p) {
        y_p = y_num + 1;//no particles don't do anything
    }

    if (particle_global_index_begin == particle_global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }


    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)
    local_patch[threadIdx.z][threadIdx.x][(N - 2) % N] = 0; //this is at (y-2)

    const int filter_offset = 2;


    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T
        __syncthreads();
        //Check if its time to update the parent level


        if (j == (2 * y_p)) {
            local_patch[threadIdx.z][threadIdx.x][(j) % N] = f_p; //initial update
            local_patch[threadIdx.z][threadIdx.x][(j + 1) % N] = f_p;
        }


        //Check if its time to update current level
        if (j == y_l) {
            local_patch[threadIdx.z][threadIdx.x][j % N] = f_l; //initial update
            y_update_flag[j % 3] = 1;
            y_update_index[j % 3] = particle_index_l;
        } else {
            y_update_flag[j % 3] = 0;
        }

        //update at current level
        if ((y_l <= j) && ((particle_index_l + 1) < particle_global_index_end)) {
            particle_index_l++;
            y_l = particle_y[particle_index_l];
            f_l = particle_data_input[particle_index_l];
        }

        //parent update loop
        if ((2 * y_p <= j) && ((particle_index_p + 1) < particle_global_index_end_p)) {
            particle_index_p++;
            y_p = particle_y[particle_index_p];
            f_p = particle_data_input[particle_index_p];
        }

        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if (y_update_flag[(j - filter_offset + 3) % 3] == 1) {
            float neighbour_sum = 0;
            LOCALPATCHCONV555(particle_data_output, y_update_index[(j + 3 - filter_offset) % 3], threadIdx.z,
                              threadIdx.x, j - 2, neighbour_sum);


        }

    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num - 2) % 3] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;
        LOCALPATCHCONV555(particle_data_output, y_update_index[(y_num + 3 - 2) % 3], threadIdx.z, threadIdx.x,
                          y_num - 2, neighbour_sum);

    }

    __syncthreads();
    local_patch[threadIdx.z][threadIdx.x][(y_num + 1) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num - 1) % 3] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;
        LOCALPATCHCONV555(particle_data_output, y_update_index[(y_num + 3 - 1) % 3], threadIdx.z, threadIdx.x,
                          y_num - 1, neighbour_sum);
    }


}

__global__ void conv_interior_555(const std::size_t *row_info,
                                  const std::uint16_t *particle_y,
                                  const std::size_t *level_offset,
                                  const std::uint16_t *particle_data_input,
                                  const std::size_t *row_info_child,
                                  const std::uint16_t *particle_y_child,
                                  const std::size_t *level_offset_child,
                                  const std::float_t *particle_data_input_child,
                                  std::uint16_t *particle_data_output,
                                  const std::uint16_t *level_x_num,
                                  const std::uint16_t *level_z_num,
                                  const std::uint16_t *level_y_num,
                                  const std::size_t level) {
    /*
     *
     *  Here we update both those Particle Cells at a level below and above.
     *
     */

    const int x_num = level_x_num[level];
    const int y_num = level_y_num[level];
    const int z_num = level_z_num[level];

    const int x_num_p = level_x_num[level - 1];
    const int y_num_p = level_y_num[level - 1];
    const int z_num_p = level_z_num[level - 1];

    const unsigned int N = 6;


    __shared__
    std::uint16_t local_patch[12][12][6]; // This is block wise shared memory this is assuming an 8*8 block with pad()


    if (threadIdx.x >= 12) {
        return;
    }
    if (threadIdx.z >= 12) {
        return;
    }


    int x_index = (8 * blockIdx.x + threadIdx.x - 2);
    int z_index = (8 * blockIdx.z + threadIdx.z - 2);


    bool not_ghost = false;

    if ((threadIdx.x > 1) && (threadIdx.x < 10) && (threadIdx.z > 1) && (threadIdx.z < 10)) {
        not_ghost = true;
    }


    if ((x_index >= x_num) || (x_index < 0)) {
        //set the whole buffer to the boundary condition
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        //set the whole buffer to the zero boundary condition
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;

        return; //out of bounds
    }

    int x_index_p = (8 * blockIdx.x + threadIdx.x - 2) / 2;
    int z_index_p = (8 * blockIdx.z + threadIdx.z - 2) / 2;


    std::size_t current_row = level_offset[level] + (x_index) + (z_index) *
                                                                x_num; // the input to each kernel is its chunk index for which it should iterate over
    std::size_t current_row_p = level_offset[level - 1] + (x_index_p) + (z_index_p) *
                                                                        x_num_p; // the input to each kernel is its chunk index for which it should iterate over

    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    std::size_t particle_global_index_begin_p;
    std::size_t particle_global_index_end_p;

    /*
    * Current level variable initialization,
    */

    // current level
    get_row_begin_end(&particle_global_index_begin, &particle_global_index_end, current_row, row_info);
    // parent level, level - 1, one resolution lower (coarser)
    get_row_begin_end(&particle_global_index_begin_p, &particle_global_index_end_p, current_row_p, row_info);

    std::size_t y_block = 1;
    std::uint16_t y_update_flag[3] = {0};
    std::size_t y_update_index[3] = {0};

    //current level variables
    std::size_t particle_index_l = particle_global_index_begin;
    std::uint16_t y_l = particle_y[particle_index_l];
    std::uint16_t f_l = particle_data_input[particle_index_l];

    /*
    * Parent level variable initialization,
    */

    //parent level variables
    std::size_t particle_index_p = particle_global_index_begin_p;
    std::uint16_t y_p = particle_y[particle_index_p];
    std::uint16_t f_p = particle_data_input[particle_index_p];

    /*
    * Child level variable initialization, using 'Tree'
    * This is the same row as the current level
    */

    std::size_t current_row_child = level_offset_child[level] + (x_index) + (z_index) *
                                                                            x_num; // the input to each kernel is its chunk index for which it should iterate over

    std::size_t particle_global_index_begin_child;
    std::size_t particle_global_index_end_child;

    get_row_begin_end(&particle_global_index_begin_child, &particle_global_index_end_child, current_row_child,
                      row_info_child);

    std::size_t particle_index_child = particle_global_index_begin_child;
    std::uint16_t y_child = particle_y_child[particle_index_child];
    std::float_t f_child = particle_data_input_child[particle_index_child];

    if (particle_global_index_begin_child == particle_global_index_end_child) {
        y_child = y_num + 1;//no particles don't do anything
    }

    if (particle_global_index_begin_p == particle_global_index_end_p) {
        y_p = y_num + 1;//no particles don't do anything
    }

    if (particle_global_index_begin == particle_global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }

    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)
    local_patch[threadIdx.z][threadIdx.x][(N - 2) % N] = 0; //this is at (y-2)

    const int filter_offset = 2;


    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T

        //Check if its time to update the parent level
        if (j == (2 * y_p)) {
            local_patch[threadIdx.z][threadIdx.x][(j) % N] = f_p; //initial update
            local_patch[threadIdx.z][threadIdx.x][(j + 1) % N] = f_p;
        }

        //Check if its time to update child level
        if (j == y_child) {
            local_patch[threadIdx.z][threadIdx.x][y_child % N] = f_child; //initial update
        }

        //Check if its time to update current level
        if (j == y_l) {
            local_patch[threadIdx.z][threadIdx.x][y_l % N] = f_l; //initial update
            y_update_flag[j % 3] = 1;
            y_update_index[j % 3] = particle_index_l;
        } else {
            y_update_flag[j % 3] = 0;
        }


        //update at current level
        if ((y_l <= j) && ((particle_index_l + 1) < particle_global_index_end)) {
            particle_index_l++;
            y_l = particle_y[particle_index_l];
            f_l = particle_data_input[particle_index_l];
        }

        //parent update loop
        if ((2 * y_p <= j) && ((particle_index_p + 1) < particle_global_index_end_p)) {
            particle_index_p++;
            y_p = particle_y[particle_index_p];
            f_p = particle_data_input[particle_index_p];
        }


        //update at child level
        if ((y_child <= j) && ((particle_index_child + 1) < particle_global_index_end_child)) {
            particle_index_child++;
            y_child = particle_y_child[particle_index_child];
            f_child = particle_data_input_child[particle_index_child];
        }


        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if (y_update_flag[(j - filter_offset + 3) % 3] == 1) {
            float neighbour_sum = 0;

            LOCALPATCHCONV555(particle_data_output, y_update_index[(j + 3 - filter_offset) % 3], threadIdx.z,
                              threadIdx.x, j - 2, neighbour_sum);
        }
        __syncthreads();

    }

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();
    //set the boundary condition (zeros in this case)

    if (y_update_flag[(y_num - 2) % 3] == 1) { //the last particle (if it exists)

        float neighbour_sum = 0;

        LOCALPATCHCONV555(particle_data_output, y_update_index[(y_num + 3 - 2) % 3], threadIdx.z, threadIdx.x,
                          y_num - 2, neighbour_sum);

    }

    __syncthreads();
    local_patch[threadIdx.z][threadIdx.x][(y_num + 1) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num - 1) % 3] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;

        LOCALPATCHCONV555(particle_data_output, y_update_index[(y_num + 3 - 1) % 3], threadIdx.z, threadIdx.x,
                          y_num - 1, neighbour_sum);
    }


}


__global__ void conv_min_555(const std::size_t *row_info,
                             const std::uint16_t *particle_y,
                             const std::size_t *level_offset,
                             const std::uint16_t *particle_data_input,
                             const std::size_t *row_info_child,
                             const std::uint16_t *particle_y_child,
                             const std::size_t *level_offset_child,
                             const std::float_t *particle_data_input_child,
                             std::uint16_t *particle_data_output,
                             const std::uint16_t *level_x_num,
                             const std::uint16_t *level_z_num,
                             const std::uint16_t *level_y_num,
                             const std::size_t level) {

    /*
     *
     *  Here we introduce updating Particle Cells at a level below.
     *
     */

    const int x_num = level_x_num[level];
    const int y_num = level_y_num[level];
    const int z_num = level_z_num[level];

    const unsigned int N = 6;

    __shared__
    std::uint16_t local_patch[12][12][6]; // This is block wise shared memory this is assuming an 8*8 block with pad()


    if (threadIdx.x >= 12) {
        return;
    }
    if (threadIdx.z >= 12) {
        return;
    }


    int x_index = (8 * blockIdx.x + threadIdx.x - 2);
    int z_index = (8 * blockIdx.z + threadIdx.z - 2);


    bool not_ghost = false;

    if ((threadIdx.x > 1) && (threadIdx.x < 10) && (threadIdx.z > 1) && (threadIdx.z < 10)) {
        not_ghost = true;
    }


    if ((x_index >= x_num) || (x_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;

        return; //out of bounds
    }


    /*
     * Current level variable initialization
     *
     */

    std::size_t current_row = level_offset[level] + (x_index) + (z_index) *
                                                                x_num; // the input to each kernel is its chunk index for which it should iterate over
    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    // current level
    get_row_begin_end(&particle_global_index_begin, &particle_global_index_end, current_row, row_info);

    std::size_t y_block = 1;
    std::uint16_t y_update_flag[3] = {0};
    std::size_t y_update_index[3] = {0};

    //current level variables
    std::size_t particle_index_l = particle_global_index_begin;
    std::uint16_t y_l = particle_y[particle_index_l];
    std::uint16_t f_l = particle_data_input[particle_index_l];

    /*
    * Child level variable initialization, using 'Tree'
    * This is the same row as the current level
    */

    std::size_t current_row_child = level_offset_child[level] + (x_index) + (z_index) *
                                                                            x_num; // the input to each kernel is its chunk index for which it should iterate over

    std::size_t particle_global_index_begin_child;
    std::size_t particle_global_index_end_child;

    get_row_begin_end(&particle_global_index_begin_child, &particle_global_index_end_child, current_row_child,
                      row_info_child);

    std::size_t particle_index_child = particle_global_index_begin_child;
    std::uint16_t y_child = particle_y_child[particle_index_child];
    std::float_t f_child = particle_data_input_child[particle_index_child];


    if (particle_global_index_begin_child == particle_global_index_end_child) {
        y_child = y_num + 1;//no particles don't do anything
    }

    if (particle_global_index_begin == particle_global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }


    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)
    local_patch[threadIdx.z][threadIdx.x][(N - 2) % N] = 0; //this is at (y-2)

    const int filter_offset = 2;


    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T

        /*
         *
         * Current Level Update
         *
         */

        __syncthreads();

        //Check if its time to update current level
        if (j == y_l) {
            local_patch[threadIdx.z][threadIdx.x][y_l % N] = f_l; //initial update
            y_update_flag[j % 3] = 1;
            y_update_index[j % 3] = particle_index_l;
        } else {
            y_update_flag[j % 3] = 0;
        }

        //update at current level
        if ((y_l <= j) && ((particle_index_l + 1) < particle_global_index_end)) {
            particle_index_l++;
            y_l = particle_y[particle_index_l];
            f_l = particle_data_input[particle_index_l];
        }

        /*
         *
         * Child Level Update
         *
         */


        //Check if its time to update current level
        if (j == y_child) {
            local_patch[threadIdx.z][threadIdx.x][y_child % N] = f_child; //initial update
        }

        //update at current level
        if ((y_child <= j) && ((particle_index_child + 1) < particle_global_index_end_child)) {
            particle_index_child++;
            y_child = particle_y_child[particle_index_child];
            f_child = particle_data_input_child[particle_index_child];
        }


        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if (y_update_flag[(j - filter_offset + 3) % 3] == 1) {
            float neighbour_sum = 0;

            LOCALPATCHCONV555(particle_data_output, y_update_index[(j + 3 - filter_offset) % 3], threadIdx.z,
                              threadIdx.x, j - 2, neighbour_sum);
        }

    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num + 3 - 2) % 3] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;
        LOCALPATCHCONV555(particle_data_output, y_update_index[(y_num + 3 - 2) % 3], threadIdx.z, threadIdx.x,
                          y_num - 2, neighbour_sum);
    }

    __syncthreads();
    local_patch[threadIdx.z][threadIdx.x][(y_num + 1) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num + 3 - 1) % 3] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;
        LOCALPATCHCONV555(particle_data_output, y_update_index[(y_num + 3 - 1) % 3], threadIdx.z, threadIdx.x,
                          y_num - 1, neighbour_sum);
    }


}

#endif //LIBAPR_APRGPUISOCONV_HPP
