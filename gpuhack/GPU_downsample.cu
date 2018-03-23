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
    int num_rep = 100;
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

    if(command_option_exists(argv, argv + argc, "-numrep"))
    {
        result.num_rep = std::stoi(std::string(get_command_option(argv, argv + argc, "-numrep")));
    }

    return result;
}


__global__ void down_sample_avg_mid(const std::size_t *row_info,
                                const std::uint16_t *particle_y,
                                const std::size_t* level_offset,
                                const std::uint16_t *particle_data_input,
                                const std::size_t *row_info_child,
                                const std::uint16_t *particle_y_child,
                                const std::size_t* level_offset_child,
                                std::float_t *particle_data_output,
                                const std::uint16_t* level_x_num,
                                const std::uint16_t* level_z_num,
                                const std::uint16_t* level_y_num,
                                const std::size_t level);

__global__ void down_sample_avg(const std::size_t *row_info,
                                const std::uint16_t *particle_y,
                                const std::size_t* level_offset,
                                const std::uint16_t *particle_data_input,
                                const std::size_t *row_info_child,
                                const std::uint16_t *particle_y_child,
                                const std::size_t* level_offset_child,
                                std::float_t *particle_data_output,
                                const std::uint16_t* level_x_num,
                                const std::uint16_t* level_z_num,
                                const std::uint16_t* level_y_num,
                                const std::size_t level);

__global__ void shared_update(const std::size_t *row_info,
                              const std::uint16_t *particle_y,
                              const std::uint16_t *particle_data_input,
                              std::uint16_t *particle_data_output,
                              const std::size_t offset,
                              const std::size_t x_num,
                              const std::size_t z_num,
                              const std::size_t y_num,
                              const std::size_t level);


ExtraParticleData<float> meanDownsamplingOld(APR<uint16_t> &aInputApr, APRTree<uint16_t> &aprTree) {
    APRIterator<uint16_t> aprIt(aInputApr);
    APRTreeIterator<uint16_t> treeIt(aprTree);
    APRTreeIterator<uint16_t> parentTreeIt(aprTree);
    ExtraParticleData<float> outputTree(aprTree);
    ExtraParticleData<uint8_t> childCnt(aprTree);
    auto &intensities = aInputApr.particles_intensities;

    for (unsigned int level = aprIt.level_max(); level >= aprIt.level_min(); --level) {
        for (size_t particle_number = aprIt.particles_level_begin(level);
             particle_number < aprIt.particles_level_end(level);
             ++particle_number)
        {
            aprIt.set_iterator_to_particle_by_number(particle_number);
            parentTreeIt.set_iterator_to_parent(aprIt);

            auto val = intensities[aprIt];
            outputTree[parentTreeIt] += val;
            childCnt[parentTreeIt]++;
        }
    }

    //then do the rest of the tree where order matters (it goes to level_min since we need to eventually average data there).
    for (unsigned int level = treeIt.level_max(); level >= treeIt.level_min(); --level) {
        // average intensities first
        for (size_t particleNumber = treeIt.particles_level_begin(level);
             particleNumber < treeIt.particles_level_end(level);
             ++particleNumber)
        {
            treeIt.set_iterator_to_particle_by_number(particleNumber);
            outputTree[treeIt] /= (1.0*childCnt[treeIt]);
        }

        // push changes
        if (level > treeIt.level_min())
            for (uint64_t parentNumber = treeIt.particles_level_begin(level);
                 parentNumber < treeIt.particles_level_end(level);
                 ++parentNumber)
            {
                treeIt.set_iterator_to_particle_by_number(parentNumber);
                if (parentTreeIt.set_iterator_to_parent(treeIt)) {
                    outputTree[parentTreeIt] += outputTree[treeIt];
                    childCnt[parentTreeIt]++;
                }
            }
    }
    return outputTree;
}


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
    GPUAPRAccess gpuaprAccess(aprIt);

    // Get dense representation of APR tree
    timer.start_timer("gen tree");
    APRTree<uint16_t> aprTree(apr);
    timer.stop_timer();

    APRTreeIterator<uint16_t> treeIt(aprTree);

    GPUAPRAccess gpuaprAccessTree(treeIt);

    /*
     *
     *  Calculate the down-sampled Particle Values
     *
     *
     */

    timer.start_timer("generate ds particles");
    ExtraParticleData<float> ds_parts =  meanDownsamplingOld(apr, aprTree);
    timer.stop_timer();

    std::cout << "Number parts: " << aprIt.total_number_particles() << " number in interior tree: " << ds_parts.data.size() << std::endl;


    ds_parts.copy_data_to_gpu();


    int number_reps = options.num_rep;




    /*
    *  Test the x,y,z,level information is correct
    *
    */


    apr.particles_intensities.copy_data_to_gpu();


    //copy data back from gpu

    bool success = true;

    uint64_t c_fail= 0;
    uint64_t c_pass= 0;


    ExtraParticleData<float> tree_mean_gpu(aprTree);
    tree_mean_gpu.init_gpu(aprTree.total_number_parent_cells());

    ExtraParticleData<float> dummy(apr);
    dummy.init_gpu(apr.total_number_particles());


    for (int i = 0; i < 2; ++i) {

        timer.start_timer("summing the sptial informatino for each partilce on the GPU");
        for (int rep = 0; rep < number_reps; ++rep) {

            for (int level = apr.level_max(); level > aprIt.level_min(); --level) {

                std::size_t number_rows_l = apr.spatial_index_x_max(level) * apr.spatial_index_z_max(level);
                std::size_t offset = gpuaprAccess.h_level_offset[level];

                std::size_t x_num = apr.spatial_index_x_max(level);
                std::size_t z_num = apr.spatial_index_z_max(level);
                std::size_t y_num = apr.spatial_index_y_max(level);

                dim3 threads_l(32, 1, 1);

                int x_blocks = (number_rows_l + 32 - 1) / 32;
                int z_blocks = 1;

                dim3 blocks_l(x_blocks, 1, z_blocks);



                down_sample_avg << < blocks_l, threads_l >> >
                                                       (gpuaprAccess.gpu_access.row_global_index,
                                                               gpuaprAccess.gpu_access.y_part_coord,
                                                               gpuaprAccess.gpu_access.level_offsets,
                                                               apr.particles_intensities.gpu_pointer,
                                                               gpuaprAccessTree.gpu_access.row_global_index,
                                                               gpuaprAccessTree.gpu_access.y_part_coord,
                                                               gpuaprAccessTree.gpu_access.level_offsets,
                                                               dummy.gpu_pointer,
                                                               gpuaprAccess.gpu_access.level_x_num,
                                                               gpuaprAccess.gpu_access.level_z_num,
                                                               gpuaprAccess.gpu_access.level_y_num,
                                                               level);


                cudaDeviceSynchronize();
            }
        }

        timer.stop_timer();
    }

    float gpu_iterate_time_batch = timer.timings.back();
    std::cout << "Average time NEW for loop insert max: " << (gpu_iterate_time_batch/(number_reps*1.0f))*1000 << " ms" << std::endl;
    std::cout << "Average time NEW for loop insert max per million: " << (gpu_iterate_time_batch/(number_reps*1.0f*apr.total_number_particles()))*1000.0*1000000.0f << " ms" << std::endl;


    //copy data back from gpu
    dummy.copy_data_to_host();

    dummy.gpu_data.clear();
    dummy.gpu_data.shrink_to_fit();


    cudaDeviceSynchronize();
    ExtraParticleData<uint16_t> spatial_info_test(apr);
    spatial_info_test.init_gpu(apr.total_number_particles());


    timer.start_timer("summing the sptial informatino for each partilce on the GPU");
    for (int rep = 0; rep < number_reps; ++rep) {

        for (int level = apr.level_max(); level > aprIt.level_min(); --level) {

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
                                        (gpuaprAccess.gpu_access.row_global_index, gpuaprAccess.gpu_access.y_part_coord, apr.particles_intensities.gpu_pointer,spatial_info_test.gpu_pointer, offset,x_num,z_num,y_num,level);

            cudaDeviceSynchronize();
        }
    }

    timer.stop_timer();

    float gpu_iterate_time_si = timer.timings.back();

    std::cout << "Average time old shared for loop insert max: " << (gpu_iterate_time_si/(number_reps*1.0f))*1000 << " ms" << std::endl;
    std::cout << "Average time old shared for loop insert max per million: " << (gpu_iterate_time_si/(number_reps*1.0f*apr.total_number_particles()))*1000.0*1000000.0f << " ms" << std::endl;

    //copy data back from gpu
    spatial_info_test.copy_data_to_host();

    spatial_info_test.gpu_data.clear();
    spatial_info_test.gpu_data.shrink_to_fit();

    //////////////////////////
    ///
    /// Now check the data
    ///
    ////////////////////////////

    c_pass = 0;
    c_fail = 0;
    success=true;
    uint64_t output_c=0;

//    for (uint64_t particle_number = 0; particle_number < aprTree.total_number_parent_cells(); ++particle_number) {
//        //This step is required for all loops to set the iterator by the particle number
//        treeIt.set_iterator_to_particle_by_number(particle_number);
//        //if(spatial_info_test[aprIt]==(aprIt.x() + aprIt.y() + aprIt.z() + aprIt.level())){
//        if(tree_mean_gpu[treeIt]==ds_parts[treeIt]){
//            c_pass++;
//        } else {
//            c_fail++;
//            success = false;
//            if(treeIt.level() <= treeIt.level_max()) {
//                if (output_c < 1) {
//                    std::cout << "Expected: " << ds_parts[treeIt] << " Recieved: " << tree_mean_gpu[treeIt] << " Level: " << treeIt.level() << " x: " << treeIt.x()
//                              << " z: " << treeIt.z() << " y: " << treeIt.y() << std::endl;
//                    output_c++;
//                }
//                //spatial_info_test3[aprIt] = 0;
//            }
//
//        }
//    }


    for (uint64_t particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        aprIt.set_iterator_to_particle_by_number(particle_number);
        //if(spatial_info_test[aprIt]==(aprIt.x() + aprIt.y() + aprIt.z() + aprIt.level())){
        if(dummy[aprIt]==apr.particles_intensities[aprIt]){
            c_pass++;
        } else {
            c_fail++;
            success = false;
            if(aprIt.level() <= aprIt.level_max()) {
                if (output_c < 5) {
                    std::cout << "Expected: " << apr.particles_intensities[aprIt] << " Recieved: " << dummy[aprIt] << " Level: " << aprIt.level() << " x: " << aprIt.x()
                              << " z: " << aprIt.z() << " y: " << aprIt.y() << std::endl;
                    output_c++;
                }
                //spatial_info_test3[aprIt] = 0;
            }

        }
    }


    if(success){
        std::cout << "Direct insert, PASS" << std::endl;
    } else {
        std::cout << "Direct insert Check, FAIL Total: " << c_fail << " Pass Total:  " << c_pass << std::endl;
    }



}




__device__ void get_row_begin_end(std::size_t* index_begin,
                                  std::size_t* index_end,
                                  std::size_t current_row,
                                  const std::size_t *row_info){

    *index_end = (row_info[current_row]);

    if (current_row == 0) {
        *index_begin = 0;
    } else {
        *index_begin =(row_info[current_row-1]);
    }


};



__global__ void down_sample_avg(const std::size_t *row_info,
                                const std::uint16_t *particle_y,
                                const std::size_t* level_offset,
                                const std::uint16_t *particle_data_input,
                                const std::size_t *row_info_child,
                                const std::uint16_t *particle_y_child,
                                const std::size_t* level_offset_child,
                                std::float_t *particle_data_output,
                                const std::uint16_t* level_x_num,
                                const std::uint16_t* level_z_num,
                                const std::uint16_t* level_y_num,
                                const std::size_t level) {

    const int x_num = level_x_num[level];
    const int y_num = level_y_num[level];
    const int z_num = level_z_num[level];

    const int x_num_p = level_x_num[level-1];
    const int y_num_p = level_y_num[level-1];
    const int z_num_p = level_z_num[level-1];

    const int local_row_index = (blockDim.x * blockIdx.x + threadIdx.x );

    if(blockDim.x * blockIdx.x >= x_num*z_num){
        return;
    }

    std::size_t global_row_index_begin = blockDim.x * blockIdx.x + level_offset[level];
    std::size_t global_row_index_end = min(blockDim.x * blockIdx.x + 31,z_num*x_num-1) + level_offset[level];

    std::size_t global_index_begin_0;
    std::size_t global_index_end_0;

    std::size_t global_index_begin_p;
    std::size_t global_index_end_p;

    __shared__ std::uint16_t f_cache[32];
    __shared__ std::uint16_t y_cache[32];

    __shared__ std::uint16_t p_y_cache[32];

    __shared__ std::uint16_t y_buffer[32];

    std::size_t num_rows = global_row_index_end - global_row_index_begin;


    //ying printf("hello begin %d end %d chunks %d number parts %d \n",(int) global_index_begin_0,(int) global_index_end_f, (int) number_chunk, (int) number_parts);

    for (std::size_t row = global_row_index_begin; row <= global_row_index_end; ++row) {

        get_row_begin_end(&global_index_begin_0, &global_index_end_0, row, row_info);
        std::size_t number_parts = global_index_end_0 - global_index_begin_0;
        std::uint16_t number_chunk = ((number_parts+31)/32);

        std::uint16_t z_  = local_row_index/x_num;

        std::uint16_t x_ = local_row_index - z_*x_num;

        std::uint16_t x_p = x_/2;
        std::uint16_t z_p = z_/2;
        std::size_t parent_row = z_p*x_num_p + x_p + level_offset_child[level-1];

        get_row_begin_end(&global_index_begin_p, &global_index_end_p, parent_row, row_info);

        for (int i = 0; i < (number_chunk); ++i) {

            //read in as blocks
            if (i * 32 + global_index_begin_0 + threadIdx.x < global_index_end_0) {
                f_cache[threadIdx.x] = particle_data_input[i * 32 + global_index_begin_0 + threadIdx.x];
            } else {
                f_cache[threadIdx.x] = 0;
            }

            if (i * 32 + global_index_begin_0 + threadIdx.x < global_index_end_0) {
                y_cache[threadIdx.x] = particle_y[i * 32 + global_index_begin_0 + threadIdx.x];
            } else {
                y_cache[threadIdx.x] = 0;
            }



            // Do something

            //write out as blocks
            if (i * 32 + global_index_begin_0 + threadIdx.x < global_index_end_0) {

                y_buffer[y_cache[threadIdx.x]%32] = f_cache[threadIdx.x];
            }

            //inner y loop they all check if they have the lucky winner   NEED TO USE 32 length case lines


            //write out as blocks
            if (i * 32 + global_index_begin_0 + threadIdx.x < global_index_end_0) {

                particle_data_output[i * 32 + global_index_begin_0 + threadIdx.x] = f_cache[threadIdx.x];
            }
        }
    }

}

__global__ void loop_no_xz(const std::size_t *row_info,
                                    const std::uint16_t *particle_y,
                                    const std::size_t* level_offset,
                                    const std::uint16_t *particle_data_input,
                                    const std::size_t *row_info_child,
                                    const std::uint16_t *particle_y_child,
                                    const std::size_t* level_offset_child,
                                    std::float_t *particle_data_output,
                                    const std::uint16_t* level_x_num,
                                    const std::uint16_t* level_z_num,
                                    const std::uint16_t* level_y_num,
                                    const std::size_t level) {

    const int x_num = level_x_num[level];
    const int y_num = level_y_num[level];
    const int z_num = level_z_num[level];

    const int local_row_index = (blockDim.x * blockIdx.x + threadIdx.x );

    if(blockDim.x * blockIdx.x >= x_num*z_num){
        return;
    }

    std::size_t global_row_index_begin = blockDim.x * blockIdx.x + level_offset[level];
    std::size_t global_row_index_end = min(blockDim.x * blockIdx.x + 31,z_num*x_num-1) + level_offset[level];

    std::size_t global_index_begin_0;
    std::size_t global_index_end_0;

    std::size_t global_index_begin_f;
    std::size_t global_index_end_f;

    __shared__ std::uint16_t f_cache[32];
    __shared__ std::uint16_t y_cache[32];


    std::size_t num_rows = global_row_index_end - global_row_index_begin;


    //ying printf("hello begin %d end %d chunks %d number parts %d \n",(int) global_index_begin_0,(int) global_index_end_f, (int) number_chunk, (int) number_parts);

    for (std::size_t row = global_row_index_begin; row <= global_row_index_end; ++row) {

        get_row_begin_end(&global_index_begin_0, &global_index_end_0, row, row_info);
        std::size_t number_parts = global_index_end_0 - global_index_begin_0;
        std::size_t number_chunk = ((number_parts+31)/32);

        std::size_t z_  = local_row_index/x_num;

        std::size_t x_ = local_row_index - z_*x_num;

        for (int i = 0; i < (number_chunk); ++i) {

            //read in as blocks
            if (i * 32 + global_index_begin_0 + threadIdx.x < global_index_end_0) {
                f_cache[threadIdx.x] = particle_data_input[i * 32 + global_index_begin_0 + threadIdx.x];
            } else {
                f_cache[threadIdx.x] = 0;
            }

            if (i * 32 + global_index_begin_0 + threadIdx.x < global_index_end_0) {
                y_cache[threadIdx.x] = particle_y[i * 32 + global_index_begin_0 + threadIdx.x];
            } else {
                y_cache[threadIdx.x] = 0;
            }

            // Do something

            //inner y loop they all check if they have the lucky winner   NEED TO USE 32 length case lines


            //write out as blocks
            if (i * 32 + global_index_begin_0 + threadIdx.x < global_index_end_0) {

                particle_data_output[i * 32 + global_index_begin_0 + threadIdx.x] = f_cache[threadIdx.x];
            }
        }
    }

}




__global__ void shared_update(const std::size_t *row_info,
                              const std::uint16_t *particle_y,
                              const std::uint16_t *particle_data_input,
                              std::uint16_t *particle_data_output,
                              const std::size_t offset,
                              const std::size_t x_num,
                              const std::size_t z_num,
                              const std::size_t y_num,
                              const std::size_t level) {

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

    particle_global_index_end = (row_info[current_row]);

    if (current_row == 0) {
        particle_global_index_begin = 0;
    } else {
        particle_global_index_begin = (row_info[current_row-1]);
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
