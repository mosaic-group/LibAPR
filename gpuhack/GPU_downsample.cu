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


    float gpu_iterate_time_si = timer.timings.back();
    //copy data back from gpu

    bool success = true;

    uint64_t c_fail= 0;
    uint64_t c_pass= 0;


    ExtraParticleData<float> tree_mean_gpu(aprTree);
    tree_mean_gpu.init_gpu(aprTree.total_number_parent_cells());

    cudaDeviceSynchronize();
    for (int i = 0; i < 2; ++i) {

        timer.start_timer("summing the sptial informatino for each partilce on the GPU");
        for (int rep = 0; rep < number_reps; ++rep) {

            for (int level = apr.level_min(); level <= apr.level_max(); ++level) {

                std::size_t number_rows_l = apr.spatial_index_x_max(level) * apr.spatial_index_z_max(level);
                std::size_t offset = gpuaprAccess.h_level_offset[level];

                std::size_t x_num = apr.spatial_index_x_max(level);
                std::size_t z_num = apr.spatial_index_z_max(level);
                std::size_t y_num = apr.spatial_index_y_max(level);

                dim3 threads_l(8, 1, 8);

                int x_blocks = (x_num + 8 - 1) / 8;
                int z_blocks = (z_num + 8 - 1) / 8;

                dim3 blocks_l(x_blocks, 1, z_blocks);


                if((level < apr.level_max()) && (level >= apr.level_min()) ) {
                    down_sample_avg_mid << < blocks_l, threads_l >> >
                                                   (gpuaprAccess.gpu_access.row_global_index,
                                                           gpuaprAccess.gpu_access.y_part_coord,
                                                           gpuaprAccess.gpu_access.level_offsets,
                                                           apr.particles_intensities.gpu_pointer,
                                                           gpuaprAccessTree.gpu_access.row_global_index,
                                                           gpuaprAccessTree.gpu_access.y_part_coord,
                                                           gpuaprAccessTree.gpu_access.level_offsets,
                                                           tree_mean_gpu.gpu_pointer,
                                                           gpuaprAccess.gpu_access.level_x_num,
                                                           gpuaprAccess.gpu_access.level_z_num,
                                                           gpuaprAccess.gpu_access.level_y_num,
                                                           level);
                }

                cudaDeviceSynchronize();
            }
        }

        timer.stop_timer();
    }

    float gpu_iterate_time_si3 = timer.timings.back();
    tree_mean_gpu.copy_data_to_host();

    tree_mean_gpu.gpu_data.clear();
    tree_mean_gpu.gpu_data.shrink_to_fit();

    std::cout << "Average time for loop insert max: " << (gpu_iterate_time_si3/(number_reps*1.0f))*1000 << " ms" << std::endl;
    std::cout << "Average time for loop insert max per million: " << (gpu_iterate_time_si3/(number_reps*1.0f*apr.total_number_particles()))*1000.0*1000000.0f << " ms" << std::endl;

    //////////////////////////
    ///
    /// Now check the data
    ///
    ////////////////////////////

    c_pass = 0;
    c_fail = 0;
    success=true;
    uint64_t output_c=0;

    for (uint64_t particle_number = 0; particle_number < aprTree.total_number_parent_cells(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        treeIt.set_iterator_to_particle_by_number(particle_number);
        //if(spatial_info_test[aprIt]==(aprIt.x() + aprIt.y() + aprIt.z() + aprIt.level())){
        if(tree_mean_gpu[treeIt]==ds_parts[treeIt]){
            c_pass++;
        } else {
            c_fail++;
            success = false;
            if(treeIt.level() <= treeIt.level_max()) {
                if (output_c < 1) {
                    std::cout << "Expected: " << ds_parts[treeIt] << " Recieved: " << tree_mean_gpu[treeIt] << " Level: " << treeIt.level() << " x: " << treeIt.x()
                              << " z: " << treeIt.z() << " y: " << treeIt.y() << std::endl;
                    output_c++;
                }
                //spatial_info_test3[aprIt] = 0;
            }

        }
    }

    if(success){
        std::cout << "Spatial information Check, PASS" << std::endl;
    } else {
        std::cout << "Spatial information Check, FAIL Total: " << c_fail << " Pass Total:  " << c_pass << std::endl;
    }



}




__device__ void get_row_begin_end(std::size_t* index_begin,
                                  std::size_t* index_end,
                                  std::size_t* current_row,
                                  const std::size_t *row_info){

    *index_end = (row_info[*current_row]);

    if (*current_row == 0) {
        *index_begin = 0;
    } else {
        *index_begin =(row_info[*current_row-1]);
    }


};

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
                                const std::size_t level) {
    /*
     *
     *  Here we update both those Particle Cells at a level below and above.
     *
     */

    const int x_num = level_x_num[level];
    const int y_num = level_y_num[level];
    const int z_num = level_z_num[level];

    const int x_num_p = level_x_num[level-1];
    const int y_num_p = level_y_num[level-1];
    const int z_num_p = level_z_num[level-1];

    int N = 2;

    __shared__ std::float_t local_patch[8][8][1]; // This is block wise shared memory this is assuming an 8*8 block with pad()

    if(threadIdx.x >= 8){
        return;
    }
    if(threadIdx.z >= 8){
        return;
    }

    int x_index = (8 * blockIdx.x + threadIdx.x );
    int z_index = (8 * blockIdx.z + threadIdx.z );


    bool not_ghost=false;

    if((threadIdx.x > 0) && (threadIdx.x < 9) && (threadIdx.z > 0) && (threadIdx.z < 9)){
        not_ghost = true;
    }


    if((x_index >= x_num) || (x_index < 0)){
        //set the whole buffer to the boundary condition

        return; //out of bounds
    }

    if((z_index >= z_num) || (z_index < 0)){
        //set the whole buffer to the zero boundary condition

        return; //out of bounds
    }

    int x_index_p = (8 * blockIdx.x + threadIdx.x )/2;
    int z_index_p = (8 * blockIdx.z + threadIdx.z )/2;


    std::size_t current_row = level_offset[level] + (x_index) + (z_index)*x_num; // the input to each kernel is its chunk index for which it should iterate over
    std::size_t current_row_p = level_offset[level-1] + (x_index_p) + (z_index_p)*x_num_p; // the input to each kernel is its chunk index for which it should iterate over

    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    std::size_t particle_global_index_begin_p;
    std::size_t particle_global_index_end_p;

    /*
    * Current level variable initialization,
    */

    // current level
    get_row_begin_end(&particle_global_index_begin, &particle_global_index_end, &current_row, row_info);
    // parent level, level - 1, one resolution lower (coarser)
    get_row_begin_end(&particle_global_index_begin_p, &particle_global_index_end_p, &current_row_p, row_info_child);


    //current level variables
    std::size_t particle_index_l = particle_global_index_begin;
    std::uint16_t y_l= particle_y[particle_index_l];
    std::uint16_t f_l = particle_data_input[particle_index_l];

    /*
    * Parent level variable initialization,
    */

    //parent level variables
    std::size_t particle_index_p = particle_global_index_begin_p;
    std::uint16_t y_p= particle_y_child[particle_index_p];
    std::uint16_t f_p = particle_data_output[particle_index_p];

    /*
    * Child level variable initialization, using 'Tree'
    * This is the same row as the current level
    */

    std::size_t current_row_child = level_offset_child[level] + (x_index) + (z_index)*x_num; // the input to each kernel is its chunk index for which it should iterate over

    std::size_t particle_global_index_begin_child;
    std::size_t particle_global_index_end_child;

    get_row_begin_end(&particle_global_index_begin_child, &particle_global_index_end_child, &current_row_child, row_info_child);

    std::size_t particle_index_child = particle_global_index_begin_child;
    std::uint16_t y_child= particle_y_child[particle_index_child];
    std::float_t f_child = particle_data_output[particle_index_child];

    if(particle_global_index_begin_child == particle_global_index_end_child){
        y_child = y_num+1;//no particles don't do anything
    }

    if(particle_global_index_begin_p == particle_global_index_end_p){
        y_p = y_num+1;//no particles don't do anything
    }

    if(particle_global_index_begin == particle_global_index_end){
        y_l = y_num+1;//no particles don't do anything
    }


    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T

        //Check if its time to update the parent level
        if(j==(2*y_p+1)) {
            local_patch[threadIdx.z][threadIdx.x][0] =  f_p; //initial update
            //y_update_flag[j%2]=1;
            //y_update_index[j%2] = particle_index_l;
        }

        //Check if its time to update child level
        if(j==y_child) {
            local_patch[threadIdx.z][threadIdx.x][0] =  f_child; //initial update
        }

        //Check if its time to update current level
        if(j==y_l) {
            local_patch[threadIdx.z][threadIdx.x][0] =  f_l; //initial update

        } else {

        }


        //update at current level
        if((y_l <= j) && ((particle_index_l+1) <particle_global_index_end)){
            particle_index_l++;
            y_l= particle_y[particle_index_l];
            f_l = particle_data_input[particle_index_l];
        }

        //parent update loop
        if((2*y_p <= j) && ((particle_index_p+1) <particle_global_index_end_p)){
            particle_index_p++;
            y_p= particle_y[particle_index_p];
            //f_p = particle_data_input[particle_index_p];
        }


        //update at child level
        if((y_child <= j) && ((particle_index_child+1) <particle_global_index_end_child)){
            particle_index_child++;
            y_child= particle_y_child[particle_index_child];
            f_child = particle_data_output[particle_index_child];
        }

        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter


    }

    //set the boundary condition (zeros in this case)

//    if(y_update_flag[(y_num-1)%2]==1){ //the last particle (if it exists)
//
//
//    }


}

