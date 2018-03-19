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

void create_test_particles(APR<uint16_t>& apr,APRIterator<uint16_t>& apr_iterator,APRTreeIterator<uint16_t>& apr_tree_iterator,ExtraParticleData<float> &test_particles,ExtraParticleData<uint16_t>& particles,ExtraParticleData<float>& part_tree,std::vector<double>& stencil, const int stencil_size, const int stencil_half){

    for (uint64_t level_local = apr_iterator.level_max(); level_local >= apr_iterator.level_min(); --level_local) {


        MeshData<float> by_level_recon;
        by_level_recon.init(apr_iterator.spatial_index_y_max(level_local),apr_iterator.spatial_index_x_max(level_local),apr_iterator.spatial_index_z_max(level_local),0);

        for (uint64_t level = std::max((uint64_t)(level_local-1),(uint64_t)apr_iterator.level_min()); level <= level_local; ++level) {


            const float step_size = pow(2, level_local - level);

            uint64_t particle_number;

            for (particle_number = apr_iterator.particles_level_begin(level);
                 particle_number < apr_iterator.particles_level_end(level); ++particle_number) {
                //
                //  Parallel loop over level
                //
                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                int dim1 = apr_iterator.y() * step_size;
                int dim2 = apr_iterator.x() * step_size;
                int dim3 = apr_iterator.z() * step_size;

                float temp_int;
                //add to all the required rays

                temp_int = particles[apr_iterator];

                const int offset_max_dim1 = std::min((int) by_level_recon.y_num, (int) (dim1 + step_size));
                const int offset_max_dim2 = std::min((int) by_level_recon.x_num, (int) (dim2 + step_size));
                const int offset_max_dim3 = std::min((int) by_level_recon.z_num, (int) (dim3 + step_size));

                for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                    for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                        for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                            by_level_recon.mesh[i + (k) * by_level_recon.y_num + q * by_level_recon.y_num * by_level_recon.x_num] = temp_int;
                        }
                    }
                }
            }
        }


        if(level_local < apr_iterator.level_max()){

            uint64_t level = level_local;

            const float step_size = 1;

            uint64_t particle_number;

            for (particle_number = apr_tree_iterator.particles_level_begin(level);
                 particle_number < apr_tree_iterator.particles_level_end(level); ++particle_number) {
                //
                //  Parallel loop over level
                //
                apr_tree_iterator.set_iterator_to_particle_by_number(particle_number);

                int dim1 = apr_tree_iterator.y() * step_size;
                int dim2 = apr_tree_iterator.x() * step_size;
                int dim3 = apr_tree_iterator.z() * step_size;

                float temp_int;
                //add to all the required rays

                temp_int = part_tree[apr_tree_iterator];


                const int offset_max_dim1 = std::min((int) by_level_recon.y_num, (int) (dim1 + step_size));
                const int offset_max_dim2 = std::min((int) by_level_recon.x_num, (int) (dim2 + step_size));
                const int offset_max_dim3 = std::min((int) by_level_recon.z_num, (int) (dim3 + step_size));

                for (int64_t q = dim3; q < offset_max_dim3; ++q) {

                    for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                        for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                            by_level_recon.mesh[i + (k) * by_level_recon.y_num + q * by_level_recon.y_num * by_level_recon.x_num] = temp_int;
                        }
                    }
                }
            }

        }


        int x = 0;
        int z = 0;
        uint64_t level = level_local;

        for (z = 0; z < apr.spatial_index_z_max(level); ++z) {
            //lastly loop over particle locations and compute filter.
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.particles_zx_end(level, z,
                                                                                 x); apr_iterator.set_iterator_to_particle_next_particle()) {
                    double neigh_sum = 0;
                    float counter = 0;

                    const int k = apr_iterator.y(); // offset to allow for boundary padding
                    const int i = x;

                    for (int l = -stencil_half; l < stencil_half+1; ++l) {
                        for (int q = -stencil_half; q < stencil_half+1; ++q) {
                            for (int w = -stencil_half; w < stencil_half+1; ++w) {

                                if((k+w)>=0 & (k+w) < (apr.spatial_index_y_max(level))){
                                    if((i+q)>=0 & (i+q) < (apr.spatial_index_x_max(level))){
                                        if((z+l)>=0 & (z+l) < (apr.spatial_index_z_max(level))){
                                            neigh_sum += stencil[counter] * by_level_recon.at(k + w, i + q, z+l);
                                        }
                                    }
                                }


                                counter++;
                            }
                        }
                    }

                    test_particles[apr_iterator] = std::round(neigh_sum/(1.0f*pow(stencil_size,3)));

                }
            }
        }




        // std::string image_file_name = apr.parameters.input_dir + std::to_string(level_local) + "_by_level.tif";
        //TiffUtils::saveMeshAsTiff(image_file_name, by_level_recon);

    }

}


__global__ void shared_update_max(const std::size_t *row_info,
                                  const std::uint16_t *particle_y,
                                  const std::uint16_t *particle_data_input,
                                  std::uint16_t *particle_data_output,
                                  const std::size_t* level_offset,
                                  const std::uint16_t* level_x_num,
                                  const std::uint16_t* level_z_num,
                                  const std::uint16_t* level_y_num,
                                  const std::size_t level) ;

__global__ void shared_update_min(const std::size_t *row_info,
                                  const std::uint16_t *particle_y,
                                  const std::size_t* level_offset,
                                  const std::uint16_t *particle_data_input,
                                  const std::size_t *row_info_child,
                                  const std::uint16_t *particle_y_child,
                                  const std::size_t* level_offset_child,
                                  const std::float_t *particle_data_input_child,
                                  std::uint16_t *particle_data_output,
                                  const std::uint16_t* level_x_num,
                                  const std::uint16_t* level_z_num,
                                  const std::uint16_t* level_y_num,
                                  const std::size_t level) ;

__global__ void shared_update_interior_level(const std::size_t *row_info,
                                             const std::uint16_t *particle_y,
                                             const std::size_t* level_offset,
                                             const std::uint16_t *particle_data_input,
                                             const std::size_t *row_info_child,
                                             const std::uint16_t *particle_y_child,
                                             const std::size_t* level_offset_child,
                                             const std::float_t *particle_data_input_child,
                                             std::uint16_t *particle_data_output,
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


    /*
     *  Now launch the kernels across all the chunks determiend by the load balancing
     *
     */

    ExtraParticleData<uint16_t> iteration_check_particles(apr);
    iteration_check_particles.init_gpu(apr.total_number_particles());

    int number_reps = options.num_rep;

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



    ExtraParticleData<uint16_t> spatial_info_test3(apr);
    spatial_info_test3.init_gpu(apr.total_number_particles());

    cudaDeviceSynchronize();

    timer.start_timer("summing the sptial informatino for each partilce on the GPU");
    for (int rep = 0; rep < number_reps; ++rep) {

        for (int level = apr.level_min(); level <= apr.level_max(); ++level) {

            std::size_t number_rows_l = apr.spatial_index_x_max(level) * apr.spatial_index_z_max(level);
            std::size_t offset = gpuaprAccess.h_level_offset[level];

            std::size_t x_num = apr.spatial_index_x_max(level);
            std::size_t z_num = apr.spatial_index_z_max(level);
            std::size_t y_num = apr.spatial_index_y_max(level);

            dim3 threads_l(12, 1, 12);

            int x_blocks = (x_num + 8 - 1) / 8;
            int z_blocks = (z_num + 8 - 1) / 8;

            dim3 blocks_l(x_blocks, 1, z_blocks);

            if(level==apr.level_min()){
                shared_update_min <<< blocks_l, threads_l >>>
                                                (gpuaprAccess.gpu_access.row_global_index,
                                                        gpuaprAccess.gpu_access.y_part_coord,
                                                        gpuaprAccess.gpu_access.level_offsets,
                                                        apr.particles_intensities.gpu_pointer,
                                                        gpuaprAccessTree.gpu_access.row_global_index,
                                                        gpuaprAccessTree.gpu_access.y_part_coord,
                                                        gpuaprAccessTree.gpu_access.level_offsets,
                                                        ds_parts.gpu_pointer,
                                                        spatial_info_test3.gpu_pointer,
                                                        gpuaprAccess.gpu_access.level_x_num,
                                                        gpuaprAccess.gpu_access.level_z_num,
                                                        gpuaprAccess.gpu_access.level_y_num,
                                                        level);

            } else if(level==apr.level_max()) {
                shared_update_max <<< blocks_l, threads_l >>>
                                                (gpuaprAccess.gpu_access.row_global_index, gpuaprAccess.gpu_access.y_part_coord, apr.particles_intensities.gpu_pointer, spatial_info_test3.gpu_pointer, gpuaprAccess.gpu_access.level_offsets, gpuaprAccess.gpu_access.level_x_num, gpuaprAccess.gpu_access.level_z_num, gpuaprAccess.gpu_access.level_y_num, level);

            } else {
                shared_update_interior_level <<< blocks_l, threads_l >>>
                                                           (gpuaprAccess.gpu_access.row_global_index,
                                                                   gpuaprAccess.gpu_access.y_part_coord,
                                                                   gpuaprAccess.gpu_access.level_offsets,
                                                                   apr.particles_intensities.gpu_pointer,
                                                                   gpuaprAccessTree.gpu_access.row_global_index,
                                                                   gpuaprAccessTree.gpu_access.y_part_coord,
                                                                   gpuaprAccessTree.gpu_access.level_offsets,
                                                                   ds_parts.gpu_pointer,
                                                                   spatial_info_test3.gpu_pointer,
                                                                   gpuaprAccess.gpu_access.level_x_num,
                                                                   gpuaprAccess.gpu_access.level_z_num,
                                                                   gpuaprAccess.gpu_access.level_y_num,
                                                                   level);
            }
            cudaDeviceSynchronize();
        }
    }

    timer.stop_timer();

    float gpu_iterate_time_si3 = timer.timings.back();
    spatial_info_test3.copy_data_to_host();

    spatial_info_test3.gpu_data.clear();
    spatial_info_test3.gpu_data.shrink_to_fit();

    std::cout << "Average time for loop insert max: " << (gpu_iterate_time_si3/(number_reps*1.0f))*1000 << " ms" << std::endl;

    //////////////////////////
    ///
    /// Now check the data
    ///
    ////////////////////////////
    std::vector<double> stencil;
    stencil.resize(27,1);
    ExtraParticleData<float> output(apr);
    create_test_particles( apr, aprIt,treeIt,output,apr.particles_intensities,ds_parts,stencil, 3, 1);


    uint64_t c_pass = 0;
    uint64_t c_fail = 0;
    bool success=true;
    uint64_t output_c=0;

    for (uint64_t particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        aprIt.set_iterator_to_particle_by_number(particle_number);
        //if(spatial_info_test[aprIt]==(aprIt.x() + aprIt.y() + aprIt.z() + aprIt.level())){
        if(spatial_info_test3[aprIt]==output[aprIt]){
            c_pass++;
        } else {
            c_fail++;
            success = false;
            if(aprIt.level() <= aprIt.level_max()) {
                if (output_c < 1) {
                    std::cout << "Expected: " << output[aprIt] << " Recieved: " << spatial_info_test3[aprIt] << " Level: " << aprIt.level() << " x: " << aprIt.x()
                              << " z: " << aprIt.z() << " y: " << aprIt.y() << std::endl;
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



//    MeshData<uint16_t> check_mesh;
//
//    apr.interp_img(check_mesh,spatial_info_test3);
//
//    std::string image_file_name = options.directory +  "conv3_gpu.tif";
//    TiffUtils::saveMeshAsTiff(image_file_name, check_mesh);
//
//    apr.interp_img(check_mesh,output);
//
//    image_file_name = options.directory +  "conv3_gt.tif";
//    TiffUtils::saveMeshAsTiff(image_file_name, check_mesh);



}


//
//  This kernel checks that every particle is only visited once in the iteration
//


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

#define LOCALPATCHUPDATE(particle_output,index,z,x,j)\
if (not_ghost) {\
    particle_output[index] = local_patch[z][x][j];\
}\

#define LOCALPATCHCONV(particle_output,index,z,x,y,neighbour_sum)\
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
particle_output[index] = std::round(neighbour_sum / 27.0f);\
}\




__global__ void shared_update_max(const std::size_t *row_info,
                                  const std::uint16_t *particle_y,
                                  const std::uint16_t *particle_data_input,
                                  std::uint16_t *particle_data_output,
                                  const std::size_t* level_offset,
                                  const std::uint16_t* level_x_num,
                                  const std::uint16_t* level_z_num,
                                  const std::uint16_t* level_y_num,
                                  const std::size_t level)  {

    /*
     *
     *  Here we introduce updating Particle Cells at a level below.
     *
     */

    const int x_num = level_x_num[level];
    const int y_num = level_y_num[level];
    const int z_num = level_z_num[level];

    const int x_num_p = level_x_num[level-1];
    const int y_num_p = level_y_num[level-1];
    const int z_num_p = level_z_num[level-1];

    const unsigned int N = 5;
    const unsigned int Nd = 10;

    __shared__ std::float_t local_patch[12][12][5]; // This is block wise shared memory this is assuming an 8*8 block with pad()


    if(threadIdx.x >= 12){
        return;
    }
    if(threadIdx.z >= 12){
        return;
    }


    int x_index = (8 * blockIdx.x + threadIdx.x - 2);
    int z_index = (8 * blockIdx.z + threadIdx.z - 2);


    bool not_ghost=false;

    if((threadIdx.x > 1) && (threadIdx.x < 10) && (threadIdx.z > 1) && (threadIdx.z < 10)){
        not_ghost = true;
    }


    if((x_index >= x_num) || (x_index < 0)){
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][2 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][3 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][4 ] = 0;

        return; //out of bounds
    }

    if((z_index >= z_num) || (z_index < 0)){
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][2 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][3 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][4 ] = 0;
        return; //out of bounds
    }

    int x_index_p = (x_index)/2;
    int z_index_p = (z_index)/2;

    std::size_t current_row = level_offset[level] + (x_index) + (z_index)*x_num; // the input to each kernel is its chunk index for which it should iterate over
    std::size_t current_row_p = level_offset[level-1] + (x_index_p) + (z_index_p)*x_num_p; // the input to each kernel is its chunk index for which it should iterate over

    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    std::size_t particle_global_index_begin_p;
    std::size_t particle_global_index_end_p;

    // current level
    get_row_begin_end(&particle_global_index_begin, &particle_global_index_end, &current_row, row_info);
    // parent level, level - 1, one resolution lower (coarser)
    get_row_begin_end(&particle_global_index_begin_p, &particle_global_index_end_p, &current_row_p, row_info);

    std::size_t y_block = 1;
    std::uint16_t y_update_flag[3] = {0};
    std::size_t y_update_index[3] = {0};

    //current level variables
    std::size_t particle_index_l = particle_global_index_begin;
    std::uint16_t y_l= particle_y[particle_index_l];
    std::uint16_t f_l = particle_data_input[particle_index_l];


    //parent level variables
    std::size_t particle_index_p = particle_global_index_begin_p;
    std::uint16_t y_p= particle_y[particle_index_p];
    std::uint16_t f_p = particle_data_input[particle_index_p];


    if(particle_global_index_begin_p == particle_global_index_end_p){
        y_p = y_num+1;//no particles don't do anything
    }

    if(particle_global_index_begin == particle_global_index_end){
        y_l = y_num+1;//no particles don't do anything
    }


    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N-1) % N ] = 0; //this is at (y-1)

    const int filter_offset = 2;

    double neighbour_sum = 0;

    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T
        __syncthreads();
        //Check if its time to update the parent level


        if(j==(2*y_p)) {
            local_patch[threadIdx.z][threadIdx.x][(j) % N ] =  f_p; //initial update
            local_patch[threadIdx.z][threadIdx.x][(j+1) % N ] =  f_p;
        }


        //Check if its time to update current level
        if(j==y_l) {
            local_patch[threadIdx.z][threadIdx.x][j % N ] =  f_l; //initial update
            y_update_flag[j%3]=1;
            y_update_index[j%3] = particle_index_l;
        } else {
            y_update_flag[j%3]=0;
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
            f_p = particle_data_input[particle_index_p];
        }

        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if(y_update_flag[(j-filter_offset+3)%3]==1) {
            //LOCALPATCHUPDATE(particle_data_output,y_update_index[(j+2-filter_offset)%2],threadIdx.z,threadIdx.x,(j+N-filter_offset) % N);
            //particle_data_output[y_update_index[(j+2-filter_offset)%2]] = local_patch[threadIdx.z][threadIdx.x][(j+N-filter_offset) % N];

            LOCALPATCHCONV(particle_data_output,y_update_index[(j+3-filter_offset)%3],threadIdx.z,threadIdx.x,j-2,neighbour_sum);

//            if (not_ghost) {
//                neighbour_sum = 0;
//#pragma unroll
//                for (int q = 0; q < 5; ++q) {
//#pragma unroll
//                    for (int l = 0; l < 5; ++l) {
//
//                        neighbour_sum += local_patch[threadIdx.z + q - 2][threadIdx.x + l - 2][(j - 2 + N - 2) % N]
//                                         + local_patch[threadIdx.z + q - 2][threadIdx.x + l - 2][(j - 2 + N - 1) % N]
//                                         + local_patch[threadIdx.z + q - 2][threadIdx.x + l - 2][(j - 2 + N) % N]
//                                         + local_patch[threadIdx.z + q - 2][threadIdx.x + l - 2][(j - 2 + N + 1) % N]
//                                         + local_patch[threadIdx.z + q - 2][threadIdx.x + l - 2][(j - 2 + N + 2) % N];
//
//                    }

//                    neighbour_sum += local_patch[threadIdx.z + q - 2][threadIdx.x + 0 - 2][(j - 2 + N - 2) % N]
//                                     + local_patch[threadIdx.z + q - 2][threadIdx.x + 0 - 2][(j - 2 + N - 1) % N]
//                                     + local_patch[threadIdx.z + q - 2][threadIdx.x + 0 - 2][(j - 2 + N) % N]
//                                     + local_patch[threadIdx.z + q - 2][threadIdx.x + 0 - 2][(j - 2 + N + 1) % N]
//                                     + local_patch[threadIdx.z + q - 2][threadIdx.x + 0 - 2][(j - 2 + N + 2) % N]
//                    + local_patch[threadIdx.z + q - 2][threadIdx.x + 1 - 2][(j - 2 + N - 2) % N]
//                    + local_patch[threadIdx.z + q - 2][threadIdx.x + 1 - 2][(j - 2 + N - 1) % N]
//                    + local_patch[threadIdx.z + q - 2][threadIdx.x + 1 - 2][(j - 2 + N) % N]
//                    + local_patch[threadIdx.z + q - 2][threadIdx.x + 1 - 2][(j - 2 + N + 1) % N]
//                    + local_patch[threadIdx.z + q - 2][threadIdx.x + 1 - 2][(j - 2 + N + 2) % N]
//                      + local_patch[threadIdx.z + q - 2][threadIdx.x + 2 - 2][(j - 2 + N - 2) % N]
//                      + local_patch[threadIdx.z + q - 2][threadIdx.x + 2 - 2][(j - 2 + N - 1) % N]
//                      + local_patch[threadIdx.z + q - 2][threadIdx.x + 2 - 2][(j - 2 + N) % N]
//                      + local_patch[threadIdx.z + q - 2][threadIdx.x + 2 - 2][(j - 2 + N + 1) % N]
//                      + local_patch[threadIdx.z + q - 2][threadIdx.x + 2 - 2][(j - 2 + N + 2) % N]
//                        + local_patch[threadIdx.z + q - 2][threadIdx.x + 3 - 2][(j - 2 + N - 2) % N]
//                        + local_patch[threadIdx.z + q - 2][threadIdx.x + 3 - 2][(j - 2 + N - 1) % N]
//                        + local_patch[threadIdx.z + q - 2][threadIdx.x + 3 - 2][(j - 2 + N) % N]
//                        + local_patch[threadIdx.z + q - 2][threadIdx.x + 3 - 2][(j - 2 + N + 1) % N]
//                        + local_patch[threadIdx.z + q - 2][threadIdx.x + 3 - 2][(j - 2 + N + 2) % N]
//                          + local_patch[threadIdx.z + q - 2][threadIdx.x + 4 - 2][(j - 2 + N - 2) % N]
//                          + local_patch[threadIdx.z + q - 2][threadIdx.x + 4 - 2][(j - 2 + N - 1) % N]
//                          + local_patch[threadIdx.z + q - 2][threadIdx.x + 4 - 2][(j - 2 + N) % N]
//                          + local_patch[threadIdx.z + q - 2][threadIdx.x + 4 - 2][(j - 2 + N + 1) % N]
//                          + local_patch[threadIdx.z + q - 2][threadIdx.x + 4 - 2][(j - 2 + N + 2) % N];
//
//                }
//            }
//            if (not_ghost) {
//                 particle_data_output[y_update_index[(j+3-filter_offset)%3]] = std::round(neighbour_sum / 27.0f);
//            }


        }

    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N ]=0;
    __syncthreads();

    if(y_update_flag[(y_num-2)%3]==1){ //the last particle (if it exists)

        LOCALPATCHCONV(particle_data_output,particle_index_l,threadIdx.z,threadIdx.x,y_num-2,neighbour_sum);
        //LOCALPATCHUPDATE(particle_data_output,particle_index_l,threadIdx.z,threadIdx.x,(y_num-1) % N);

    }


}

__global__ void shared_update_interior_level(const std::size_t *row_info,
                                             const std::uint16_t *particle_y,
                                             const std::size_t* level_offset,
                                             const std::uint16_t *particle_data_input,
                                             const std::size_t *row_info_child,
                                             const std::uint16_t *particle_y_child,
                                             const std::size_t* level_offset_child,
                                             const std::float_t *particle_data_input_child,
                                             std::uint16_t *particle_data_output,
                                             const std::uint16_t* level_x_num,
                                             const std::uint16_t* level_z_num,
                                             const std::uint16_t* level_y_num,
                                             const std::size_t level)  {
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

    const unsigned int N = 5;
    const unsigned int N_t = N+2;

    __shared__ std::float_t local_patch[12][12][5]; // This is block wise shared memory this is assuming an 8*8 block with pad()

    uint16_t y_cache[N]={0}; // These are local register/private caches
    uint16_t index_cache[N]={0}; // These are local register/private caches

    if(threadIdx.x >= 12){
        return;
    }
    if(threadIdx.z >= 12){
        return;
    }


    int x_index = (8 * blockIdx.x + threadIdx.x - 2);
    int z_index = (8 * blockIdx.z + threadIdx.z - 2);


    bool not_ghost=false;

    if((threadIdx.x > 1) && (threadIdx.x < 10) && (threadIdx.z > 1) && (threadIdx.z < 10)){
        not_ghost = true;
    }


    if((x_index >= x_num) || (x_index < 0)){
        //set the whole buffer to the boundary condition
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][2 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][3 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][4 ] = 0;

        return; //out of bounds
    }

    if((z_index >= z_num) || (z_index < 0)){
        //set the whole buffer to the zero boundary condition
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][2 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][3 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][4 ] = 0;

        return; //out of bounds
    }

    int x_index_p = (8 * blockIdx.x + threadIdx.x - 2)/2;
    int z_index_p = (8 * blockIdx.z + threadIdx.z - 2)/2;


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
    get_row_begin_end(&particle_global_index_begin_p, &particle_global_index_end_p, &current_row_p, row_info);

    std::size_t y_block = 1;
    std::uint16_t y_update_flag[3] = {0};
    std::size_t y_update_index[3] = {0};

    //current level variables
    std::size_t particle_index_l = particle_global_index_begin;
    std::uint16_t y_l= particle_y[particle_index_l];
    std::uint16_t f_l = particle_data_input[particle_index_l];

    /*
    * Parent level variable initialization,
    */

    //parent level variables
    std::size_t particle_index_p = particle_global_index_begin_p;
    std::uint16_t y_p= particle_y[particle_index_p];
    std::uint16_t f_p = particle_data_input[particle_index_p];

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
    std::float_t f_child = particle_data_input_child[particle_index_child];

    if(particle_global_index_begin_child == particle_global_index_end_child){
        y_child = y_num+1;//no particles don't do anything
    }

    if(particle_global_index_begin_p == particle_global_index_end_p){
        y_p = y_num+1;//no particles don't do anything
    }

    if(particle_global_index_begin == particle_global_index_end){
        y_l = y_num+1;//no particles don't do anything
    }

    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N-1)%N] = 0; //this is at (y-1)

    const int filter_offset = 2;
    double neighbour_sum = 0;

    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T

        //Check if its time to update the parent level
        if(j==(2*y_p)) {
            local_patch[threadIdx.z][threadIdx.x][(j) % N ] =  f_p; //initial update
            local_patch[threadIdx.z][threadIdx.x][(j+1) % N ] =  f_p;
        }

        //Check if its time to update child level
        if(j==y_child) {
            local_patch[threadIdx.z][threadIdx.x][y_child % N ] =  f_child; //initial update
        }

        //Check if its time to update current level
        if(j==y_l) {
            local_patch[threadIdx.z][threadIdx.x][y_l % N ] =  f_l; //initial update
            y_update_flag[j%3]=1;
            y_update_index[j%3] = particle_index_l;
        } else {
            y_update_flag[j%3]=0;
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
            f_p = particle_data_input[particle_index_p];
        }


        //update at child level
        if((y_child <= j) && ((particle_index_child+1) <particle_global_index_end_child)){
            particle_index_child++;
            y_child= particle_y_child[particle_index_child];
            f_child = particle_data_input_child[particle_index_child];
        }


        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if(y_update_flag[(j-filter_offset+3)%3]==1){

            //LOCALPATCHUPDATE(particle_data_output,y_update_index[(j+2-filter_offset)%2],threadIdx.z,threadIdx.x,(j+N-filter_offset) % N);
            LOCALPATCHCONV(particle_data_output,y_update_index[(j+3-filter_offset)%3],threadIdx.z,threadIdx.x,j-2,neighbour_sum);
        }
        __syncthreads();

    }

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N ]=0;
    __syncthreads();
    //set the boundary condition (zeros in this case)

    if(y_update_flag[(y_num-2)%3]==1){ //the last particle (if it exists)


        //LOCALPATCHUPDATE(particle_data_output,particle_index_l,threadIdx.z,threadIdx.x,(y_num-1) % N);
        LOCALPATCHCONV(particle_data_output,particle_index_l,threadIdx.z,threadIdx.x,y_num-2,neighbour_sum);

    }




}


__global__ void shared_update_min(const std::size_t *row_info,
                                  const std::uint16_t *particle_y,
                                  const std::size_t* level_offset,
                                  const std::uint16_t *particle_data_input,
                                  const std::size_t *row_info_child,
                                  const std::uint16_t *particle_y_child,
                                  const std::size_t* level_offset_child,
                                  const std::float_t *particle_data_input_child,
                                  std::uint16_t *particle_data_output,
                                  const std::uint16_t* level_x_num,
                                  const std::uint16_t* level_z_num,
                                  const std::uint16_t* level_y_num,
                                  const std::size_t level)  {

    /*
     *
     *  Here we introduce updating Particle Cells at a level below.
     *
     */

    const int x_num = level_x_num[level];
    const int y_num = level_y_num[level];
    const int z_num = level_z_num[level];

    const unsigned int N = 5;

    __shared__ std::float_t local_patch[12][12][5]; // This is block wise shared memory this is assuming an 8*8 block with pad()

    uint16_t y_cache[N]={0}; // These are local register/private caches
    uint16_t index_cache[N]={0}; // These are local register/private caches

    if(threadIdx.x >= 12){
        return;
    }
    if(threadIdx.z >= 12){
        return;
    }


    int x_index = (8 * blockIdx.x + threadIdx.x - 2);
    int z_index = (8 * blockIdx.z + threadIdx.z - 2);


    bool not_ghost=false;

    if((threadIdx.x > 1) && (threadIdx.x < 10) && (threadIdx.z > 1) && (threadIdx.z < 10)){
        not_ghost = true;
    }


    if((x_index >= x_num) || (x_index < 0)){
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][2 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][3 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][4 ] = 0;

        return; //out of bounds
    }

    if((z_index >= z_num) || (z_index < 0)){
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][2 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][3 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][4 ] = 0;

        return; //out of bounds
    }


    /*
     * Current level variable initialization
     *
     */

    std::size_t current_row = level_offset[level] + (x_index) + (z_index)*x_num; // the input to each kernel is its chunk index for which it should iterate over
    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    // current level
    get_row_begin_end(&particle_global_index_begin, &particle_global_index_end, &current_row, row_info);

    std::size_t y_block = 1;
    std::uint16_t y_update_flag[3] = {0};
    std::size_t y_update_index[3] = {0};

    //current level variables
    std::size_t particle_index_l = particle_global_index_begin;
    std::uint16_t y_l= particle_y[particle_index_l];
    std::uint16_t f_l = particle_data_input[particle_index_l];

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
    std::float_t f_child = particle_data_input_child[particle_index_child];



    if(particle_global_index_begin_child == particle_global_index_end_child){
        y_child = y_num+1;//no particles don't do anything
    }

    if(particle_global_index_begin == particle_global_index_end){
        y_l = y_num+1;//no particles don't do anything
    }


    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N-1) % N ] = 0; //this is at (y-1)

    const int filter_offset = 2;
    double neighbour_sum = 0;

    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T

        /*
         *
         * Current Level Update
         *
         */

        __syncthreads();

        //Check if its time to update current level
        if(j==y_l) {
            local_patch[threadIdx.z][threadIdx.x][y_l % N ] =  f_l; //initial update
            y_update_flag[j%3]=1;
            y_update_index[j%3] = particle_index_l;
        } else {
            y_update_flag[j%3]=0;
        }

        //update at current level
        if((y_l <= j) && ((particle_index_l+1) <particle_global_index_end)){
            particle_index_l++;
            y_l= particle_y[particle_index_l];
            f_l = particle_data_input[particle_index_l];
        }

        /*
         *
         * Child Level Update
         *
         */


        //Check if its time to update current level
        if(j==y_child) {
            local_patch[threadIdx.z][threadIdx.x][y_child % N ] =  f_child; //initial update
        }

        //update at current level
        if((y_child <= j) && ((particle_index_child+1) <particle_global_index_end_child)){
            particle_index_child++;
            y_child= particle_y_child[particle_index_child];
            f_child = particle_data_input_child[particle_index_child];
        }


        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if(y_update_flag[(j-filter_offset+3)%3]==1){

            //LOCALPATCHUPDATE(particle_data_output,y_update_index[(j+2-filter_offset)%2],threadIdx.z,threadIdx.x,(j+N-filter_offset) % N);
            LOCALPATCHCONV(particle_data_output,y_update_index[(j+3-filter_offset)%3],threadIdx.z,threadIdx.x,j-2,neighbour_sum);
        }

    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N ]=0;
    __syncthreads();

    if(y_update_flag[(y_num-2)%3]==1){ //the last particle (if it exists)

        //LOCALPATCHUPDATE(particle_data_output,particle_index_l,threadIdx.z,threadIdx.x,(y_num-1) % N);
        LOCALPATCHCONV(particle_data_output,particle_index_l,threadIdx.z,threadIdx.x,y_num-2,neighbour_sum);
    }


}






