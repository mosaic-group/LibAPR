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


__global__ void shared_update(const std::size_t *row_info,
                              const std::uint16_t *particle_y,
                              const std::uint16_t *particle_data_input,
                              std::uint16_t *particle_data_output,
                              const std::size_t offset,
                              const std::size_t x_num,
                              const std::size_t z_num,
                              const std::size_t y_num,
                              const std::size_t level);

__global__ void shared_update_conv(const std::size_t *row_info,
                                   const std::uint16_t *particle_y,
                                   const std::uint16_t *particle_data_input,
                                   std::uint16_t *particle_data_output,
                                   const std::size_t offset,
                                   const std::size_t x_num,
                                   const std::size_t z_num,
                                   const std::size_t y_num,
                                   const std::size_t level);


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

__global__ void down_sample_avg_interior(const std::size_t *row_info,
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

//    timer.start_timer("generate ds particles");
//    ExtraParticleData<float> ds_parts =  meanDownsamplingOld(apr, aprTree);
//    timer.stop_timer();

    std::cout << "Number parts: " << aprIt.total_number_particles() << " number in interior tree: " << aprTree.total_number_parent_cells() << std::endl;

//
//    ds_parts.copy_data_to_gpu();


    int number_reps = options.num_rep;




    ExtraParticleData<float> tree_mean_gpu(aprTree);
    tree_mean_gpu.init_gpu(aprTree.total_number_parent_cells());

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


    ExtraParticleData<uint16_t> spatial_info_test3(apr);
    spatial_info_test3.init_gpu(apr.total_number_particles());

    cudaDeviceSynchronize();
    for (int i = 0; i < 2; ++i) {

        timer.start_timer("summing the sptial informatino for each partilce on the GPU");
        for (int rep = 0; rep < number_reps; ++rep) {

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

                    down_sample_avg << < blocks_l, threads_l >> >
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


                } else {

                    down_sample_avg_interior<< < blocks_l, threads_l >> >
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
                    shared_update_min << < blocks_l, threads_l >> >
                                                     (gpuaprAccess.gpu_access.row_global_index,
                                                             gpuaprAccess.gpu_access.y_part_coord,
                                                             gpuaprAccess.gpu_access.level_offsets,
                                                             apr.particles_intensities.gpu_pointer,
                                                             gpuaprAccessTree.gpu_access.row_global_index,
                                                             gpuaprAccessTree.gpu_access.y_part_coord,
                                                             gpuaprAccessTree.gpu_access.level_offsets,
                                                             tree_mean_gpu.gpu_pointer,
                                                             spatial_info_test3.gpu_pointer,
                                                             gpuaprAccess.gpu_access.level_x_num,
                                                             gpuaprAccess.gpu_access.level_z_num,
                                                             gpuaprAccess.gpu_access.level_y_num,
                                                             level);

                } else if (level == apr.level_max()) {
                    shared_update_max << < blocks_l, threads_l >> >
                                                     (gpuaprAccess.gpu_access.row_global_index, gpuaprAccess.gpu_access.y_part_coord, apr.particles_intensities.gpu_pointer, spatial_info_test3.gpu_pointer, gpuaprAccess.gpu_access.level_offsets, gpuaprAccess.gpu_access.level_x_num, gpuaprAccess.gpu_access.level_z_num, gpuaprAccess.gpu_access.level_y_num, level);



                } else {
                    shared_update_interior_level << < blocks_l, threads_l >> >
                                                                (gpuaprAccess.gpu_access.row_global_index,
                                                                        gpuaprAccess.gpu_access.y_part_coord,
                                                                        gpuaprAccess.gpu_access.level_offsets,
                                                                        apr.particles_intensities.gpu_pointer,
                                                                        gpuaprAccessTree.gpu_access.row_global_index,
                                                                        gpuaprAccessTree.gpu_access.y_part_coord,
                                                                        gpuaprAccessTree.gpu_access.level_offsets,
                                                                        tree_mean_gpu.gpu_pointer,
                                                                        spatial_info_test3.gpu_pointer,
                                                                        gpuaprAccess.gpu_access.level_x_num,
                                                                        gpuaprAccess.gpu_access.level_z_num,
                                                                        gpuaprAccess.gpu_access.level_y_num,
                                                                        level);
                }
                cudaDeviceSynchronize();

            }

            cudaDeviceSynchronize();
        }

        timer.stop_timer();
    }

    float gpu_iterate_time_si3 = timer.timings.back();
    spatial_info_test3.copy_data_to_host();

    spatial_info_test3.gpu_data.clear();
    spatial_info_test3.gpu_data.shrink_to_fit();


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
    std::vector<double> stencil;
    stencil.resize(27,1);
    ExtraParticleData<float> output(apr);
    create_test_particles( apr, aprIt,treeIt,output,apr.particles_intensities,tree_mean_gpu,stencil, 3, 1);


        c_pass = 0;
    c_fail = 0;
    success=true;
    uint64_t output_c=0;

    for (uint64_t particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        aprIt.set_iterator_to_particle_by_number(particle_number);
        //if(spatial_info_test[aprIt]==(aprIt.x() + aprIt.y() + aprIt.z() + aprIt.level())){
        if(spatial_info_test3[aprIt]==output[aprIt]){
            c_pass++;
        } else {

            success = false;
            if(aprIt.level() == aprIt.level_max()) {
                c_fail++;
                if (output_c < 200) {
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

#define LOCALPATCHUPDATE(particle_output,index,z,x,j)\
if (not_ghost) {\
    particle_output[index] = local_patch[z][x][j];\
}\

#define LOCALPATCHCONV(particle_output,index,z,x,y,neighbour_sum)\
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

    const int z_num = level_z_num[level];

    const int x_num_p = level_x_num[level-1];
    const int y_num_p = level_y_num[level-1];
    const int z_num_p = level_z_num[level-1];



     // This is block wise shared memory this is assuming an 8*8 block with pad()


    if(threadIdx.x >= 10){
        return;
    }
    if(threadIdx.z >= 10){
        return;
    }



    bool not_ghost=false;

    if((threadIdx.x > 0) && (threadIdx.x < 9) && (threadIdx.z > 0) && (threadIdx.z < 9)){
        not_ghost = true;
    }

    int x_index = (8 * blockIdx.x + threadIdx.x - 1);
    int z_index = (8 * blockIdx.z + threadIdx.z - 1);


    __shared__ std::uint16_t local_patch[10][10][4];

    if((x_index >= x_num) || (x_index < 0)){
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][2 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][3 ] = 0;

        return; //out of bounds
    }

    if((z_index >= z_num) || (z_index < 0)){
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][2 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][3 ] = 0;
        return; //out of bounds
    }


    std::size_t particle_global_index_begin;
    std::size_t particle_global_index_end;

    std::size_t particle_global_index_begin_p;
    std::size_t particle_global_index_end_p;

    // current level
    std::size_t current_row = level_offset[level] + (x_index) + (z_index)*x_num; // the input to each kernel is its chunk index for which it should iterate over
    get_row_begin_end(&particle_global_index_begin, &particle_global_index_end, current_row, row_info);
    std::size_t particle_index_l = particle_global_index_begin;
    std::uint16_t y_l= particle_y[particle_index_l];
    std::uint16_t f_l = particle_data_input[particle_index_l];

    int x_index_p = (x_index)/2;
    int z_index_p = (z_index)/2;
    std::size_t current_row_p = level_offset[level-1] + (x_index_p) + (z_index_p)*x_num_p; // the input to each kernel is its chunk index for which it should iterate over
    // parent level, level - 1, one resolution lower (coarser)
    get_row_begin_end(&particle_global_index_begin_p, &particle_global_index_end_p, current_row_p, row_info);

    //parent level variables
    std::size_t particle_index_p = particle_global_index_begin_p;
    std::uint16_t y_p= particle_y[particle_index_p];
    std::uint16_t f_p = particle_data_input[particle_index_p];



    //current level variables


    const int y_num = level_y_num[level];
    if(particle_global_index_begin_p == particle_global_index_end_p){
        y_p = y_num+1;//no particles don't do anything
    }

    if(particle_global_index_begin == particle_global_index_end){
        y_l = y_num+1;//no particles don't do anything
    }

    const unsigned int N = 4;
    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N-1) % N ] = 0; //this is at (y-1)

    const int filter_offset = 1;

    __shared__ std::uint16_t y_update_flag[10][10][2];
    __shared__ std::size_t y_update_index[10][10][2];

    y_update_flag[threadIdx.z][threadIdx.x][0] = 0;
    y_update_flag[threadIdx.z][threadIdx.x][1] = 0;

    //std::uint16_t y_update_flag[2] = {0};
    //std::size_t y_update_index[2] = {0};



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
            y_update_flag[threadIdx.z][threadIdx.x][j%2]=1;
            y_update_index[threadIdx.z][threadIdx.x][j%2] = particle_index_l;
        } else {
            y_update_flag[threadIdx.z][threadIdx.x][j%2]=0;
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

        if(y_update_flag[threadIdx.z][threadIdx.x][(j-filter_offset+2)%2]==1){
            //LOCALPATCHUPDATE(particle_data_output,y_update_index[(j+2-filter_offset)%2],threadIdx.z,threadIdx.x,(j+N-filter_offset) % N);
                //particle_data_output[y_update_index[(j+2-filter_offset)%2]] = local_patch[threadIdx.z][threadIdx.x][(j+N-filter_offset) % N];
            float neighbour_sum = 0;
            LOCALPATCHCONV(particle_data_output,y_update_index[threadIdx.z][threadIdx.x][(j+2-filter_offset)%2],threadIdx.z,threadIdx.x,j-1,neighbour_sum);

        }

    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N ]=0;
    __syncthreads();

    if(y_update_flag[threadIdx.z][threadIdx.x][(y_num-1)%2]==1){ //the last particle (if it exists)
        float neighbour_sum = 0;
        LOCALPATCHCONV(particle_data_output,particle_index_l,threadIdx.z,threadIdx.x,y_num-1,neighbour_sum);
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

    const unsigned int N = 4;
    const unsigned int N_t = N+2;

    __shared__ std::float_t local_patch[10][10][4]; // This is block wise shared memory this is assuming an 8*8 block with pad()


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
        //set the whole buffer to the boundary condition
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][2 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][3 ] = 0;

        return; //out of bounds
    }

    if((z_index >= z_num) || (z_index < 0)){
        //set the whole buffer to the zero boundary condition
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][2 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][3 ] = 0;
        return; //out of bounds
    }

    int x_index_p = (8 * blockIdx.x + threadIdx.x - 1)/2;
    int z_index_p = (8 * blockIdx.z + threadIdx.z - 1)/2;


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
    get_row_begin_end(&particle_global_index_begin, &particle_global_index_end, current_row, row_info);
    // parent level, level - 1, one resolution lower (coarser)
    get_row_begin_end(&particle_global_index_begin_p, &particle_global_index_end_p, current_row_p, row_info);

    std::size_t y_block = 1;
    std::uint16_t y_update_flag[2] = {0};
    std::size_t y_update_index[2] = {0};

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

    get_row_begin_end(&particle_global_index_begin_child, &particle_global_index_end_child, current_row_child, row_info_child);

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

    const int filter_offset = 1;
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
            y_update_flag[j%2]=1;
            y_update_index[j%2] = particle_index_l;
        } else {
            y_update_flag[j%2]=0;
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

        if(y_update_flag[(j-filter_offset+2)%2]==1){

            //LOCALPATCHUPDATE(particle_data_output,y_update_index[(j+2-filter_offset)%2],threadIdx.z,threadIdx.x,(j+N-filter_offset) % N);
            LOCALPATCHCONV(particle_data_output,y_update_index[(j+2-filter_offset)%2],threadIdx.z,threadIdx.x,j-1,neighbour_sum);
        }
        __syncthreads();

    }

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N ]=0;
    __syncthreads();
    //set the boundary condition (zeros in this case)

    if(y_update_flag[(y_num-1)%2]==1){ //the last particle (if it exists)


        //LOCALPATCHUPDATE(particle_data_output,particle_index_l,threadIdx.z,threadIdx.x,(y_num-1) % N);
        LOCALPATCHCONV(particle_data_output,particle_index_l,threadIdx.z,threadIdx.x,y_num-1,neighbour_sum);

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

    const unsigned int N = 4;

    __shared__ std::float_t local_patch[10][10][4]; // This is block wise shared memory this is assuming an 8*8 block with pad()

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
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][2 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][3 ] = 0;

        return; //out of bounds
    }

    if((z_index >= z_num) || (z_index < 0)){
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][2 ] = 0;
        local_patch[threadIdx.z][threadIdx.x][3 ] = 0;

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
    get_row_begin_end(&particle_global_index_begin, &particle_global_index_end, current_row, row_info);

    std::size_t y_block = 1;
    std::uint16_t y_update_flag[2] = {0};
    std::size_t y_update_index[2] = {0};

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

    get_row_begin_end(&particle_global_index_begin_child, &particle_global_index_end_child, current_row_child, row_info_child);

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

    const int filter_offset = 1;
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
            y_update_flag[j%2]=1;
            y_update_index[j%2] = particle_index_l;
        } else {
            y_update_flag[j%2]=0;
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

        if(y_update_flag[(j-filter_offset+2)%2]==1){

            //LOCALPATCHUPDATE(particle_data_output,y_update_index[(j+2-filter_offset)%2],threadIdx.z,threadIdx.x,(j+N-filter_offset) % N);
            LOCALPATCHCONV(particle_data_output,y_update_index[(j+2-filter_offset)%2],threadIdx.z,threadIdx.x,j-1,neighbour_sum);
        }

    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N ]=0;
    __syncthreads();

    if(y_update_flag[(y_num-1)%2]==1){ //the last particle (if it exists)

        //LOCALPATCHUPDATE(particle_data_output,particle_index_l,threadIdx.z,threadIdx.x,(y_num-1) % N);
        LOCALPATCHCONV(particle_data_output,particle_index_l,threadIdx.z,threadIdx.x,y_num-1,neighbour_sum);
    }


}


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


    // const int y_num = level_y_num[level];
    // const int z_num = level_z_num[level];


    const int y_num_p = level_y_num[level-1];
    // const int z_num_p = level_z_num[level-1];

    const int x_index = (2 * blockIdx.x + threadIdx.x/64);
    const int z_index = (2 * blockIdx.z + ((threadIdx.x)/32)%2);


    //int x_index_p = blockIdx.x;
    //int z_index_p = blockIdx.z;

    const int block = threadIdx.x/32;
    const int local_th = (threadIdx.x%32);


    float scale_factor_xz = (((2*level_x_num[level-1] != level_x_num[level]) && blockIdx.x==(level_x_num[level-1]-1) ) + ((2*level_z_num[level-1] != level_z_num[level]) && blockIdx.z==(level_z_num[level-1]-1) ))*2;

    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    float scale_factor_yxz = scale_factor_xz;

    if((2*level_y_num[level-1] != level_y_num[level])){
        scale_factor_yxz = scale_factor_xz*2;
    }

//    std::size_t row_index_p =blockIdx.x + blockIdx.z*level_x_num[level-1] + level_offset_child[level-1];
//
//
//    std::size_t row_index =x_index + z_index*level_x_num[level] + level_offset[level];

    std::size_t global_index_begin_0;
    std::size_t global_index_end_0;

    std::size_t global_index_begin_p;
    std::size_t global_index_end_p;

    //remove these with registers
    //__shared__ float f_cache[5][32];
    //__shared__ int y_cache[5][32];

    //keep these
    __shared__ float parent_cache[8][16];


    float current_val = 0;

    //initialization to zero
    //f_cache[block][local_th]=0;
    //y_cache[block][local_th]=-1;


    parent_cache[2*block][local_th/2]=0;
    parent_cache[2*block+1][local_th/2]=0;

    int current_y=-1;
    int current_y_p=-1;
    //ying printf("hello begin %d end %d chunks %d number parts %d \n",(int) global_index_begin_0,(int) global_index_end_f, (int) number_chunk, (int) number_parts);


    if((x_index >= level_x_num[level]) || (z_index >= level_z_num[level]) ){

        global_index_begin_0 = 1;
        global_index_end_0 = 0;

        // return; //out of bounds
    } else {
        get_row_begin_end(&global_index_begin_0, &global_index_end_0, x_index + z_index*level_x_num[level] + level_offset[level], row_info);


    }

    get_row_begin_end(&global_index_begin_p, &global_index_end_p, blockIdx.x + blockIdx.z*level_x_num[level-1] + level_offset_child[level-1], row_info_child);



    const std::uint16_t number_y_chunk = (level_y_num[level]+31)/32;


    //initialize (i=0)
    if ((global_index_begin_0 + local_th) < global_index_end_0) {
        current_val = particle_data_input[global_index_begin_0 + local_th];

        //y_cache[block][local_th] = particle_y[ global_index_begin_0 + local_th];
        current_y =  particle_y[ global_index_begin_0 + local_th];
    }


    if (block == 3) {

        if (( global_index_begin_p + local_th) < global_index_end_p) {

            //y_cache[4][local_th] = particle_y_child[ global_index_begin_p + local_th];
            current_y_p = particle_y_child[ global_index_begin_p + local_th];

        }

    }

    //current_y = y_cache[block][local_th ];
    //current_y_p = y_cache[4][local_th ];


    uint16_t sparse_block = 0;
    int sparse_block_p =0;

    for (int y_block = 0; y_block < number_y_chunk; ++y_block) {

        __syncthreads();
        //value less then current chunk then update.
        if (current_y < y_block * 32) {
            sparse_block++;
            if ((sparse_block * 32 + global_index_begin_0 + local_th) < global_index_end_0) {
                current_val = particle_data_input[sparse_block * 32 + global_index_begin_0 +
                                                  local_th];

                current_y = particle_y[sparse_block * 32 + global_index_begin_0 + local_th];
            }

        }

        //current_y = y_cache[block][local_th];
        __syncthreads();


        //update the down-sampling caches
        if ((current_y < (y_block + 1) * 32) && (current_y >= (y_block) * 32)) {

            parent_cache[2*block+current_y%2][(current_y/2) % 16] = (1.0/8.0f)*current_val;
            //parent_cache[2*block+current_y%2][(current_y/2) % 16] = 1;

        }

        __syncthreads();
        //fetch the parent particle data
        if (block == 3) {
            if (current_y_p < ((y_block * 32)/2)) {
                sparse_block_p++;


                if ((sparse_block_p * 32 + global_index_begin_p + local_th) < global_index_end_p) {

                    current_y_p = particle_y_child[sparse_block_p * 32 + global_index_begin_p + local_th];

                }

            }


        }
        __syncthreads();

        if(block ==3) {
            //output

            if (current_y_p < ((y_block+1) * 32)/2) {
                if ((sparse_block_p * 32 + global_index_begin_p + local_th) < global_index_end_p) {

                    if(current_y_p == (y_num_p-1)) {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p + local_th] =
                                scale_factor_yxz*( parent_cache[0][current_y_p % 16] + parent_cache[1][current_y_p % 16] +
                                                   parent_cache[2][current_y_p % 16]
                                                   + parent_cache[3][current_y_p % 16] + parent_cache[4][current_y_p % 16] +
                                                   parent_cache[5][current_y_p % 16] + parent_cache[6][current_y_p % 16] +
                                                   parent_cache[7][current_y_p % 16]);



                    } else {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p + local_th] =
                                scale_factor_xz*( parent_cache[0][current_y_p % 16] + parent_cache[1][current_y_p % 16] +
                                                  parent_cache[2][current_y_p % 16]
                                                  + parent_cache[3][current_y_p % 16] + parent_cache[4][current_y_p % 16] +
                                                  parent_cache[5][current_y_p % 16] + parent_cache[6][current_y_p % 16] +
                                                  parent_cache[7][current_y_p % 16]);


                    }
                }
            }

        }
        __syncthreads();
        parent_cache[2*block][local_th/2] = 0;
        parent_cache[2*block+1][local_th/2] = 0;

    }



}

__global__ void down_sample_avg_interior(const std::size_t *row_info,
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
    //
    //  This step is required for the interior down-sampling
    //

//    const int x_num = level_x_num[level];
//    const int y_num = level_y_num[level];
//    const int z_num = level_z_num[level];
//
//    const int x_num_p = level_x_num[level-1];
    const int y_num_p = level_y_num[level-1];
//    const int z_num_p = level_z_num[level-1];

    int x_index = (2 * blockIdx.x + threadIdx.x/64);
    int z_index = (2 * blockIdx.z + ((threadIdx.x)/32)%2);

//    int x_index_p = blockIdx.x;
//    int z_index_p = blockIdx.z;

    float scale_factor_xz = (((2*level_x_num[level-1] != level_x_num[level]) && blockIdx.x==(level_x_num[level-1]-1) ) + ((2*level_z_num[level-1] != level_z_num[level]) && blockIdx.z==(level_z_num[level-1]-1) ))*2;

    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    float scale_factor_yxz = scale_factor_xz;

    if((2*level_y_num[level-1] != level_y_num[level])){
        scale_factor_yxz = scale_factor_xz*2;
    }


    const int block = threadIdx.x/32;
    const int local_th = (threadIdx.x%32);


    //std::size_t row_index_p =blockIdx.x + blockIdx.z*level_x_num[level-1] + level_offset_child[level-1];

    // std::size_t row_index =x_index + z_index*level_x_num[level] + level_offset[level];
    // std::size_t row_index_t =x_index + z_index*level_x_num[level] + level_offset_child[level];

    //Particles
    std::size_t global_index_begin_0;
    std::size_t global_index_end_0;

    //Parent Tree Particle Cells
    std::size_t global_index_begin_p;
    std::size_t global_index_end_p;

    //Interior Tree Particle Cells
    std::size_t global_index_begin_t;
    std::size_t global_index_end_t;

    //shared memory caches
    // __shared__ float f_cache[5][32];
    //__shared__ int y_cache[5][32];

    // __shared__ float f_cache_t[4][32];
    // __shared__ int y_cache_t[4][32];

    __shared__ float parent_cache[8][16];

    //initialization to zero
//    f_cache[block][local_th]=0;
//    y_cache[block][local_th]=-1;
//
//    y_cache_t[block][local_th]=-1;
//    f_cache_t[block][local_th]=0;

    //  if(block==0){
//        f_cache[4][local_th]=0;
//        y_cache[4][local_th]=-1;

    // }

    parent_cache[2*block][local_th/2]=0;
    parent_cache[2*block+1][local_th/2]=0;

    int current_y=-1;
    int current_y_p=-1;
    int current_y_t=-1;
    float current_val=0;
    float current_val_t = 0;

    if((x_index >= level_x_num[level]) || (z_index >= level_z_num[level]) ){

        global_index_begin_0 = 1;
        global_index_end_0 = 0;

        global_index_begin_t = 1;
        global_index_end_t = 0;

        // return; //out of bounds
    } else {
        get_row_begin_end(&global_index_begin_t, &global_index_end_t, x_index + z_index*level_x_num[level] + level_offset_child[level], row_info_child);
        get_row_begin_end(&global_index_begin_0, &global_index_end_0, x_index + z_index*level_x_num[level] + level_offset[level], row_info);

    }

    get_row_begin_end(&global_index_begin_p, &global_index_end_p, blockIdx.x + blockIdx.z*level_x_num[level-1] + level_offset_child[level-1], row_info_child);

    const std::uint16_t number_y_chunk = (level_y_num[level]+31)/32;


    //initialize (i=0)
    if ((global_index_begin_0 + local_th) < global_index_end_0) {

        //y_cache[block][local_th] = particle_y[global_index_begin_0 + local_th];
        current_y = particle_y[global_index_begin_0 + local_th];

        //f_cache[block][local_th] = particle_data_input[global_index_begin_0 + local_th];
        current_val = particle_data_input[global_index_begin_0 + local_th];

    }

    //tree interior
    if ((global_index_begin_t + local_th) < global_index_end_t) {

        //y_cache_t[block][local_th] = particle_y_child[global_index_begin_t + local_th];
        current_y_t = particle_y_child[global_index_begin_t + local_th];

        //f_cache_t[block][local_th] = particle_data_output[global_index_begin_t + local_th];
        current_val_t = particle_data_output[global_index_begin_t + local_th];

        // current_y_t = y_cache_t[block][local_th ];

    }




    if (block == 3) {

        if (( global_index_begin_p + local_th) < global_index_end_p) {

            current_y_p = particle_y_child[ global_index_begin_p + local_th];

        }

    }

    uint16_t sparse_block = 0;
    int sparse_block_p =0;
    int sparse_block_t =0;

    float local_sum = 0;



    for (int y_block = 0; y_block < (number_y_chunk); ++y_block) {

        __syncthreads();
        //value less then current chunk then update.
        if (current_y < y_block * 32) {
            sparse_block++;
            if ((sparse_block * 32 + global_index_begin_0 + local_th) < global_index_end_0) {
                //f_cache[block][local_th] = particle_data_input[sparse_block * 32 + global_index_begin_0 +
                //   local_th];
                current_val = particle_data_input[sparse_block * 32 + global_index_begin_0 +
                                                  local_th];

                //y_cache[block][local_th] = particle_y[sparse_block * 32 + global_index_begin_0 + local_th];
                current_y = particle_y[sparse_block * 32 + global_index_begin_0 + local_th];
            }

        }
        //current_y = y_cache[block][local_th];

        //interior tree update
        if (current_y_t < y_block * 32) {
            sparse_block_t++;
            if ((sparse_block_t * 32 + global_index_begin_t + local_th) < global_index_end_t) {

                //f_cache_t[block][local_th] = particle_data_output[sparse_block_t * 32 + global_index_begin_t +
                //                                            local_th];
                current_val_t = particle_data_output[sparse_block_t * 32 + global_index_begin_t +
                                                     local_th];

                //y_cache_t[block][local_th] = particle_y_child[sparse_block_t * 32 + global_index_begin_t + local_th];
                current_y_t = particle_y_child[sparse_block_t * 32 + global_index_begin_t + local_th];


            }

        }
        // current_y_t = y_cache_t[block][local_th];


        __syncthreads();
        //update the down-sampling caches
        if ((current_y < (y_block + 1) * 32) && (current_y >= (y_block) * 32)) {

            parent_cache[2*block+current_y%2][(current_y/2) % 16] = (1.0/8.0f)*current_val;
            //parent_cache[2*block+current_y%2][(current_y/2) % 16] = 1;

        }
        __syncthreads();



        //now the interior tree nodes
        if ((current_y_t < (y_block + 1) * 32) && (current_y_t >= (y_block) * 32)) {

            parent_cache[2*block+current_y_t%2][(current_y_t/2) % 16] = (1.0/8.0f)*current_val_t;
            //parent_cache[2*block+current_y_t%2][(current_y_t/2) % 16] =1;
            //parent_cache[0][(current_y_t/2) % 16] = current_y_t/2;


        }
        __syncthreads();


        if (block == 3) {


            if (current_y_p < ((y_block * 32)/2)) {
                sparse_block_p++;


                if ((sparse_block_p * 32 + global_index_begin_p + local_th) < global_index_end_p) {

                    //y_cache[4][local_th] = particle_y_child[sparse_block_p * 32 + global_index_begin_p + local_th];
                    current_y_p = particle_y_child[sparse_block_p * 32 + global_index_begin_p + local_th];

                }

            }

        }



        __syncthreads();


        //local_sum


        if(block ==3) {
            //output
            //current_y_p = y_cache[4][local_th];
            current_y_p = particle_y_child[sparse_block_p * 32 + global_index_begin_p + local_th];

            if (current_y_p < ((y_block+1) * 32)/2 && current_y_p >= ((y_block) * 32)/2) {
                if ((sparse_block_p * 32 + global_index_begin_p + local_th) < global_index_end_p) {



                    if (current_y_p == (y_num_p - 1)) {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p + local_th] =
                                scale_factor_yxz *
                                (parent_cache[0][current_y_p % 16] + parent_cache[1][current_y_p % 16] +
                                 parent_cache[2][current_y_p % 16] + parent_cache[3][current_y_p % 16] +
                                 parent_cache[4][current_y_p % 16] + parent_cache[5][current_y_p % 16] +
                                 parent_cache[6][current_y_p % 16] + parent_cache[7][current_y_p % 16]);


                    } else {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p + local_th] =
                                scale_factor_xz *
                                (parent_cache[0][current_y_p % 16] + parent_cache[1][current_y_p % 16] +
                                 parent_cache[2][current_y_p % 16] + parent_cache[3][current_y_p % 16] +
                                 parent_cache[4][current_y_p % 16] + parent_cache[5][current_y_p % 16] +
                                 parent_cache[6][current_y_p % 16] + parent_cache[7][current_y_p % 16]);


                    }


                }
            }

        }

        __syncthreads();

        parent_cache[2*block][local_th/2] = 0;
        parent_cache[2*block+1][local_th/2] = 0;


    }



}



