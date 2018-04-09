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

#include "APRDownsampleGPU.hpp"

#include "APRIsoConvGPU.hpp"

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



    ExtraParticleData<uint16_t> output_particles(apr);
    output_particles.init_gpu(apr.total_number_particles());

    std::vector<float> conv_stencil;

    ExtraParticleData<uint16_t> tree_temp(aprTree);
    tree_temp.init_gpu(aprTree.total_number_parent_cells());

//    isotropic_convolve_333(apr.particles_intensities,
//                           output_particles,
//                           conv_stencil,
//                           tree_temp);



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
                    conv_min_333 << < blocks_l, threads_l >> >
                                                     (gpuaprAccess.gpu_access.row_global_index,
                                                             gpuaprAccess.gpu_access.y_part_coord,
                                                             gpuaprAccess.gpu_access.level_offsets,
                                                             apr.particles_intensities.gpu_pointer,
                                                             gpuaprAccessTree.gpu_access.row_global_index,
                                                             gpuaprAccessTree.gpu_access.y_part_coord,
                                                             gpuaprAccessTree.gpu_access.level_offsets,
                                                             tree_mean_gpu.gpu_pointer,
                                                             output_particles.gpu_pointer,
                                                             gpuaprAccess.gpu_access.level_x_num,
                                                             gpuaprAccess.gpu_access.level_z_num,
                                                             gpuaprAccess.gpu_access.level_y_num,
                                                             level);

                } else if (level == apr.level_max()) {
                    conv_max_333 << < blocks_l, threads_l >> >
                                                     (gpuaprAccess.gpu_access.row_global_index,
                                                             gpuaprAccess.gpu_access.y_part_coord,
                                                             apr.particles_intensities.gpu_pointer,
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
                                                                        tree_mean_gpu.gpu_pointer,
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

        timer.stop_timer();
    }

    float gpu_iterate_time_si3 = timer.timings.back();
    output_particles.copy_data_to_host();

    output_particles.gpu_data.clear();
    output_particles.gpu_data.shrink_to_fit();


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
    bool success = true;

    uint64_t c_fail= 0;
    uint64_t c_pass= 0;

        c_pass = 0;
    c_fail = 0;
    success=true;
    uint64_t output_c=0;

    for (uint64_t particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        aprIt.set_iterator_to_particle_by_number(particle_number);
        //if(spatial_info_test[aprIt]==(aprIt.x() + aprIt.y() + aprIt.z() + aprIt.level())){
        if(output_particles[aprIt]==output[aprIt]){
            c_pass++;
        } else {

            success = false;
            if(aprIt.level() == aprIt.level_max()) {
                c_fail++;
                if (output_c < 200) {
                    std::cout << "Expected: " << output[aprIt] << " Recieved: " << output_particles[aprIt] << " Level: " << aprIt.level() << " x: " << aprIt.x()
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






#define LOCALPATCHUPDATE(particle_output,index,z,x,j)\
if (not_ghost) {\
    particle_output[index] = local_patch[z][x][j];\
}\




