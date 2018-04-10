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

void create_test_particles(APR<uint16_t>& apr,APRIterator<uint16_t>& apr_iterator,APRTreeIterator<uint16_t>& apr_tree_iterator,ExtraParticleData<uint16_t> &test_particles,ExtraParticleData<uint16_t>& particles,ExtraParticleData<uint16_t>& part_tree,std::vector<float>& stencil, const int stencil_size, const int stencil_half){

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
                    float neigh_sum = 0;
                    int counter = 0;

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

                    test_particles[apr_iterator] = std::roundf(neigh_sum);

                }
            }
        }




        // std::string image_file_name = apr.parameters.input_dir + std::to_string(level_local) + "_by_level.tif";
        //TiffUtils::saveMeshAsTiff(image_file_name, by_level_recon);

    }

}


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

    std::cout << "Number parts: " << aprIt.total_number_particles() << " number in interior tree: " << aprTree.total_number_parent_cells() << std::endl;

    /*
     *  Now launch the kernels across all the chunks determiend by the load balancing
     *
     */


    int number_reps = options.num_rep;


    /*
     *  Off-load the particle data from the GPU
     *
     */


    /*
    *  Test the x,y,z,level information is correct
    *
    */


    apr.particles_intensities.copy_data_to_gpu();


    std::vector<float> stencil;
    //stencil.resize(125,.30f);
    stencil.resize(125,1.0/125.0f);

    ExtraParticleData<uint16_t> output_particles(apr);
    output_particles.init_gpu(apr.total_number_particles());

    ExtraParticleData<uint16_t> tree_temp(aprTree);
    tree_temp.init_gpu(aprTree.total_number_parent_cells());

    APRIsoConvGPU isoConvGPU(apr,aprTree);

    cudaDeviceSynchronize();
    for (int i = 0; i < 2; ++i) {

        timer.start_timer("summing the sptial informatino for each partilce on the GPU");
        for (int rep = 0; rep < number_reps; ++rep) {

            isoConvGPU.isotropic_convolve_555(apr,apr.particles_intensities,
                                              output_particles,
                                              stencil,
                                              tree_temp);

            cudaDeviceSynchronize();
        }

        timer.stop_timer();
    }

    float gpu_iterate_time_si3 = timer.timings.back();
    output_particles.copy_data_to_host();

    output_particles.gpu_data.clear();
    output_particles.gpu_data.shrink_to_fit();

    tree_temp.copy_data_to_host();

    tree_temp.gpu_data.clear();
    tree_temp.gpu_data.shrink_to_fit();

    std::cout << "Average time for loop insert max: " << (gpu_iterate_time_si3/(number_reps*1.0f))*1000 << " ms" << std::endl;
    std::cout << "Average time for loop insert max per million particles: " << (gpu_iterate_time_si3/(apr.total_number_particles()*number_reps*1.0f))*1000*1000000.0f << " ms" << std::endl;

    //////////////////////////
    ///
    /// Now check the data
    ///
    ////////////////////////////

    ExtraParticleData<uint16_t> output(apr);
    create_test_particles( apr, aprIt,treeIt,output,apr.particles_intensities,tree_temp,stencil, 5, 2);

    uint64_t c_pass = 0;
    uint64_t c_fail = 0;
    bool success=true;
    uint64_t output_c=0;

    for (uint64_t particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        aprIt.set_iterator_to_particle_by_number(particle_number);
        //if(spatial_info_test[aprIt]==(aprIt.x() + aprIt.y() + aprIt.z() + aprIt.level())){
        if((output_particles[aprIt]-output[aprIt])<=1){
            c_pass++;
        } else {
            c_fail++;
            success = false;
            //if(aprIt.level() >= aprIt.level_max()) {
                if (output_c < 1000) {
                    std::cout << "Expected: " << output[aprIt] << " Recieved: " << output_particles[aprIt] << " Level: " << aprIt.level() << " x: " << aprIt.x()
                              << " z: " << aprIt.z() << " y: " << aprIt.y() << std::endl;
                    output_c++;
                }
                //spatial_info_test3[aprIt] = 0;
            //}

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



#define LOCALPATCHUPDATE(particle_output,index,z,x,j)\
if (not_ghost) {\
    particle_output[index] = local_patch[z][x][j];\
}\






