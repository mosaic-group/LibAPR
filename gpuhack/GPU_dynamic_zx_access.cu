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

#define KEY_EMPTY_MASK ((((uint64_t)1) << 1) - 1) << 0 //first bit stores if the row is empty or not can be used to avoid computations and accessed using &key
#define KEY_EMPTY_SHIFT 0

#define KEY_X_MASK ((((uint64_t)1) << 16) - 1) << 1
#define KEY_X_SHIFT 1

#define KEY_Z_MASK ((((uint64_t)1) << 16) - 1) << 17
#define KEY_Z_SHIFT 17

#define KEY_LEVEL_MASK ((((uint64_t)1) << 8) - 1) << 33
#define KEY_LEVEL_SHIFT 33



uint64_t encode_xzl(uint16_t x,uint16_t z,uint8_t level,bool nonzero){

    uint64_t raw_key=0;

    raw_key |= ((uint64_t)x << KEY_X_SHIFT);
    raw_key |= ((uint64_t)z << KEY_Z_SHIFT);
    raw_key |= ((uint64_t)level << KEY_LEVEL_SHIFT);

    if(nonzero){
        raw_key |= (1 << KEY_EMPTY_SHIFT);
    } else {
        raw_key |= (0 << KEY_EMPTY_SHIFT);
    }


    uint64_t output_x = (raw_key & KEY_X_MASK) >> KEY_X_SHIFT;
    uint64_t output_z = (raw_key & KEY_Z_MASK) >> KEY_Z_SHIFT;
    uint64_t output_level = (raw_key & KEY_LEVEL_MASK) >> KEY_LEVEL_SHIFT;
    uint64_t output_nz = (raw_key & KEY_EMPTY_MASK) >> KEY_EMPTY_SHIFT;

    uint64_t short_nz = raw_key&1;

    return raw_key;

}

bool decode_xzl(std::uint64_t raw_key,uint16_t& output_x,uint16_t& output_z,uint8_t& output_level){


    output_x = (raw_key & KEY_X_MASK) >> KEY_X_SHIFT;
    output_z = (raw_key & KEY_Z_MASK) >> KEY_Z_SHIFT;
    output_level = (raw_key & KEY_LEVEL_MASK) >> KEY_LEVEL_SHIFT;


    return raw_key&1;

}



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


void create_test_particles_surya(APR<uint16_t>& apr,APRIterator<uint16_t>& apr_iterator,ExtraParticleData<float> &test_particles,ExtraParticleData<uint16_t>& particles,std::vector<float>& stencil, const int stencil_size,
                                 const int stencil_half);


__global__ void load_balance_xzl(const thrust::tuple<std::size_t,std::size_t>* row_info,std::size_t*  _chunk_index_end,
                                 std::size_t total_number_chunks,std::float_t parts_per_block,std::size_t total_number_rows);

__global__ void test_dynamic_balance(const thrust::tuple<std::size_t,std::size_t>* row_info,std::size_t*  _chunk_index_end,
                                     std::size_t total_number_chunks,const std::uint16_t* particle_y,std::uint16_t* particle_data_output);

__global__ void test_dynamic_balance_XZYL(const thrust::tuple<std::size_t,std::size_t>* row_info,std::size_t*  _chunk_index_end,
                                          std::size_t total_number_chunks,const std::uint16_t* particle_y,std::uint16_t* particle_data_output);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    // Read provided APR file
    cmdLineOptions options = read_command_line_options(argc, argv);
    const int reps = 100;

    std::string fileName = options.directory + options.input;
    APR<uint16_t> apr;
    apr.read_apr(fileName);

    // Get dense representation of APR
    APRIterator<uint16_t> aprIt(apr);

#ifdef APR_USE_CUDA
    std::cout << "hello" << std::endl;
#endif



    ///////////////////////////
    ///
    /// Sparse Data for GPU
    ///
    ///////////////////////////

    std::vector<std::tuple<std::size_t,std::size_t>> level_zx_index_start;//size = number of rows on all levels
    std::vector<std::uint16_t> y_explicit;y_explicit.reserve(aprIt.total_number_particles());//size = number of particles
    std::vector<std::uint16_t> particle_values;particle_values.reserve(aprIt.total_number_particles());//size = number of particles
    std::vector<std::size_t> level_offset(aprIt.level_max()+1,UINT64_MAX);//size = number of levels


    std::size_t x = 0;
    std::size_t z = 0;

    std::size_t zx_counter = 0;
    std::size_t pcounter = 0;


    uint64_t bundle_xzl=0;

    APRTimer timer;
    timer.verbose_flag = true;

    timer.start_timer("initialize structure");

    for (int level = aprIt.level_min(); level <= aprIt.level_max(); ++level) {
        level_offset[level] = zx_counter;

        for (z = 0; z < aprIt.spatial_index_z_max(level); ++z) {
            for (x = 0; x < aprIt.spatial_index_x_max(level); ++x) {

                zx_counter++;
                uint64_t key;
                if (aprIt.set_new_lzx(level, z, x) < UINT64_MAX) {

                     key = encode_xzl(x,z,level,1);
                    level_zx_index_start.emplace_back(std::make_tuple<std::size_t,std::size_t>((std::size_t)key,
                                                                                               (std::size_t)aprIt.particles_zx_end(level,z,x))); //This stores the begining and end global index for each level_xz_row
                } else {
                     key = encode_xzl(x,z,level,0);
                    level_zx_index_start.emplace_back(std::make_tuple<std::size_t,std::size_t>((std::size_t)key,(std::size_t) pcounter)); //This stores the begining and end global index for each level_
                }


                for (aprIt.set_new_lzx(level, z, x);
                     aprIt.global_index() < aprIt.particles_zx_end(level, z,
                                                                   x); aprIt.set_iterator_to_particle_next_particle()) {
                    y_explicit.emplace_back(aprIt.y());
                    particle_values.emplace_back(apr.particles_intensities[aprIt]);
                    pcounter++;

                }
            }

        }
    }

    timer.stop_timer();



    ////////////////////
    ///
    /// Example of doing our level,z,x access using the GPU data structure
    ///
    /////////////////////
    timer.start_timer("transfer structures to GPU");


    uint64_t total_number_rows = level_zx_index_start.size();

    thrust::host_vector<thrust::tuple<std::size_t,std::size_t> > h_level_zx_index_start(level_zx_index_start.size());
    thrust::transform(level_zx_index_start.begin(), level_zx_index_start.end(),
                      h_level_zx_index_start.begin(),
                      [] ( const auto& _el ){
                          return thrust::make_tuple(std::get<0>(_el), std::get<1>(_el));
                      } );

    thrust::device_vector<thrust::tuple<std::size_t,std::size_t> > d_level_zx_index_start = h_level_zx_index_start;


    thrust::device_vector<std::uint16_t> d_y_explicit(y_explicit.begin(), y_explicit.end()); //y-coordinates
    thrust::device_vector<std::uint16_t> d_particle_values(particle_values.begin(), particle_values.end()); //particle values


    thrust::device_vector<std::size_t> d_level_offset(level_offset.begin(),level_offset.end()); //cumsum of number of rows in lower levels

    /*
     * Dynamic load balancing of the APR data-structure variables
     *
     */

    std::size_t max_number_chunks = 8191;
    thrust::device_vector<std::size_t> d_ind_end(max_number_chunks,0);
    std::size_t*   chunk_index_end  =  thrust::raw_pointer_cast(d_ind_end.data());

    const thrust::tuple<std::size_t,std::size_t>* row_info =  thrust::raw_pointer_cast(d_level_zx_index_start.data());
    const std::uint16_t*             particle_y   =  thrust::raw_pointer_cast(d_y_explicit.data());
    const std::uint16_t*             pdata  =  thrust::raw_pointer_cast(d_particle_values.data());
    const std::size_t*             offsets= thrust::raw_pointer_cast(d_level_offset.data());


    ExtraParticleData<uint16_t> iteration_check_particles(apr);
    iteration_check_particles.init_gpu(apr.total_number_particles());


    timer.stop_timer();

    /*
     * Dynamic load balancing of the APR data-structure variables
     *
     */

    timer.start_timer("load balancing");

    std::cout << "Total number of rows: " << total_number_rows << std::endl;

    std::size_t total_number_particles = apr.total_number_particles();

    //Figuring out how many particles per chunk are required
    std::size_t max_particles_per_row = apr.orginal_dimensions(0); //maximum number of particles in a row
    std::size_t parts_per_chunk = std::max((std::size_t)(max_particles_per_row+1),(std::size_t) floor(total_number_particles/max_number_chunks)); // to gurantee every chunk stradles across more then one row, the minimum particle chunk needs ot be larger then the largest possible number of particles in a row

    std::size_t actual_number_chunks = total_number_particles/parts_per_chunk + 1; // actual number of chunks realized based on the constraints on the total number of particles and maximum row

    dim3 threads(32);
    dim3 blocks((total_number_rows + threads.x - 1)/threads.x);

    std::cout << "Particles per chunk: " << parts_per_chunk << " Total number of chunks: " << actual_number_chunks << std::endl;

    load_balance_xzl<<<blocks,threads>>>(row_info,chunk_index_end,actual_number_chunks,parts_per_chunk,total_number_rows);
    cudaDeviceSynchronize();

    timer.stop_timer();


    /*
     *  Now launch the kernels across all the chunks determiend by the load balancing
     *
     */


    int number_reps = 40;


    timer.start_timer("iterate over all particles");

    dim3 threads_dyn(32);
    dim3 blocks_dyn((actual_number_chunks + threads_dyn.x - 1)/threads_dyn.x);

    for (int rep = 0; rep < number_reps; ++rep) {

        test_dynamic_balance << < blocks_dyn, threads_dyn >> >
                                              (row_info, chunk_index_end, actual_number_chunks, particle_y, iteration_check_particles.gpu_pointer);
        cudaDeviceSynchronize();
    }


    timer.stop_timer();

    float gpu_iterate_time = timer.timings.back();




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



    timer.start_timer("summing the sptial informatino for each partilce on the GPU");
    for (int rep = 0; rep < number_reps; ++rep) {

        test_dynamic_balance_XZYL << < blocks_dyn, threads_dyn >> >
                                                   (row_info, chunk_index_end, actual_number_chunks, particle_y, spatial_info_test.gpu_pointer);

        cudaDeviceSynchronize();
    }

    timer.stop_timer();

    float gpu_iterate_time_si = timer.timings.back();

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

            test_cpu[aprIt] += 1;

        }
    }

    timer.stop_timer();

    float cpu_iterate_time = timer.timings.back();



    timer.start_timer("Performance comparison on CPU access sum"); //not working
    for (int rep = 0; rep < number_reps; ++rep) {

#pragma omp parallel for schedule(static) private(particle_number) firstprivate(aprIt)
        for (uint64_t particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
            //This step is required for all loops to set the iterator by the particle number
            aprIt.set_iterator_to_particle_by_number(particle_number);

            test_cpu[aprIt] = aprIt.x() + aprIt.y() + aprIt.z() + aprIt.level();

        }
    }

    timer.stop_timer();

    float cpu_iterate_time_si = timer.timings.back();

    std::cout << "SPEEDUP GPU vs. CPU iterate= " << cpu_iterate_time/gpu_iterate_time << std::endl;
    std::cout << "SPEEDUP GPU vs. CPU iterate (Spatial Info)= " << cpu_iterate_time_si/gpu_iterate_time_si << std::endl;

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
        if(iteration_check_particles[aprIt]==number_reps){
            c_pass++;
        } else {
            c_fail++;
            success = false;
            //std::cout << test_access_data[particle_number] << " Level: " < aprIt.level() << std::endl;
        }
    }

    if(success){
        std::cout << "Iteration Check, PASS" << std::endl;
    } else {
        std::cout << "Iteration Check, FAIL Total: " << c_fail << " Pass Total:  " << c_pass << std::endl;
    }


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
        if(spatial_info_test[aprIt]==(aprIt.x() + aprIt.y() + aprIt.z() + aprIt.level())){
            c_pass++;
        } else {
            c_fail++;
            success = false;
            //std::cout << test_access_data[particle_number] << " Level: " < aprIt.level() << std::endl;
        }
    }

    if(success){
        std::cout << "Spatial information Check, PASS" << std::endl;
    } else {
        std::cout << "Spatial information Check, FAIL Total: " << c_fail << " Pass Total:  " << c_pass << std::endl;
    }

}

__global__ void test_dynamic_balance(const thrust::tuple<std::size_t,std::size_t>* row_info,std::size_t*  _chunk_index_end,
                                     std::size_t total_number_chunks,const std::uint16_t* particle_y,std::uint16_t* particle_data_output){

    int chunk_index = blockDim.x * blockIdx.x + threadIdx.x; // the input to each kernel is its chunk index for which it should iterate over

    if(chunk_index >= total_number_chunks){
        return; //out of bounds
    }

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

            //loop over the particles in the row
            for (std::size_t particle_global_index = particle_global_index_begin; particle_global_index < particle_global_index_end; ++particle_global_index) {

                particle_data_output[particle_global_index]+=1;
            }
        }
    }


}

__global__ void test_dynamic_balance_XZYL(const thrust::tuple<std::size_t,std::size_t>* row_info,std::size_t*  _chunk_index_end,
                                     std::size_t total_number_chunks,const std::uint16_t* particle_y,std::uint16_t* particle_data_output){

    int chunk_index = blockDim.x * blockIdx.x + threadIdx.x; // the input to each kernel is its chunk index for which it should iterate over

    if(chunk_index >= total_number_chunks){
        return; //out of bounds
    }

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
                particle_data_output[particle_global_index]=current_y+x+z+level;
            }

        }

    }


}


__global__ void load_balance_xzl(const thrust::tuple<std::size_t,std::size_t>* row_info,std::size_t*  _chunk_index_end,
                                 std::size_t total_number_chunks,std::float_t parts_per_block,std::size_t total_number_rows){

    int row_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(row_index>=total_number_rows){
        return;
    }

//    std::size_t key= thrust::get<0>(row_info[row_index]);
//
//    if(!key&1){
//        return; //empty row
//    }

    std::size_t index_end = thrust::get<1>(row_info[row_index]);
    std::size_t index_begin;

    if(row_index > 0){
        index_begin = thrust::get<1>(row_info[row_index-1]);
    } else {
        index_begin =0;
    }

    std::size_t chunk_start = floor(index_begin/parts_per_block);
    std::size_t chunk_end =  floor(index_end/parts_per_block);

    if(chunk_start!=chunk_end){
        _chunk_index_end[chunk_end]=row_index;
    }

    if(row_index == (total_number_rows-1)){
        _chunk_index_end[total_number_chunks-1]=total_number_rows-1;
    }



}



void create_test_particles_surya(APR<uint16_t>& apr,APRIterator<uint16_t>& apr_iterator,ExtraParticleData<float> &test_particles,ExtraParticleData<uint16_t>& particles,std::vector<float>& stencil, const int stencil_size, const int stencil_half){

    for (uint64_t level_local = apr_iterator.level_max(); level_local >= apr_iterator.level_min(); --level_local) {


        MeshData<float> by_level_recon;
        by_level_recon.init(apr_iterator.spatial_index_y_max(level_local),apr_iterator.spatial_index_x_max(level_local),apr_iterator.spatial_index_z_max(level_local),0);

        uint64_t level = level_local;

        const int step_size = 1;

        uint64_t particle_number;

        for (particle_number = apr_iterator.particles_level_begin(level);
             particle_number < apr_iterator.particles_level_end(level); ++particle_number) {
            //
            //  Parallel loop over level
            //
            apr_iterator.set_iterator_to_particle_by_number(particle_number);

            int dim1 = apr_iterator.y() ;
            int dim2 = apr_iterator.x() ;
            int dim3 = apr_iterator.z() ;

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


        int x = 0;
        int z = 0;


        for (z = 0; z < (apr.spatial_index_z_max(level)); ++z) {
            //lastly loop over particle locations and compute filter.
            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.particles_zx_end(level, z,
                                                                                 x); apr_iterator.set_iterator_to_particle_next_particle()) {
                    double neigh_sum = 0;
                    float counter = 0;

                    const int k = apr_iterator.y(); // offset to allow for boundary padding
                    const int i = x;

                    //test_particles[apr_iterator]=0;

                    for (int l = -stencil_half; l < stencil_half+1; ++l) {
                        for (int q = -stencil_half; q < stencil_half+1; ++q) {
                            for (int w = -stencil_half; w < stencil_half+1; ++w) {

                                if((k+w)>=0 & (k+w) < (apr.spatial_index_y_max(level))){
                                    if((i+q)>=0 & (i+q) < (apr.spatial_index_x_max(level))){
                                        if((z+l)>=0 & (z+l) < (apr.spatial_index_z_max(level))){
                                            neigh_sum += stencil[counter] * by_level_recon.at(k + w, i + q, z+l);
                                            //neigh_sum += by_level_recon.at(k + w, i + q, z+l);
                                            //if(l==1) {
                                            //  test_particles[apr_iterator] = by_level_recon.at(k, i , z+l);
                                            //}
                                        }
                                    }
                                }
                                counter++;
                            }
                        }
                    }

                    test_particles[apr_iterator] = std::round(neigh_sum/(pow(stencil_size,3)*1.0));
                    test_particles[apr_iterator] = 1;

                }
            }
        }

        //std::string image_file_name = apr.parameters.input_dir + std::to_string(level_local) + "_by_level.tif";
       // TiffUtils::saveMeshAsTiff(image_file_name, by_level_recon);
    }

}



