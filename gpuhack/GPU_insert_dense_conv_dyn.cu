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


void create_test_particles_surya(APR<uint16_t>& apr,APRIterator<uint16_t>& apr_iterator,ExtraParticleData<float> &test_particles,ExtraParticleData<uint16_t>& particles,std::vector<float>& stencil, const int stencil_size, const int stencil_half);


__global__ void insert(
        std::size_t _level,
        std::size_t _z_index,
        const thrust::tuple<std::size_t,std::size_t>* _line_offsets,
        const std::uint16_t*           _y_ex,
        const std::uint16_t*           _pdata,
        const std::size_t*             _offsets,
        std::size_t                    _max_y,
        std::size_t                    _max_x,
        std::size_t                    _nparticles,
        std::uint16_t*                 _temp_vec,
        std::size_t                    _stencil_size,
        std::size_t                    _stencil_half
){

    unsigned int x_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(x_index >= _max_x){
        return; // out of bounds
    }

    auto level_zx_offset = _offsets[_level] + _max_x * _z_index + x_index;
    auto row_start = _line_offsets[level_zx_offset];


    auto particle_index_begin = thrust::get<0>(row_start);
    auto particle_index_end   = thrust::get<1>(row_start);

    //_z_index = 0;
    auto t_index = x_index*_max_y + ((_z_index % _stencil_size)*_max_y*_max_x) ;

//    for(auto y = 0;y < _max_y;y++){
//        _temp_vec[t_index+y]  = 0;
//    }

    if((particle_index_end-particle_index_begin) == 0)
        return;


    for (std::size_t global_index = particle_index_begin;
         global_index < particle_index_end; ++global_index) {

        uint16_t current_particle_value = _pdata[global_index];
        auto y = _y_ex[global_index];
        _temp_vec[t_index+y] = current_particle_value;

    }

}

__global__ void insert_dynamic(
        std::size_t _level,
        std::size_t _z_index,
        const thrust::tuple<std::size_t,std::size_t>* _line_offsets,
        const std::uint16_t*           _y_ex,
        const std::uint16_t*           _pdata,
        const std::size_t*             _offsets,
        std::size_t                    _max_y,
        std::size_t                    _max_x,
        std::size_t                    _nparticles,
        std::uint16_t*                 _temp_vec,
        std::size_t                    _stencil_size,
        std::size_t                    _stencil_half,std::uint16_t*  _xend,std::size_t*  _ind_end,std::size_t num_blocks
){


    int load_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(load_index>=num_blocks){
        return;
    }
    //_z_index = 0;



    std::size_t x_begin;
    std::size_t x_end = _xend[load_index];

    auto level_zx_offset = _offsets[_level] + _max_x * _z_index;

    std::size_t parts_begin;
    std::size_t parts_end;

    if(load_index==0){
        x_begin = 0;
        parts_begin = thrust::get<0>(_line_offsets[level_zx_offset]);

    } else {
        x_begin = _xend[load_index-1];
        parts_begin = _ind_end[load_index-1];
    }

    //

    for (auto x = x_begin; x <= x_end; ++x) {


        auto t_index = x*_max_y + ((_z_index % _stencil_size)*_max_y*_max_x) ;

        if(x!=x_begin){
            parts_begin  = thrust::get<0>(_line_offsets[level_zx_offset+x]);
        }

        if(x!=x_end){
            parts_end  = thrust::get<1>(_line_offsets[level_zx_offset+x]);
        } else {
            parts_end  = _ind_end[load_index];
        }


        for (std::size_t p = parts_begin; p < parts_end ; ++p) {
            //uint16_t current_particle_value = ;
            auto y = _y_ex[p];
            _temp_vec[t_index+y] = _pdata[p];
        }

    }


}


__global__ void load_balance(std::size_t _level,std::size_t _z_index,const thrust::tuple<std::size_t,std::size_t>* _line_offsets,std::uint16_t*  _xend,std::size_t*  _ind_end,const std::size_t* _offsets,
                             std::size_t   _max_x,std::size_t num_blocks,std::float_t parts_per_block,std::size_t parts_begin){

    int x_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(x_index >= _max_x){
        return; // out of bounds
    }

    //printf("Hello from dim: %d block: %d, thread: %d  x index: %d z: %d \n",blockDim.x, blockIdx.x, threadIdx.x,x_index,(int) _z_index);

    auto level_zx_offset = _offsets[_level] + _max_x * _z_index + x_index;

    std::size_t parts_end = thrust::get<1>(_line_offsets[level_zx_offset]);

    std::size_t index_begin =  floor((thrust::get<0>(_line_offsets[level_zx_offset])-parts_begin)/parts_per_block);

    std::size_t index_end;

    if(parts_end==parts_begin){
        index_end=0;
    } else {
        index_end = floor((parts_end-parts_begin)/parts_per_block);
    }

    //need to add the loop
    if(index_begin!=index_end){

        if(index_begin==0){
            _ind_end[0] = parts_begin;
            _xend[0] = 0;
        }

        for (int i = (index_begin+1); i <= index_end; ++i) {
            _xend[i]=x_index;
            _ind_end[i]= floor(i*parts_per_block) + parts_begin;

        }
    }



    if(x_index==(_max_x-1)){
        _ind_end[num_blocks-1] = parts_end;
        _xend[num_blocks-1] = (_max_x-1);

    }



}


__global__ void test_iterate(std::size_t _level,std::size_t _z_index,const thrust::tuple<std::size_t,std::size_t>* _line_offsets,std::uint16_t*  _xend,std::size_t*  _ind_end,const std::size_t* _offsets,
        std::size_t   _max_x,std::uint16_t*  _pdata,std::size_t num_blocks){

    int load_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(load_index>=num_blocks){
        return;
    }

    std::size_t x_begin;
    std::size_t x_end = _xend[load_index];

    auto level_zx_offset = _offsets[_level] + _max_x * _z_index;

    std::size_t parts_begin;
    std::size_t parts_end;

    if(load_index==0){
        x_begin = 0;
        parts_begin = thrust::get<0>(_line_offsets[level_zx_offset]);

    } else {
        x_begin = _xend[load_index-1];
        parts_begin = _ind_end[load_index-1];
    }

    //

    for (auto x = x_begin; x <= x_end; ++x) {


        if(x!=x_begin){
            parts_begin  = thrust::get<0>(_line_offsets[level_zx_offset+x]);
        }

        if(x!=x_end){
            parts_end  = thrust::get<1>(_line_offsets[level_zx_offset+x]);
        } else {
            parts_end  = _ind_end[load_index];
        }


         for (std::size_t p = parts_begin; p < parts_end ; ++p) {
           _pdata[p]+=1;
         }

    }

}


__global__ void push_back_dynamic(
        std::size_t _level,
        std::size_t _z_index,
        const thrust::tuple<std::size_t,std::size_t>* _line_offsets,
        const std::uint16_t*           _y_ex,
        const std::uint16_t*           _temp_vec,
        const std::size_t*             _offsets,
        std::size_t                    _max_y,
        std::size_t                    _max_x,
        std::size_t 		   _max_z,
        std::size_t                    _nparticles,
        std::uint16_t*                 _pdata,
        std::size_t                    _stencil_size,
        std::size_t                    _stencil_half,
        const std::float_t*           _stencil,std::uint16_t*  _xend,std::size_t*  _ind_end,std::size_t num_blocks
){


    int load_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(load_index>=num_blocks){
        return;
    }
    //_z_index = 0;



    std::size_t x_begin;
    std::size_t x_end = _xend[load_index];

    auto level_zx_offset = _offsets[_level] + _max_x * _z_index;

    std::size_t parts_begin;
    std::size_t parts_end;

    if(load_index==0){
        x_begin = 0;
        parts_begin = thrust::get<0>(_line_offsets[level_zx_offset]);

    } else {
        x_begin = _xend[load_index-1];
        parts_begin = _ind_end[load_index-1];
    }

    //
    std::size_t temp_index;

    for (auto x = x_begin; x <= x_end; ++x) {


        auto t_index = x*_max_y + ((_z_index % _stencil_size)*_max_y*_max_x) ;

        if(x!=x_begin){
            parts_begin  = thrust::get<0>(_line_offsets[level_zx_offset+x]);
        }

        if(x!=x_end){
            parts_end  = thrust::get<1>(_line_offsets[level_zx_offset+x]);
        } else {
            parts_end  = _ind_end[load_index];
        }


        for (std::size_t global_index = parts_begin; global_index < parts_end ; ++global_index) {

            int counter = 0;
            double neighbour_sum = 0;
            auto y = _y_ex[global_index];

            int lower_bound = (_stencil_half);

            for(int q = -(lower_bound); q < (lower_bound+1); ++q){	 // z stencil
                for(int l = -(lower_bound); l < (lower_bound+1); ++l){   // x stencil
                    for(int w = -(lower_bound); w < (lower_bound+1); ++w){	// y stencil

                        if((x + l) >= 0 && (x + l) < _max_x){
                            if((_z_index + q) >= 0 && (_z_index + q) < _max_z){
                                if((y + w) >= 0 && (y + w) < _max_y){

                                    temp_index = (x + l)*_max_y + (((_z_index+q+ _stencil_size) % _stencil_size)*_max_y*_max_x) +y+w ;
                                    neighbour_sum += (_stencil[counter]*_temp_vec[temp_index]);
                                    //
                                    counter++;

                                }
                            }
                        }


                    }
                }
            }

            _pdata[global_index] = std::round(neighbour_sum/(pow(_stencil_size,3)*1.0));


        }

    }

}



__global__ void push_back(
        std::size_t _level,
        std::size_t _z_index,
        const thrust::tuple<std::size_t,std::size_t>* _line_offsets,
        const std::uint16_t*           _y_ex,
        const std::uint16_t*           _temp_vec,
        const std::size_t*             _offsets,
        std::size_t                    _max_y,
        std::size_t                    _max_x,
        std::size_t 		   _max_z,
        std::size_t                    _nparticles,
        std::uint16_t*                 _pdata,
        std::size_t                    _stencil_size,
        std::size_t                    _stencil_half,
        const std::float_t*           _stencil
){

    unsigned int x_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(x_index >= _max_x){
        return; // out of bounds
    }

    auto level_zx_offset = _offsets[_level] + _max_x * _z_index + x_index;
    auto row_start = _line_offsets[level_zx_offset];



    auto particle_index_begin = thrust::get<0>(row_start);
    auto particle_index_end   = thrust::get<1>(row_start);

    if((particle_index_end-particle_index_begin) == 0)
        return;

    //_z_index=0;
    auto t_index = x_index*_max_y + ((_z_index % _stencil_size)*_max_y*_max_x) ;
    auto temp_index = 0;

    // Convolution begin
    for (std::size_t global_index = particle_index_begin;
         global_index < particle_index_end; ++global_index) {

        int counter = 0;
        double neighbour_sum = 0;
        auto y = _y_ex[global_index];

        //_pdata[global_index]=1;
        _pdata[global_index]=0;


        int lower_bound = (_stencil_half);

        for(int q = -(lower_bound); q < (lower_bound+1); ++q){	 // z stencil
            for(int l = -(lower_bound); l < (lower_bound+1); ++l){   // x stencil
                for(int w = -(lower_bound); w < (lower_bound+1); ++w){	// y stencil

                    if((x_index + l) >= 0 && (x_index + l) < _max_x){
                        if((_z_index + q) >= 0 && (_z_index + q) < _max_z){
                            if((y + w) >= 0 && (y + w) < _max_y){

                                temp_index = (x_index + l)*_max_y + (((_z_index+q+ _stencil_size) % _stencil_size)*_max_y*_max_x) +y+w ;
                                neighbour_sum += (_stencil[counter]*_temp_vec[temp_index]);
                                //
                                counter++;

                            }
                        }
                    }


                }
            }
        }

        _pdata[global_index] = std::round(neighbour_sum/(pow(_stencil_size,3)*1.0));


    }
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    // Read provided APR file
    cmdLineOptions options = read_command_line_options(argc, argv);
    const int reps = 20;

    std::string fileName = options.directory + options.input;
    APR<uint16_t> apr;
    apr.read_apr(fileName);

    // Get dense representation of APR
    APRIterator<uint16_t> aprIt(apr);

    ///////////////////////////
    ///
    /// Sparse Data for GPU
    ///
    ///////////////////////////

    std::vector<std::tuple<std::size_t,std::size_t>> level_zx_index_start;//size = number of rows on all levels
    std::vector<std::uint16_t> y_explicit;y_explicit.reserve(aprIt.total_number_particles());//size = number of particles
    std::vector<std::uint16_t> particle_values;particle_values.reserve(aprIt.total_number_particles());//size = number of particles
    std::vector<std::size_t> level_offset(aprIt.level_max()+1,UINT64_MAX);//size = number of levels
    const int stencil_half = 2;
    const int stencil_size = 2*stencil_half+1;
    std::vector<std::float_t> stencil;		// the stencil on the host
    std::float_t stencil_value = 1;
    stencil.resize(pow(stencil_half*2 + 1,stencil_size),stencil_value);

    std::cout << stencil[0] << std::endl;


    std::size_t x = 0;
    std::size_t z = 0;

    std::size_t zx_counter = 0;
    std::size_t pcounter = 0;


    for (int level = aprIt.level_min(); level <= aprIt.level_max(); ++level) {
        level_offset[level] = zx_counter;

        for (z = 0; z < aprIt.spatial_index_z_max(level); ++z) {
            for (x = 0; x < aprIt.spatial_index_x_max(level); ++x) {

                zx_counter++;
                if (aprIt.set_new_lzx(level, z, x) < UINT64_MAX) {
                    level_zx_index_start.emplace_back(std::make_tuple<std::size_t,std::size_t>(aprIt.global_index(),
                                                                                               aprIt.particles_zx_end(level,z,x))); //This stores the begining and end global index for each level_xz_row
                } else {
                    level_zx_index_start.emplace_back(std::make_tuple<std::size_t,std::size_t>((std::size_t)pcounter,(std::size_t) pcounter)); //This stores the begining and end global index for each level_
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


    ////////////////////
    ///
    /// Example of doing our level,z,x access using the GPU data structure
    ///
    /////////////////////
    auto start = std::chrono::high_resolution_clock::now();


    thrust::host_vector<thrust::tuple<std::size_t,std::size_t> > h_level_zx_index_start(level_zx_index_start.size());
    thrust::transform(level_zx_index_start.begin(), level_zx_index_start.end(),
                      h_level_zx_index_start.begin(),
                      [] ( const auto& _el ){
                          return thrust::make_tuple(std::get<0>(_el), std::get<1>(_el));
                      } );

    thrust::device_vector<thrust::tuple<std::size_t,std::size_t> > d_level_zx_index_start = h_level_zx_index_start;


    thrust::device_vector<std::float_t> d_stencil(stencil.begin(), stencil.end());		// device stencil
    thrust::device_vector<std::uint16_t> d_y_explicit(y_explicit.begin(), y_explicit.end());
    thrust::device_vector<std::uint16_t> d_particle_values(particle_values.begin(), particle_values.end());
    thrust::device_vector<std::uint16_t> d_test_access_data(d_particle_values.size(),0);

    thrust::device_vector<std::size_t> d_level_offset(level_offset.begin(),level_offset.end());


    std::size_t number_blocks = 8000;

    thrust::device_vector<std::uint16_t> d_x_end(number_blocks,0);
    std::uint16_t*   _x_end  =  thrust::raw_pointer_cast(d_x_end.data());

    thrust::device_vector<std::size_t> d_ind_end(number_blocks,0);
    std::size_t*   _ind_end  =  thrust::raw_pointer_cast(d_ind_end.data());

    std::size_t max_elements = 0;

    for (int level = aprIt.level_min(); level <= aprIt.level_max(); ++level) {
        auto xtimesy = aprIt.spatial_index_y_max(level);// + (stencil_size - 1);
        xtimesy *= aprIt.spatial_index_x_max(level);// + (stencil_size - 1);
        if(max_elements < xtimesy)
            max_elements = xtimesy;
    }
    thrust::device_vector<std::uint16_t> d_temp_vec(max_elements*stencil_size,0);

    const thrust::tuple<std::size_t,std::size_t>* levels =  thrust::raw_pointer_cast(d_level_zx_index_start.data());
    const std::uint16_t*             y_ex   =  thrust::raw_pointer_cast(d_y_explicit.data());
    const std::uint16_t*             pdata  =  thrust::raw_pointer_cast(d_particle_values.data());
    const std::size_t*             offsets= thrust::raw_pointer_cast(d_level_offset.data());
    std::uint16_t*                   tvec = thrust::raw_pointer_cast(d_temp_vec.data());
    std::uint16_t*                   expected = thrust::raw_pointer_cast(d_test_access_data.data());
    const std::float_t*		     stencil_pointer =  thrust::raw_pointer_cast(d_stencil.data());		// stencil pointer

    if(cudaGetLastError()!=cudaSuccess){
        std::cerr << "memory transfers failed!\n";
    }
    auto end_gpu_tx = std::chrono::high_resolution_clock::now();

    for ( int r = 0;r<reps;++r){

        auto start_gpu_kernel = std::chrono::high_resolution_clock::now();

        for (int lvl = aprIt.level_min(); lvl <= aprIt.level_max(); ++lvl) {

            const int y_num = aprIt.spatial_index_y_max(lvl);
            const int x_num = aprIt.spatial_index_x_max(lvl);
            const int z_num = aprIt.spatial_index_z_max(lvl);

            dim3 threads(32);
            dim3 blocks((x_num + threads.x - 1) / threads.x);

//            for (int i = 0; i < stencil_half; ++i) {
//                insert<<<blocks,threads>>>(lvl,
//                        i,
//                        levels,
//                        y_ex,
//                        pdata,
//                        offsets,
//                        y_num,x_num,
//                        particle_values.size(),
//                        tvec,
//                        stencil_size, stencil_half);
//            }

            if (lvl <= aprIt.level_max()) {
                for (int z = 0; z < (z_num); ++z) {

                    if (aprIt.particles_z_begin(lvl, z) != UINT64_MAX) {

                        std::size_t num_parts = (aprIt.particles_z_end(lvl, z) - aprIt.particles_z_begin(lvl, z));
                        std::size_t num_blocks = std::min((std::size_t) number_blocks, num_parts);
                        std::float_t parts_per_block = num_parts / (num_blocks * 1.0f);
                        std::size_t parts_begin = aprIt.particles_z_begin(lvl, z);

                        unsigned int threads_d = 64;
                        unsigned int chunk = ceil((num_blocks) / (1.0f * threads_d))+1;

                        dim3 threads_load(threads_d);
                        dim3 blocks_load(chunk);

//                        std::cout << parts_per_block << std::endl;
//                        std::cout << parts_begin << std::endl;
//                        std::cout << num_parts << std::endl;


                        load_balance << < blocks, threads >> >
                                                  (lvl, z, levels, _x_end, _ind_end, offsets, x_num, num_blocks+1, parts_per_block, parts_begin);
                        cudaDeviceSynchronize();

                        //std::cout << num_blocks << " " << chunk << std::endl;

                        //test_iterate << < blocks_load, threads_load >> >
                                 //                      (lvl, z, levels, _x_end, _ind_end, offsets, x_num, expected, num_blocks+1);


                        insert_dynamic << < blocks_load, threads_load >> > (lvl,
                                z,
                                levels,
                                y_ex,
                                pdata,
                                offsets,
                                y_num, x_num,
                                particle_values.size(),
                                tvec,
                                stencil_size, stencil_half,_x_end,_ind_end,num_blocks+1);


//                        insert << < blocks, threads >> > (lvl,
//                                z ,
//                                levels,
//                                y_ex,
//                                pdata,
//                                offsets,
//                                y_num, x_num,
//                                particle_values.size(),
//                                tvec,
//                                stencil_size, stencil_half);
//
                    if (cudaGetLastError() != cudaSuccess) {
                        std::cerr << "on " << lvl << " the cuda kernel does not run!\n";
                        break;
                    }
                    cudaDeviceSynchronize();

//                    push_back << < blocks, threads >> > (lvl,
//                            z,
//                            levels,
//                            y_ex,
//                            tvec,
//                            offsets,
//                            y_num, x_num, z_num,
//                            particle_values.size(),
//                            expected,
//                            stencil_size, stencil_half, stencil_pointer);

//                        push_back_dynamic << < blocks_load, threads_load >> > (lvl,
//                                z,
//                                levels,
//                                y_ex,
//                                tvec,
//                                offsets,
//                                y_num, x_num, z_num,
//                                particle_values.size(),
//                                expected,
//                                stencil_size, stencil_half, stencil_pointer,_x_end,_ind_end,num_blocks+1);

                    cudaDeviceSynchronize();

                    }
                }
            }

        }
        cudaDeviceSynchronize();
        auto end_gpu_kernel = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> rep_diff = end_gpu_kernel - start_gpu_kernel;
        std::cout << std::setw(3) << r << " GPU:      " << rep_diff  .count() << " ms\n";

    }

    auto end_gpu_kernels = std::chrono::high_resolution_clock::now();

    std::vector<std::uint16_t> test_access_data(d_test_access_data.size(),std::numeric_limits<std::uint16_t>::max());
    thrust::copy(d_test_access_data.begin(), d_test_access_data.end(), test_access_data.begin());

    auto end_gpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double,std::milli> gpu_tx_up = end_gpu_tx - start;
    std::chrono::duration<double,std::milli> gpu_tx_down = end_gpu - end_gpu_kernels;

    std::cout << "   GPU: up   " << gpu_tx_up  .count() << " ms\n";
    std::cout << "   GPU: down " << gpu_tx_down.count() << " ms\n";

    // assert(test_access_data.back() != std::numeric_limits<std::uint16_t>::max());

    //////////////////////////
    ///
    /// Now check the data
    ///
    ////////////////////////////

    ExtraParticleData<float> utest_data(apr);
    apr.parameters.input_dir = options.directory;


    create_test_particles_surya(apr,aprIt, utest_data,apr.particles_intensities,stencil, stencil_size, stencil_half);

    bool success = true;

    uint64_t c_fail= 0;



    for (uint64_t particle_number = 0; particle_number < apr.total_number_particles(); ++particle_number) {
        //This step is required for all loops to set the iterator by the particle number
        aprIt.set_iterator_to_particle_by_number(particle_number);


        if(utest_data.data[particle_number]!=test_access_data[particle_number]){
            success = false;

                //if(aprIt.level() == 6) {
//                    std::cout << particle_number << std::endl;
//                std::cout << aprIt.x() << " " << aprIt.y() << " " << aprIt.z() << " " << aprIt.level() << " expected: "
//                          << utest_data.data[particle_number] << ", received: " << test_access_data[particle_number]
//                          << std::endl;
            //}
            //break;
            c_fail++;
        }

        // std::cout << aprIt.x()<< " "  << aprIt.y()<< " "  << aprIt.z() << " "<< aprIt.level() << " expected: " << utest_data.data[particle_number] << ", received: " << test_access_data[particle_number] << "\n";

    }


    if(success){
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL " << c_fail << std::endl;
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


