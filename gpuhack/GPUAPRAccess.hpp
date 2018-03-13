//
// Created by cheesema on 13.03.18.
//

#ifndef LIBAPR_GPUAPRACCESS_HPP
#define LIBAPR_GPUAPRACCESS_HPP

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/ExtraParticleData.hpp"
#include "misc/APRTimer.hpp"

#include "thrust/device_vector.h"
#include "thrust/tuple.h"
#include "thrust/copy.h"

#define KEY_EMPTY_MASK ((((uint64_t)1) << 1) - 1) << 0 //first bit stores if the row is empty or not can be used to avoid computations and accessed using &key
#define KEY_EMPTY_SHIFT 0

#define KEY_X_MASK ((((uint64_t)1) << 16) - 1) << 1
#define KEY_X_SHIFT 1

#define KEY_Z_MASK ((((uint64_t)1) << 16) - 1) << 17
#define KEY_Z_SHIFT 17

#define KEY_LEVEL_MASK ((((uint64_t)1) << 8) - 1) << 33
#define KEY_LEVEL_SHIFT 33

struct GPUAccessPtrs{
    const thrust::tuple<std::size_t,std::size_t>* row_info;
    std::size_t*  _chunk_index_end;
    std::size_t total_number_chunks;
    const std::uint16_t* y_part_coord;
};

class GPUAPRAccess {

public:

    std::size_t maximum_number_chunks;

    template<typename T>
    GPUAPRAccess(APR<T>& apr){
        initialize_gpu_access(apr);
    }
    template<typename T>
    void initialize_gpu_access(APR<T>& apr){

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

        APRIterator<T> aprIt(apr);


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

                        key = GPUAPRAccess::encode_xzl(x,z,level,1);
                        level_zx_index_start.emplace_back(std::make_tuple<std::size_t,std::size_t>((std::size_t)key,
                                                                                                   (std::size_t)aprIt.particles_zx_end(level,z,x))); //This stores the begining and end global index for each level_xz_row
                    } else {
                        key = GPUAPRAccess::encode_xzl(x,z,level,0);
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


        std::size_t max_number_chunks = 8191;
        thrust::device_vector<std::size_t> d_ind_end(max_number_chunks,0);
        std::size_t*   chunk_index_end  =  thrust::raw_pointer_cast(d_ind_end.data());

        const thrust::tuple<std::size_t,std::size_t>* row_info =  thrust::raw_pointer_cast(d_level_zx_index_start.data());
        const std::uint16_t*             particle_y   =  thrust::raw_pointer_cast(d_y_explicit.data());
        const std::uint16_t*             pdata  =  thrust::raw_pointer_cast(d_particle_values.data());
        const std::size_t*             offsets= thrust::raw_pointer_cast(d_level_offset.data());

    }


    static uint64_t encode_xzl(uint16_t x, uint16_t z, uint8_t level, bool nonzero) {

        uint64_t raw_key = 0;

        raw_key |= ((uint64_t) x << KEY_X_SHIFT);
        raw_key |= ((uint64_t) z << KEY_Z_SHIFT);
        raw_key |= ((uint64_t) level << KEY_LEVEL_SHIFT);

        if (nonzero) {
            raw_key |= (1 << KEY_EMPTY_SHIFT);
        } else {
            raw_key |= (0 << KEY_EMPTY_SHIFT);
        }


        uint64_t output_x = (raw_key & KEY_X_MASK) >> KEY_X_SHIFT;
        uint64_t output_z = (raw_key & KEY_Z_MASK) >> KEY_Z_SHIFT;
        uint64_t output_level = (raw_key & KEY_LEVEL_MASK) >> KEY_LEVEL_SHIFT;
        uint64_t output_nz = (raw_key & KEY_EMPTY_MASK) >> KEY_EMPTY_SHIFT;

        uint64_t short_nz = raw_key & 1;

        return raw_key;

    }

    static bool decode_xzl(std::uint64_t raw_key, uint16_t &output_x, uint16_t &output_z, uint8_t &output_level) {

        output_x = (raw_key & KEY_X_MASK) >> KEY_X_SHIFT;
        output_z = (raw_key & KEY_Z_MASK) >> KEY_Z_SHIFT;
        output_level = (raw_key & KEY_LEVEL_MASK) >> KEY_LEVEL_SHIFT;

        return raw_key & 1;

    }

    __global__ void load_balance_xzl(const thrust::tuple<std::size_t,std::size_t>* row_info,std::size_t*  _chunk_index_end,
                                     std::size_t total_number_chunks,std::float_t parts_per_block,std::size_t total_number_rows){

        int row_index = blockDim.x * blockIdx.x + threadIdx.x;

        if(row_index>=total_number_rows){
            return;
        }

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

};

#endif //LIBAPR_GPUAPRACCESS_HPP
