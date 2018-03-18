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
struct GPUAccessPtrs{
    const thrust::tuple<std::size_t,std::size_t>* row_info;
    std::size_t*  _chunk_index_end;
    std::size_t total_number_chunks;
    const std::uint16_t* y_part_coord;
    std::size_t* level_offsets;
    std::uint16_t* level_x_num;
    std::uint16_t* level_z_num;
    std::uint16_t* level_y_num;
    std::size_t* row_global_index;
};

class GPUAPRAccess {

public:

    const thrust::tuple<std::size_t,std::size_t>* row_info;
    std::size_t*  _chunk_index_end;
    std::size_t total_number_chunks;
    const std::uint16_t* y_part_coord;
    std::size_t* level_offsets;


    GPUAccessPtrs gpu_access;
    GPUAccessPtrs* gpu_access_ptr;

    //device access data
    thrust::device_vector<thrust::tuple<std::size_t,std::size_t> > d_level_zx_index_start;
    thrust::device_vector<std::uint16_t> d_y_part_coord; //y-coordinates

    thrust::device_vector<std::size_t> d_level_offset; //cumsum of number of rows in lower levels
    thrust::device_vector<std::size_t> d_chunk_index_end;

    thrust::device_vector<std::size_t> d_row_global_index;


    thrust::device_vector<std::uint16_t> d_x_num_level;
    thrust::device_vector<std::uint16_t> d_y_num_level;
    thrust::device_vector<std::uint16_t> d_z_num_level;

    std::vector<std::size_t> h_level_offset;

    std::size_t max_number_chunks = 8191;
    std::size_t actual_number_chunks;

    template<typename T>
    GPUAPRAccess(APRIterator<T>& aprIt){
        initialize_gpu_access_alternate(aprIt);
    }

    template<typename T>
    GPUAPRAccess(APR<T>& apr,uint64_t max_number_chunks = 8191):max_number_chunks(max_number_chunks){
        initialize_gpu_access(apr);
    }

    template<typename T>
    void initialize_gpu_access(APR<T>& apr){

        ///////////////////////////
        ///
        /// Sparse Data for GPU
        ///
        ///////////////////////////

        APRIterator<T> aprIt(apr);

        std::vector<std::tuple<std::size_t,std::size_t>> level_zx_index_start;//size = number of rows on all levels
        std::vector<std::uint16_t> y_explicit;
        y_explicit.reserve(aprIt.total_number_particles());//size = number of particles
        std::vector<std::uint16_t> particle_values;
        particle_values.reserve(aprIt.total_number_particles());//size = number of particles
        h_level_offset.resize(aprIt.level_max()+1,0);//size = number of levels

        std::size_t x = 0;
        std::size_t z = 0;

        std::size_t zx_counter = 0;
        std::size_t pcounter = 0;



        uint64_t bundle_xzl=0;

        APRTimer timer;
        timer.verbose_flag = true;

        timer.start_timer("initialize structure");

        for (int level = aprIt.level_min(); level <= aprIt.level_max(); ++level) {
            h_level_offset[level] = zx_counter;

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

        //copy to device
        d_level_zx_index_start = h_level_zx_index_start;
        d_y_part_coord.resize(apr.total_number_particles());
        thrust::copy(y_explicit.begin(), y_explicit.end(),d_y_part_coord.data()); //y-coordinates
        d_level_offset.resize(aprIt.level_max()+1);
        thrust::copy(h_level_offset.begin(),h_level_offset.end(),d_level_offset.data()); //cumsum of number of rows in lower levels

        d_chunk_index_end.resize(max_number_chunks);

        std::size_t*   chunk_index_end  =  thrust::raw_pointer_cast(d_chunk_index_end.data());

        const thrust::tuple<std::size_t,std::size_t>* row_info =  thrust::raw_pointer_cast(d_level_zx_index_start.data());
        const std::uint16_t*             particle_y   =  thrust::raw_pointer_cast(d_y_part_coord.data());
        const std::size_t*             offsets= thrust::raw_pointer_cast(h_level_offset.data());

        timer.start_timer("load balancing");

        std::cout << "Total number of rows: " << total_number_rows << std::endl;

        std::size_t total_number_particles = apr.total_number_particles();

        //Figuring out how many particles per chunk are required
        std::size_t max_particles_per_row = apr.orginal_dimensions(0); //maximum number of particles in a row
        std::size_t parts_per_chunk = std::max((std::size_t)(max_particles_per_row+1),(std::size_t) floor(total_number_particles/max_number_chunks)); // to gurantee every chunk stradles across more then one row, the minimum particle chunk needs ot be larger then the largest possible number of particles in a row

        actual_number_chunks = total_number_particles/parts_per_chunk + 1; // actual number of chunks realized based on the constraints on the total number of particles and maximum row

        dim3 threads(32);
        dim3 blocks((total_number_rows + threads.x - 1)/threads.x);

        std::cout << "Particles per chunk: " << parts_per_chunk << " Total number of chunks: " << actual_number_chunks << std::endl;

        load_balance_xzl<<<blocks,threads>>>(row_info,chunk_index_end,actual_number_chunks,parts_per_chunk,total_number_rows);
        cudaDeviceSynchronize();

        timer.stop_timer();


        std::vector<uint16_t> x_num_level;
        std::vector<uint16_t> z_num_level;
        std::vector<uint16_t> y_num_level;
        x_num_level.resize(apr.level_max()+1);
        z_num_level.resize(apr.level_max()+1);
        y_num_level.resize(apr.level_max()+1);

        d_x_num_level.resize(apr.level_max()+1);
        d_z_num_level.resize(apr.level_max()+1);
        d_y_num_level.resize(apr.level_max()+1);

        for (int i = apr.level_min(); i <= apr.level_max(); ++i) {
            x_num_level[i] = apr.spatial_index_x_max(i);
            y_num_level[i] = apr.spatial_index_y_max(i);
            z_num_level[i] = apr.spatial_index_z_max(i);
        }

        thrust::copy(x_num_level.begin(),x_num_level.end(),d_x_num_level.data());
        thrust::copy(y_num_level.begin(),y_num_level.end(),d_y_num_level.data());
        thrust::copy(z_num_level.begin(),z_num_level.end(),d_z_num_level.data());

        gpu_access.level_x_num = thrust::raw_pointer_cast(d_x_num_level.data());
        gpu_access.level_y_num = thrust::raw_pointer_cast(d_y_num_level.data());
        gpu_access.level_z_num = thrust::raw_pointer_cast(d_z_num_level.data());


        //set up gpu pointers
        row_info =  thrust::raw_pointer_cast(d_level_zx_index_start.data());
        _chunk_index_end = thrust::raw_pointer_cast(d_chunk_index_end.data());
        total_number_chunks = actual_number_chunks;
        y_part_coord = thrust::raw_pointer_cast(d_y_part_coord.data());
        level_offsets = thrust::raw_pointer_cast(d_level_offset.data());

        //set up gpu pointers
        gpu_access.row_info =  thrust::raw_pointer_cast(d_level_zx_index_start.data());
        gpu_access._chunk_index_end = thrust::raw_pointer_cast(d_chunk_index_end.data());
        gpu_access.total_number_chunks = actual_number_chunks;
        gpu_access.y_part_coord = thrust::raw_pointer_cast(d_y_part_coord.data());
        gpu_access.level_offsets = thrust::raw_pointer_cast(d_level_offset.data());

        //transfer data across
        cudaMalloc((void**)&gpu_access_ptr, sizeof(GPUAccessPtrs));
        cudaMemcpy(gpu_access_ptr, &gpu_access, sizeof(GPUAccessPtrs), cudaMemcpyHostToDevice);

    }


    template<typename T>
    void initialize_gpu_access_alternate(APRIterator<T>& aprIt){

        ///////////////////////////
        ///
        /// Sparse Data for GPU
        ///
        ///////////////////////////

        std::vector<std::size_t> row_global_index;//size = number of rows on all levels
        std::vector<std::uint16_t> y_explicit;
        y_explicit.reserve(aprIt.total_number_particles());//size = number of particles
        //row_global_index.reserve(aprIt.total_number_particles());

        h_level_offset.resize(aprIt.level_max()+1,0);//size = number of levels

        std::size_t x = 0;
        std::size_t z = 0;

        std::size_t zx_counter = 0;
        std::size_t pcounter = 0;



        uint64_t bundle_xzl=0;

        APRTimer timer;
        timer.verbose_flag = true;

        timer.start_timer("initialize structure");

        for (int level = aprIt.level_min(); level <= aprIt.level_max(); ++level) {
            h_level_offset[level] = zx_counter;

            for (z = 0; z < aprIt.spatial_index_z_max(level); ++z) {
                for (x = 0; x < aprIt.spatial_index_x_max(level); ++x) {

                    zx_counter++;
                    uint64_t key;
                    if (aprIt.set_new_lzx(level, z, x) < UINT64_MAX) {

                        row_global_index.push_back(aprIt.particles_zx_end(level,z,x)); //This stores the begining and end global index for each level_xz_row
                    } else {
                        row_global_index.push_back(pcounter);
                    }


                    for (aprIt.set_new_lzx(level, z, x);
                         aprIt.global_index() < aprIt.particles_zx_end(level, z,
                                                                       x); aprIt.set_iterator_to_particle_next_particle()) {
                        y_explicit.emplace_back(aprIt.y());
                        pcounter++;

                    }
                }

            }
        }

        timer.stop_timer();


        //copy to device
        timer.start_timer("transfer access data to GPU");

        d_y_part_coord.resize(aprIt.total_number_particles());
        thrust::copy(y_explicit.begin(), y_explicit.end(),d_y_part_coord.data()); //y-coordinates
        d_level_offset.resize(aprIt.level_max()+1);
        thrust::copy(h_level_offset.begin(),h_level_offset.end(),d_level_offset.data()); //cumsum of number of rows in lower levels

        d_row_global_index.resize(row_global_index.size());
        thrust::copy(row_global_index.begin(),row_global_index.end(),d_row_global_index.data());

        std::vector<uint16_t> x_num_level;
        std::vector<uint16_t> z_num_level;
        std::vector<uint16_t> y_num_level;
        x_num_level.resize(aprIt.level_max()+1);
        z_num_level.resize(aprIt.level_max()+1);
        y_num_level.resize(aprIt.level_max()+1);

        d_x_num_level.resize(aprIt.level_max()+1);
        d_z_num_level.resize(aprIt.level_max()+1);
        d_y_num_level.resize(aprIt.level_max()+1);

        for (int i = aprIt.level_min(); i <= aprIt.level_max(); ++i) {
            x_num_level[i] = aprIt.spatial_index_x_max(i);
            y_num_level[i] = aprIt.spatial_index_y_max(i);
            z_num_level[i] = aprIt.spatial_index_z_max(i);
        }

        thrust::copy(x_num_level.begin(),x_num_level.end(),d_x_num_level.data());
        thrust::copy(y_num_level.begin(),y_num_level.end(),d_y_num_level.data());
        thrust::copy(z_num_level.begin(),z_num_level.end(),d_z_num_level.data());


        timer.stop_timer();


        //Figuring out how many particles per chunk are required


        gpu_access.level_x_num = thrust::raw_pointer_cast(d_x_num_level.data());
        gpu_access.level_y_num = thrust::raw_pointer_cast(d_y_num_level.data());
        gpu_access.level_z_num = thrust::raw_pointer_cast(d_z_num_level.data());


        //set up gpu pointers

        gpu_access.row_global_index = thrust::raw_pointer_cast(d_row_global_index.data());
        gpu_access.total_number_chunks = actual_number_chunks;
        gpu_access.y_part_coord = thrust::raw_pointer_cast(d_y_part_coord.data());
        gpu_access.level_offsets = thrust::raw_pointer_cast(d_level_offset.data());

        //transfer data across
        cudaMalloc((void**)&gpu_access_ptr, sizeof(GPUAccessPtrs));
        cudaMemcpy(gpu_access_ptr, &gpu_access, sizeof(GPUAccessPtrs), cudaMemcpyHostToDevice);

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

        return raw_key;

    }

    static bool decode_xzl(std::uint64_t raw_key, uint16_t &output_x, uint16_t &output_z, uint8_t &output_level) {

        output_x = (raw_key & KEY_X_MASK) >> KEY_X_SHIFT;
        output_z = (raw_key & KEY_Z_MASK) >> KEY_Z_SHIFT;
        output_level = (raw_key & KEY_LEVEL_MASK) >> KEY_LEVEL_SHIFT;

        return raw_key & 1;

    }


};

#endif //LIBAPR_GPUAPRACCESS_HPP
