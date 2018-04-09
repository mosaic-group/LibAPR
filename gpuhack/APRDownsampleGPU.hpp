//
// Created by cheesema on 05.04.18.
//

#ifndef LIBAPR_GPU_DOWNSAMPLE_HPP
#define LIBAPR_GPU_DOWNSAMPLE_HPP

#endif //LIBAPR_GPU_DOWNSAMPLE_HPP


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


    __shared__ float parent_cache[8][16];


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

