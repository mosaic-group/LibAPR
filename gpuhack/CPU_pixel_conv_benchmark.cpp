//
// Created by cheesema on 28.02.18.
//

//
// Created by cheesema on 28.02.18.
//

//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2018
///
const char* usage = R"(


)";


#include <algorithm>
#include <iostream>

#include "data_structures/APR/APR.hpp"

#include "algorithm/APRConverter.hpp"
#include "data_structures/APR/APRTree.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/APRIterator.hpp"
#include "numerics/APRTreeNumerics.hpp"
#include <numerics/APRNumerics.hpp>
#include <numerics/APRComputeHelper.hpp>

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
    int num_rep = 0;
};

cmdLineOptions read_command_line_options(int argc, char **argv);

bool command_option_exists(char **begin, char **end, const std::string &option);

char* get_command_option(char **begin, char **end, const std::string &option);

template<typename U,typename V>
float pixels_linear_neighbour_access_openmp(uint64_t y_num,uint64_t x_num,uint64_t z_num,float num_repeats,int stencil_half){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //

    MeshData<U> input_data;
    MeshData<V> output_data;
    input_data.init((int)y_num,(int)x_num,(int)z_num,23);
    output_data.init((int)y_num,(int)x_num,(int)z_num,0);

    APRTimer timer;
    timer.verbose_flag = false;
    timer.start_timer("full pixel neighbour access");

    int j = 0;
    int k = 0;
    int i = 0;

    int j_n = 0;
    int k_n = 0;
    int i_n = 0;

    //float neigh_sum = 0;
    float norm = pow(stencil_half*2+1,3);

    std::vector<float> stencil;
    stencil.resize(stencil_half*2+1,1.0f);


    for(int r = 0;r < num_repeats;r++){

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared) private(j,i,k,i_n,k_n,j_n)
#endif
        for(j = 0; j < (z_num);j++){
            for(i = 0; i < (x_num);i++){
                for(k = 0;k < (y_num);k++){
                    float neigh_sum = 0;

                    int min_j = std::max(0,j-stencil_half);
                    int min_i = std::max(0,i-stencil_half);
                    int min_k = std::max(0,k-stencil_half);

                    int max_j = std::min((int)z_num-1,j+stencil_half);
                    int max_i = std::min((int)x_num-1,i+stencil_half);
                    int max_k = std::min((int)y_num-1,k+stencil_half);

                    int counter = 0;

                    for (int j_n = min_j; j_n < max_j+1; ++j_n) {
                        for (int i_n = min_i; i_n < max_i+1; ++i_n) {
                            for (int k_n = min_k; k_n < max_k+1; ++k_n) {
                                neigh_sum += stencil[counter]*input_data.mesh[j_n*x_num*y_num + i_n*y_num + k_n];
                                counter++;
                            }
                        }
                    }

                    output_data.mesh[j*x_num*y_num + i*y_num + k] = std::roundf(neigh_sum);

                }
            }
        }

    }

    timer.stop_timer();
    float elapsed_seconds = timer.t2 - timer.t1;
    float time = elapsed_seconds/num_repeats;

    return (time);

}
template<typename U,typename V>
float pixels_linear_neighbour_access_serial(uint64_t y_num,uint64_t x_num,uint64_t z_num,float num_repeats,int stencil_half){
    //
    //  Compute two, comparitive filters for speed. Original size img, and current particle size comparison
    //

    MeshData<U> input_data;
    MeshData<V> output_data;
    input_data.init((int)y_num,(int)x_num,(int)z_num,23);
    output_data.init((int)y_num,(int)x_num,(int)z_num,0);

    APRTimer timer;
    timer.verbose_flag = false;
    timer.start_timer("full pixel neighbour access");

    int j = 0;
    int k = 0;
    int i = 0;


    //float neigh_sum = 0;
    float norm = pow(stencil_half*2+1,3);

    std::vector<float> stencil;
    stencil.resize(stencil_half*2+1,1.0f);


    for(int r = 0;r < num_repeats;r++) {

        for(j = 0; j < (z_num);j++){
            for(i = 0; i < (x_num);i++){
                for(k = 0;k < (y_num);k++){
                    float neigh_sum = 0;

                    int min_j = std::max(0,j-stencil_half);
                    int min_i = std::max(0,i-stencil_half);
                    int min_k = std::max(0,k-stencil_half);

                    int max_j = std::min((int)z_num-1,j+stencil_half);
                    int max_i = std::min((int)x_num-1,i+stencil_half);
                    int max_k = std::min((int)y_num-1,k+stencil_half);

                    int counter = 0;

                    for (int j_n = min_j; j_n < max_j+1; ++j_n) {
                        for (int i_n = min_i; i_n < max_i+1; ++i_n) {
                            for (int k_n = min_k; k_n < max_k+1; ++k_n) {
                                neigh_sum += stencil[counter]*input_data.mesh[j_n*x_num*y_num + i_n*y_num + k_n];
                                counter++;
                            }
                        }
                    }

                    output_data.mesh[j*x_num*y_num + i*y_num + k] = std::roundf(neigh_sum);

                }
            }
        }

    }


    timer.stop_timer();
    float elapsed_seconds = timer.t2 - timer.t1;
    float time = elapsed_seconds/num_repeats;

    return (time);

}



int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineOptions options = read_command_line_options(argc, argv);

    // Filename
    std::string file_name = options.directory + options.input;

    // Read the apr file into the part cell structure
    APRTimer timer;

    timer.verbose_flag = true;

    // APR datastructure
    APR<uint16_t> apr;

    //read file
    apr.read_apr(file_name);

    ///////////////////////////
    ///
    /// Serial Neighbour Iteration (Only Von Neumann (Face) neighbours)
    ///
    /////////////////////////////////

    APRIterator<uint16_t> neighbour_iterator(apr);
    APRIterator<uint16_t> apr_iterator(apr);

    unsigned int number_repeats = options.num_rep;
    unsigned int number_repeats_s = std::max((number_repeats/10),(unsigned int)5);
    pixels_linear_neighbour_access_openmp<uint16_t,uint16_t>(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2),number_repeats,1);

    float time_pixels333 = pixels_linear_neighbour_access_openmp<uint16_t,uint16_t>(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2),number_repeats,1);

    std::cout << "Pixel filter 333 (OpenMP) took: " << time_pixels333*1000 << " ms" << std::endl;
    std::cout << "Pixel filter 333 (OpenMP) per-million pixels: " << 1000*(time_pixels333*1000000)/(1.0f*apr.orginal_dimensions(0)*apr.orginal_dimensions(1)*apr.orginal_dimensions(2)) << " ms" << std::endl;

    float time_pixels555 = pixels_linear_neighbour_access_openmp<uint16_t,uint16_t>(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2),number_repeats,2);

    std::cout << "Pixel filter 555 (OpenMP) took: " << time_pixels555*1000 << " ms" << std::endl;
    std::cout << "Pixel filter 555 (OpenMP) per-million pixels: " << 1000*(time_pixels555*1000000)/(1.0f*apr.orginal_dimensions(0)*apr.orginal_dimensions(1)*apr.orginal_dimensions(2)) << " ms" << std::endl;

    float time_pixels333s = pixels_linear_neighbour_access_serial<uint16_t,uint16_t>(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2),number_repeats_s,1);

    std::cout << "Pixel filter 333 (serial) took: " << time_pixels333s*1000 << " ms" << std::endl;
    std::cout << "Pixel filter 333 (serial) per-million pixels: " << 1000*(time_pixels333s*1000000)/(1.0f*apr.orginal_dimensions(0)*apr.orginal_dimensions(1)*apr.orginal_dimensions(2)) << " ms" << std::endl;

    float time_pixels555s = pixels_linear_neighbour_access_serial<uint16_t,uint16_t>(apr.orginal_dimensions(0),apr.orginal_dimensions(1),apr.orginal_dimensions(2),number_repeats_s,2);

    std::cout << "Pixel filter 555 (serial) took: " << time_pixels555s*1000 << " ms" << std::endl;
    std::cout << "Pixel filter 555 (serial) per-million pixels: " << 1000*(time_pixels555s*1000000)/(1.0f*apr.orginal_dimensions(0)*apr.orginal_dimensions(1)*apr.orginal_dimensions(2)) << " ms" << std::endl;

}


bool command_option_exists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv){

    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"Example_apr_neighbour_access -i input_apr_file -d directory\"" << std::endl;
        std::cerr << usage << std::endl;
        exit(1);
    }

    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }

    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }

    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }


    if(command_option_exists(argv, argv + argc, "-numrep"))
    {
        result.num_rep = std::stoi(std::string(get_command_option(argv, argv + argc, "-numrep")));
    }

    return result;

}

