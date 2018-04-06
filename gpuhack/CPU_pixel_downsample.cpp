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

    unsigned int num_rep = options.num_rep;

    //Down sample

    MeshData<uint16_t> input_data;
    input_data.init((int)apr.orginal_dimensions(0),(int)apr.orginal_dimensions(1),(int)apr.orginal_dimensions(2),23);


    std::vector<MeshData<uint16_t>> downsampled_img;
    //Down-sample the image for particle intensity estimation

    /*
     *
     *  Initialization of the pyramid data structure
     *
     */

    downsampled_img.resize(apr.level_max()+1); // each level is kept at same index

     size_t z_num_ds = input_data.z_num;
     size_t x_num_ds = input_data.x_num;
     size_t y_num_ds = input_data.y_num;

    for (int level = (apr.level_max()); level >= apr.level_min(); --level) {

        downsampled_img[level].init(y_num_ds, x_num_ds, z_num_ds);

        z_num_ds = ceil(z_num_ds/2.0);
        x_num_ds = ceil(x_num_ds/2.0);
        y_num_ds = ceil(y_num_ds/2.0);
    }

    downsampled_img.back().swap(input_data);


    timer.start_timer("downsample_pyramid");

    for (int i = 0; i < num_rep; ++i) {
        auto sum = [](const float x, const float y) -> float { return x + y; };
        auto divide_by_8 = [](const float x) -> float { return x/8.0; };
        for (size_t level = apr.level_max(); level > apr.level_min(); --level) {
            downsample(downsampled_img[level], downsampled_img[level - 1], sum, divide_by_8, false);
        }
    }

    timer.stop_timer();

    float cpu_iterate_time_batch = timer.timings.back();
    std::cout << "CPU down-sample average total: " << (cpu_iterate_time_batch/(num_rep*1.0f))*1000 << " ms" << std::endl;
    std::cout << "CPU down-sample average per million:  " << (cpu_iterate_time_batch/(num_rep*1.0f*input_data.mesh.size()))*1000.0*1000000.0f << " ms" << std::endl;



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



