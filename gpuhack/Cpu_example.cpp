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

    ///////////////////////////
    ///
    /// Serial Neighbour Iteration (Only Von Neumann (Face) neighbours)
    ///
    /////////////////////////////////

    APRIterator<uint16_t> neighbour_iterator(apr);
    APRIterator<uint16_t> apr_iterator(apr);

    int num_rep = 50;

    timer.start_timer("APR serial iterator neighbours loop");

    //Basic serial iteration over all particles
    uint64_t particle_number;
    //Basic serial iteration over all particles


    ExtraParticleData<float> part_sum_standard(apr);

    APRTree<uint16_t> apr_tree(apr);

    ExtraParticleData<float> tree_intensity(apr_tree);
    ExtraParticleData<uint8_t> tree_counter(apr_tree);

    APRTreeIterator<uint16_t> treeIterator(apr_tree);

    const auto ldiff = apr_iterator.level_max() - apr_iterator.level_min();

    for( int t = 0; t<ldiff;++t ){
        for (unsigned int level = apr_iterator.level_min(); level <= (apr_iterator.level_max() - t); ++level) {

            for (particle_number = apr_iterator.particles_level_begin(level);
                 particle_number < apr_iterator.particles_level_end(level); ++particle_number) {

                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                treeIterator.set_iterator_to_parent(apr_iterator);

                // //recursive mean
                if(t!=0)
                    tree_intensity[treeIterator]  = (tree_intensity[treeIterator]*tree_counter[treeIterator]*1.0f + tree_intensity[treeIterator]);
                else{
                    tree_intensity[treeIterator]  = (tree_intensity[treeIterator]*tree_counter[treeIterator]*1.0f + apr.particles_intensities[apr_iterator]);
                    tree_intensity[treeIterator] /= (1.0f*(tree_counter[treeIterator]+1.0f));
                }
                tree_counter[treeIterator]++;
            }
        }
    }


    ExtraParticleData<float> part_sum(apr);


    const int8_t dir_y[6] = { 1, -1, 0, 0, 0, 0};
    const int8_t dir_x[6] = { 0, 0, 1, -1, 0, 0};
    const int8_t dir_z[6] = { 0, 0, 0, 0, 1, -1};


    timer.start_timer("APR parallel iterator neighbour loop by level GRAPH FACE");

    for (int i = 0; i < num_rep; ++i) {
        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator, neighbour_iterator)
#endif
            for (particle_number = apr_iterator.particles_level_begin(level);
                 particle_number < apr_iterator.particles_level_end(level); ++particle_number) {

                //needed step for any  loop (update to the next part)
                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                float temp2 = 0;

                //loop over all the neighbours and set the neighbour iterator to it
                for (int direction = 0; direction < 6; ++direction) {
                    apr_iterator.find_neighbours_in_direction(direction);
                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]

                    float temp = 0;
                    float counter = 0;

                    for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {
                        if(apr_iterator.number_neighbours_in_direction(direction) ==1) {
                            if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                                //neighbour_iterator works just like apr, and apr_parallel_iterator (you could also call neighbours)
                                temp += apr.particles_intensities[neighbour_iterator];
                                counter++;
                            }
                        }

                    }
                    if(counter > 0) {
                        temp2 += temp / counter;
                    }

                }

                part_sum[apr_iterator] = temp2/6.0f;

            }
        }
    }

    timer.stop_timer();

    float graph_time = timer.timings.back();



    // --------------------------------------- USE DENSE ITERATION FOR XCHECKS ---------------------------------------
    timer.start_timer("APR parallel iterator neighbour loop by level ISOTROPIC PATCH FACE");

    for (int i = 0; i < num_rep; ++i) {
        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) private(particle_number) firstprivate(apr_iterator, neighbour_iterator,treeIterator)
#endif
            for (particle_number = apr_iterator.particles_level_begin(level);
                 particle_number < apr_iterator.particles_level_end(level); ++particle_number) {

                //needed step for any  loop (update to the next part)
                apr_iterator.set_iterator_to_particle_by_number(particle_number);

                float temp2 = 0;

                //loop over all the neighbours and set the neighbour iterator to it
                for (int direction = 0; direction < 6; ++direction) {
                    apr_iterator.find_neighbours_in_direction(direction);
                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]

                    float temp = 0;
                    float counter = 0;

                    for (int index = 0; index < apr_iterator.number_neighbours_in_direction(direction); ++index) {
                        if(apr_iterator.number_neighbours_in_direction(direction) ==1) {
                            if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                                //neighbour_iterator works just like apr, and apr_parallel_iterator (you could also call neighbours)
                                temp2 += apr.particles_intensities[neighbour_iterator];
                                //counter++;

                            }
                        } else{
                            if (neighbour_iterator.set_neighbour_iterator(apr_iterator, direction, index)) {
                                treeIterator.set_iterator_to_parent(neighbour_iterator);

                                temp2 += tree_intensity[treeIterator];
                                break;
                            }

                        }


                    }

                }

                part_sum_standard[apr_iterator] = temp2/6.0f;


            }
        }
    }

    timer.stop_timer();

    float iso_time = timer.timings.back();

    std::cout << "GRAPH RA took: " << 1000*graph_time/(num_rep*1.0f) << " ms " << std::endl;
    std::cout << "ISO RA took: " << 1000*iso_time/(num_rep*1.0f) << " ms " << std::endl;






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

    return result;

}

