///////////////////
//
//  Bevan Cheeseman 2016
//
//  Class for storing and accessing cell or particle neighbours
//
///////////////

#ifndef PARTPLAY_PARTCELLNEIGH_HPP
#define PARTPLAY_PARTCELLNEIGH_HPP
 // type T data structure base type

#include "PartCellData.hpp"

#define NUM_FACES 6

template<typename T>
class PartCellNeigh {
    
public:
    
    std::vector<std::vector<T>> neigh_face; //the neighbours arranged by face
    
    T curr; //current cell or particle
    
    PartCellNeigh(){
        neigh_face.resize(NUM_FACES);
        curr = 0;
    };
    
    
    
    
private:
    
};

#endif //PARTPLAY_PARTNEIGH_HPP