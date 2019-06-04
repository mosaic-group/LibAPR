//
// Created by cheesema on 2019-05-23.
//

#ifndef LIBAPR_LINEARITERATOR_HPP
#define LIBAPR_LINEARITERATOR_HPP

#include "APRAccessStructures.hpp"
#include "GenIterator.hpp"
#include "LinearAccess.hpp"

class LinearIterator: public GenIterator {

    uint64_t current_index;
    //uint64_t begin_index;

    // TODO: need to add the level,x,z,y into here..
    // Also need to add the datageneration. (should make it lazy as possible)
    LinearAccess* linearAccess;

public:

    explicit LinearIterator(LinearAccess& apr_access_,GenInfo& genInfo_) {
        linearAccess = &apr_access_;
        genInfo = &genInfo_;
        current_index = 0;
    }

    operator uint64_t() override { return current_index; }

    uint16_t y() const override {
        return linearAccess->y_vec[current_index];
    }

    //defining the iterator interface
    inline void operator++ (int) override{
        current_index++;
    }

    inline void operator++ () override{
        current_index++;
    }

    inline uint64_t end() override{
        return end_index;
    }

    inline uint64_t begin(const uint16_t level_,const uint16_t z_,const uint16_t x_) override {

        const auto level_start = linearAccess->level_xz_vec[level_-1]; //do i make these variables in the class
        const auto xz_start = level_start + x_ + z_*x_num(level_);

        //intialize
        current_index = linearAccess->xz_end_vec[xz_start-1];
        end_index = linearAccess->xz_end_vec[xz_start];

        return current_index;
    }

};


#endif //LIBAPR_LINEARITERATOR_HPP
