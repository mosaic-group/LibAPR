//
// Created by cheesema on 09.07.18.
//

#ifndef LIBAPR_APRCONVERTERBATCH_HPP
#define LIBAPR_APRCONVERTERBATCH_HPP

////////////////////////////////
///
/// Bevan Cheeseman 2018
///
/// APR Converter class handles the methods and functions for creating an APR from an input image
///
////////////////////////////////

#include "data_structures/Mesh/PixelData.hpp"
#include "../io/TiffUtils.hpp"
#include "../data_structures/APR/APR.hpp"

#include "APRConverter.hpp"

#include "ComputeGradient.hpp"
#include "LocalIntensityScale.hpp"
#include "LocalParticleCellSet.hpp"
#include "PullingSchemeSparse.hpp"

#ifdef APR_USE_CUDA
#include "algorithm/ComputeGradientCuda.hpp"
#endif

struct imagePatch {

    uint64_t x_begin;
    uint64_t x_end;
    uint64_t x_offset;

    uint64_t y_begin;
    uint64_t y_end;
    uint64_t y_offset;

    uint64_t z_begin;
    uint64_t z_end;
    uint64_t z_offset;

};


template<typename ImageType>
class APRConverterBatch: public LocalIntensityScale, public ComputeGradient, public LocalParticleCellSet, public PullingSchemeSparse {

public:


    APRParameters par;
    APRTimer fine_grained_timer;
    APRTimer method_timer;
    APRTimer total_timer;
    APRTimer allocation_timer;
    APRTimer computation_timer;

    bool get_apr(APR<ImageType> &aAPR) {
        apr = &aAPR;

        TiffUtils::TiffInfo inputTiff(par.input_dir + par.input_image_name);
        if (!inputTiff.isFileOpened()) return false;

        if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_UINT8) {
            return get_apr_batch_method_from_file<uint8_t>(aAPR, inputTiff);
        } else if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_FLOAT) {
            return get_apr_batch_method_from_file<float>(aAPR, inputTiff);
        } else if (inputTiff.iType == TiffUtils::TiffInfo::TiffType::TIFF_UINT16) {
            return get_apr_batch_method_from_file<uint16_t>(aAPR, inputTiff);
        } else {
            std::cerr << "Wrong file type" << std::endl;
            return false;
        }
    };

    //get apr without setting parameters, and with an already loaded image.
    template<typename T>
    bool get_apr_method_patch(APR<ImageType> &aAPR, PixelData<T>& input_image,imagePatch &patch);

    //template<typename T>
   // void auto_parameters(const PixelData<T> &input_img);

private:

    //pointer to the APR structure so member functions can have access if they need
    const APR<ImageType> *apr;

    void init_apr(APR<ImageType>& aAPR,const TiffUtils::TiffInfo &aTiffFile);

    template<typename T>
    bool get_apr_batch_method_from_file(APR<ImageType> &aAPR, const TiffUtils::TiffInfo &aTiffFile);

public:
    void get_local_particle_cell_set(PixelData<ImageType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2,imagePatch& patch);
};

/**
 * Main method for constructing the APR from an input image
 */
template<typename ImageType> template<typename T>
bool APRConverterBatch<ImageType>::get_apr_batch_method_from_file(APR<ImageType> &aAPR, const TiffUtils::TiffInfo &aTiffFile) {

    allocation_timer.start_timer("read tif input image");
    PixelData<T> inputImage = TiffUtils::getMesh<T>(aTiffFile);
    allocation_timer.stop_timer();

    //method_timer.start_timer("calculate automatic parameters");


    //can't do this getting from image.. #TODO: how would we do this normalization?
    // #TODO how do we do autoparameters?

    //auto_parameters(inputImage);

    imagePatch patch;

    patch.x_begin = 0;
    patch.x_end = inputImage.x_num;
    patch.x_offset = 0;

    patch.y_begin = 0;
    patch.y_end = inputImage.y_num;
    patch.y_offset = 0;

    patch.z_begin = 0;
    patch.z_end = inputImage.z_num;
    patch.z_offset = 0;

    //PixelData<T> inputImage;

    init_apr(aAPR, aTiffFile);

    method_timer.start_timer("initialize_particle_cell_tree");
    initialize_particle_cell_tree(aAPR);
    method_timer.stop_timer();

    get_apr_method_patch(aAPR, inputImage,patch);

    return true;
}

/**
 * Main method for constructing the APR from an input image
 */
template<typename ImageType> template<typename T>
bool APRConverterBatch<ImageType>::get_apr_method_patch(APR<ImageType> &aAPR, PixelData<T>& input_image,imagePatch &patch) {
    apr = &aAPR; // in case it was called directly

    total_timer.start_timer("Total_pipeline_excluding_IO");

    ////////////////////////////////////////
    /// Memory allocation of variables
    ////////////////////////////////////////

    //assuming uint16, the total memory cost shoudl be approximately (1 + 1 + 1/8 + 2/8 + 2/8) = 2 5/8 original image size in u16bit
    //storage of the particle cell tree for computing the pulling scheme
    allocation_timer.start_timer("init and copy image");
    PixelData<ImageType> image_temp(input_image, false /* don't copy */); // global image variable useful for passing between methods, or re-using memory (should be the only full sized copy of the image)
    PixelData<ImageType> grad_temp; // should be a down-sampled image
    grad_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0);
    PixelData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
    local_scale_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);
    PixelData<float> local_scale_temp2;
    local_scale_temp2.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);
    allocation_timer.stop_timer();

    /////////////////////////////////
    /// Pipeline
    ////////////////////////

    computation_timer.start_timer("Calculations");

    fine_grained_timer.start_timer("offset image");
    //offset image by factor (this is required if there are zero areas in the background with uint16_t and uint8_t images, as the Bspline co-efficients otherwise may be negative!)
    // Warning both of these could result in over-flow (if your image is non zero, with a 'buffer' and has intensities up to uint16_t maximum value then set image_type = "", i.e. uncomment the following line)
    float bspline_offset = 0;
    if (std::is_same<uint16_t, ImageType>::value) {
        bspline_offset = 100;
        image_temp.copyFromMeshWithUnaryOp(input_image, [=](const auto &a) { return (a + bspline_offset); });
    } else if (std::is_same<uint8_t, ImageType>::value){
        bspline_offset = 5;
        image_temp.copyFromMeshWithUnaryOp(input_image, [=](const auto &a) { return (a + bspline_offset); });
    } else {
        image_temp.copyFromMesh(input_image);
    }
    fine_grained_timer.stop_timer();

#ifndef APR_USE_CUDA

    APRConverter<ImageType> aprConverter;
    aprConverter.par = par; //copy parameters

    //method_timer.verbose_flag = true;
    method_timer.start_timer("compute_gradient_magnitude_using_bsplines");
    aprConverter.get_gradient(image_temp, grad_temp, local_scale_temp, local_scale_temp2, bspline_offset, par);
    method_timer.stop_timer();

    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "gradient_step.tif", grad_temp);
    }

    method_timer.start_timer("compute_local_intensity_scale");
    aprConverter.get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
    method_timer.stop_timer();
    //method_timer.verbose_flag = false;

    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "local_intensity_scale_step.tif", local_scale_temp);
    }
#else
    method_timer.start_timer("compute_gradient_magnitude_using_bsplines and local instensity scale CUDA");
    getFullPipeline(image_temp, grad_temp, local_scale_temp, local_scale_temp2,bspline_offset, par);
    method_timer.stop_timer();
#endif

    method_timer.start_timer("compute_local_particle_set");
    get_local_particle_cell_set(grad_temp, local_scale_temp, local_scale_temp2,patch);
    method_timer.stop_timer();

    computation_timer.stop_timer();

    total_timer.stop_timer();

    return true;
}

template<typename ImageType>
void APRConverterBatch<ImageType>::get_local_particle_cell_set(PixelData<ImageType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2,imagePatch& patch) {
    //
    //  Computes the Local Particle Cell Set from a down-sampled local intensity scale (\sigma) and gradient magnitude
    //
    //  Down-sampled due to the Equivalence Optimization
    //

    fine_grained_timer.start_timer("compute_level_first");
    //divide gradient magnitude by Local Intensity Scale (first step in calculating the Local Resolution Estimate L(y), minus constants)
#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
    for(size_t i = 0; i < grad_temp.mesh.size(); ++i) {
        local_scale_temp.mesh[i] = (1.0*grad_temp.mesh[i])/(local_scale_temp.mesh[i]*1.0);
    }
    fine_grained_timer.stop_timer();

    float min_dim = std::min(par.dy,std::min(par.dx,par.dz));
    float level_factor = pow(2,(*apr).level_max())*min_dim;

    int l_max = (*apr).level_max() - 1;
    int l_min = (*apr).level_min();

    fine_grained_timer.start_timer("compute_level_second");
    //incorporate other factors and compute the level of the Particle Cell, effectively construct LPC L_n
    compute_level_for_array(local_scale_temp,level_factor,par.rel_error);

    if(par.output_steps){
        TiffUtils::saveMeshAsTiff(par.output_dir + "local_particle_set_level_step.tif", local_scale_temp);
    }


    fill(l_max,local_scale_temp);
    fine_grained_timer.stop_timer();

    fine_grained_timer.start_timer("level_loop_initialize_tree");
    for(int l_ = l_max - 1; l_ >= l_min; l_--){

        //down sample the resolution level k, using a max reduction
        downsample(local_scale_temp, local_scale_temp2,
                   [](const float &x, const float &y) -> float { return std::max(x, y); },
                   [](const float &x) -> float { return x; }, true);
        //for those value of level k, add to the hash table
        fill(l_,local_scale_temp2);
        //assign the previous mesh to now be resampled.
        local_scale_temp.swap(local_scale_temp2);
    }
    fine_grained_timer.stop_timer();
}


template<typename ImageType>
void APRConverterBatch<ImageType>::init_apr(APR<ImageType>& aAPR,const TiffUtils::TiffInfo &aTiffFile){
    //
    //  Initializing the size of the APR, min and maximum level (in the data structures it is called depth)
    //

    aAPR.apr_access.org_dims[0] = aTiffFile.iImgWidth;
    aAPR.apr_access.org_dims[1] = aTiffFile.iImgHeight;
    aAPR.apr_access.org_dims[2] = aTiffFile.iNumberOfDirectories;

    int max_dim = std::max(std::max(aAPR.apr_access.org_dims[1], aAPR.apr_access.org_dims[0]), aAPR.apr_access.org_dims[2]);
    int min_dim = std::min(std::min(aAPR.apr_access.org_dims[1], aAPR.apr_access.org_dims[0]), aAPR.apr_access.org_dims[2]);

    int levelMax = ceil(std::log2(max_dim));
    // TODO: why minimum level is forced here to be 2?
    int levelMin = std::max( (int)(levelMax - floor(std::log2(min_dim))), 2);

    aAPR.apr_access.level_min = levelMin;
    aAPR.apr_access.level_max = levelMax;

    aAPR.parameters = par;
}


#endif //LIBAPR_APRCONVERTERBATCH_HPP