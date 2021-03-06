__Note: this repository is a mirror of [AdaptiveParticles/LibAPR](https://github.com/AdaptiveParticles/LibAPR). Please refer to the original repository for issues and pull requests.__

# LibAPR - The Adaptive Particle Representation Library

Library for producing and processing on the Adaptive Particle Representation (APR) (For article see: https://www.nature.com/articles/s41467-018-07390-9).

<img src="./docs/apr_lowfps_lossy.gif?raw=true">

Labeled Zebrafish nuclei: Gopi Shah, Huisken Lab ([MPI-CBG](https://www.mpi-cbg.de), Dresden and [Morgridge Institute for Research](https://morgridge.org/research/medical-engineering/huisken-lab/), Madison); see also [Schmid et al., _Nature Communications_ 2017](https://www.nature.com/articles/ncomms3207)

[![Build Status](https://travis-ci.org/AdaptiveParticles/LibAPR.svg?branch=master)](https://travis-ci.org/AdaptiveParticles/LibAPR)
[![DOI](https://zenodo.org/badge/70479293.svg)](https://zenodo.org/badge/latestdoi/70479293)


## Python support

We now provide python wrappers in a separate repository [PyLibAPR](https://github.com/AdaptiveParticles/PyLibAPR)

In addition to providing wrappers for most of the LibAPR functionality, the Python library contains a number of new features that simplify the generation and handling of the APR. For example:

* Interactive APR conversion
* Interactive APR z-slice viewer
* Interactive APR raycast (maximum intensity projection) viewer
* Interactive lossy compression of particle intensities


## Version 2.0 release notes

The library has changed significantly since release 1.1. __There are changes to IO and iteration that are not compatible 
with the older version__. 

* New (additional) linear access data structure, explicitly storing coordinates in the sparse dimension, 
  similar to Compressed Sparse Row.
* Block-based decomposition of the APR generation pipeline, allowing conversion of very large images.
* Expanded and improved functionality for image processing directly on the APR:
  * APR filtering (spatial convolutions).
  * [APRNumerics](./src/numerics/APRNumerics.hpp) module, including e.g. gradient computations and Richardson-Lucy deconvolution.
  * CUDA GPU-accelerated convolutions and RL deconvolution (currently only supports dense 3x3x3 and 5x5x5 stencils)


## Dependencies

* HDF5 1.8.20 or higher
* OpenMP > 3.0 (optional, but recommended)
* CMake 3.6 or higher
* LibTIFF 4.0 or higher


NB: This update to 2.0 introduces changes to IO and iteration that are not compatable with old versions.

If compiling with APR_DENOISE flag the package also requires:
* Eigen3.

## Building

The repository requires submodules, and needs to be cloned recursively:

```
git clone --recursive https://github.com/AdaptiveParticles/LibAPR.git
```

### CMake build options

Several CMake options can be given to control the build. Use the `-D` argument to set each
desired option. For example, to disable OpenMP, change the cmake calls below to
```
cmake -DAPR_USE_OPENMP=OFF ..
```

| Option | Description | Default value |
|:--|:--|:--|
| APR_BUILD_SHARED_LIB | Build shared library | ON |
| APR_BUILD_STATIC_LIB | Build static library | OFF |
| APR_BUILD_EXAMPLES | Build executable examples | OFF |
| APR_TESTS | Build unit tests | OFF |
| APR_BENCHMARK | Build executable performance benchmarks | OFF |
| APR_USE_LIBTIFF | Enable LibTIFF (Required for tests and examples) | ON |
| APR_PREFER_EXTERNAL_GTEST | Use installed gtest instead of included sources | OFF |
| APR_PREFER_EXTERNAL_BLOSC | Use installed blosc instead of included sources | OFF |
| APR_USE_OPENMP | Enable multithreading via OpenMP | ON |
| APR_USE_CUDA | Enable CUDA (Under development - APR conversion pipeline is currently not working with CUDA enabled) | OFF |

### Building on Linux

On Ubuntu, install the `cmake`, `build-essential`, `libhdf5-dev` and `libtiff5-dev` packages (on other distributions, refer to the documentation there, the package names will be similar). OpenMP support is provided by the GCC compiler installed as part of the `build-essential` package.

For denoising support also requires: `libeigen3-dev`

In the directory of the cloned repository, run

```
mkdir build
cd build
cmake ..
make
```

This will create the `libapr.so` library in the `build` directory.

### Building on OSX

On OSX, install the `cmake`, `hdf5` and `libtiff`  [homebrew](https://brew.sh) packages and have the [Xcode command line tools](http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/) installed.

If you want to compile with OpenMP support, also install the `llvm` package (this can also be done using homebrew), as the clang version shipped by Apple currently does not support OpenMP.

In the directory of the cloned repository, run

```
mkdir build
cd build
cmake ..
make
```

This will create the `libapr.dylib` library in the `build` directory.


In case you want to use the homebrew-installed clang (OpenMP support), modify the call to `cmake` above to

```
CC="/usr/local/opt/llvm/bin/clang" CXX="/usr/local/opt/llvm/bin/clang++" LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib" CPPFLAGS="-I/usr/local/opt/llvm/include" cmake ..
```

### Building on Windows

__The simplest way to utilise the library from Windows 10 is through using the Windows Subsystem for Linux; see: https://docs.microsoft.com/en-us/windows/wsl/install-win10 then follow linux instructions.__

__Compilation only works with mingw64/clang or the Intel C++ Compiler, with Intel C++ being the recommended way__

The below instructions for VS can be attempted; however they have not been reproduced.

You need to have Visual Studio 2017 installed, with [the community edition](https://www.visualstudio.com/downloads/) being sufficient. LibAPR does not compile correctly with the default Visual Studio compiler, so you also need to have the [Intel C++ Compiler, 18.0 or higher](https://software.intel.com/en-us/c-compilers) installed. [`cmake`](https://cmake.org/download/) is also a requirement.

Furthermore, you need to have HDF5 installed (binary distribution download at [The HDF Group](http://hdfgroup.org) and LibTIFF (source download from [SimpleSystems](http://www.simplesystems.org/libtiff/). LibTIFF needs to be compiled via `cmake`. LibTIFF's install target will then install the library into `C:\Program Files\tiff`.

In the directory of the cloned repository, run:

```
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -DTIFF_INCLUDE_DIR="C:/Program Files/tiff/include" -DTIFF_LIBRARY="C:/Program Files/tiff/lib/tiff.lib " -DHDF5_ROOT="C:/Program Files/HDF_Group/HDF5/1.8.17"  -T "Intel C++ Compiler 18.0" ..
cmake --build . --config Debug
```

This will set the appropriate hints for Visual Studio to find both LibTIFF and HDF5. This will create the `apr.dll` library in the `build/Debug` directory. If you need a `Release` build, run `cmake --build . --config Release` from the `build` directory.

### Docker build

We provide a working Dockerfile that installs the library within the image in a separate [repository](https://github.com/MSusik/libaprdocker).

Note: not recently tested.

## Examples and Documentation

There are 12 basic examples, that show how to generate and compute with the APR. These can be built by adding 
-DAPR_BUILD_EXAMPLES=ON to the cmake command.

| Example | How to ... |
|:--|:--|
| [Example_get_apr](./examples/Example_get_apr.cpp) | create an APR from a TIFF and store as hdf5. |
| [Example_get_apr_by_block](./examples/Example_get_apr_by_block.cpp) | create an APR from a (potentially large) TIFF, by decomposing it into smaller blocks, and store as hdf5.
| [Example_apr_iterate](./examples/Example_apr_iterate.cpp) | iterate over APR particles and their spatial properties. |
| [Example_apr_tree](./examples/Example_apr_tree.cpp) | iterate over interior APR tree particles and their spatial properties. |
| [Example_neighbour_access](./examples/Example_neighbour_access.cpp) | access particle and face neighbours. |
| [Example_compress_apr](./examples/Example_compress_apr.cpp) |  additionally compress the intensities stored in an APR. |
| [Example_random_access](./examples/Example_random_access.cpp) | perform random access operations on particles. |
| [Example_ray_cast](./examples/Example_ray_cast.cpp) | perform a maximum intensity projection ray cast directly on the APR. |
| [Example_reconstruct_image](./examples/Example_reconstruct_image.cpp) | reconstruct a pixel image from an APR. |
| [Example_compute_gradient](./examples/Example_compute_gradient.cpp) | compute the gradient magnitude of an APR. |
| [Example_apr_filter](./examples/Example_apr_filter.cpp) | apply a filter (convolution) to an APR. |
| [Example_apr_deconvolution](./examples/Example_apr_deconvolution.cpp) | perform Richardson-Lucy deconvolution on an APR. |

All examples except `Example_get_apr` and `Example_get_apr_by_block` require an already produced APR, such as those created by `Example_get_apr*`.

For tutorial on how to use the examples, and explanation of data-structures see [the library guide](./docs/lib_guide.pdf).

## LibAPR Tests

The testing framework can be turned on by adding -DAPR_TESTS=ON to the cmake command. All tests can then be run by executing
```
ctest
```
on the command line in your build folder. Please let us know by creating an issue, if any of these tests are failing on your machine.

## Java wrappers

Basic Java wrappers can be found at [LibAPR-java-wrapper](https://github.com/krzysg/LibAPR-java-wrapper)

## Coming soon

* Improved documentation and updated library guide.
* More examples of APR-based image processing and segmentation.
* CUDA GPU-accelerated APR generation and additional processing options.
* Time series support.

## Contact us

If anything is not working as you think it should, or would like it to, please get in touch with us!! Further, dont hesitate to contact us if you have a project or algorithm you would like to try using the APR for. We would be glad to help!

[![Join the chat at https://gitter.im/LibAPR](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/LibAPR/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Citing this work

If you use this library in an academic context, please cite the following paper:

* Cheeseman, G??nther, Gonciarz, Susik, Sbalzarini: _Adaptive Particle Representation of Fluorescence Microscopy Images_ (Nature Communications, 2018) https://doi.org/10.1038/s41467-018-07390-9
