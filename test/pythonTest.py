# Find path to script - all test files are kept there
import os, sys
myPath = os.path.dirname(os.path.abspath(__file__))

# Add path to location of the python APR module (your build folder)
#sys.path.insert(0, "../cmake-build-release")

# Check if APR can be read and if gives correct dimensions
#import pyApr
import numpy as np
import fnmatch


def main():
    """
    If you get ImportErrors, or if you have more than one build containing a file pyApr.so, you may want to
    manually supply the path by changing the block below to
                        sys.path.insert(0, '/path/to/dir/containing/module/')
    """
    # -------------------------------------------------- #
    print("Searching for pyApr module...")
    basePath = os.path.join(myPath, os.pardir)
    for root, dirnames, filenames in os.walk(basePath):
        if fnmatch.filter(filenames, 'pyApr.so'):
            sys.path.insert(0, root)
            print('pyApr module found in {}'.format(root))
    # -------------------------------------------------- #

    import pyApr

    successes = 0

    print("Instantiating objects...")
    success = True
    try:
        apr = pyApr.AprShort()
    except:
        print("Could not instantiate pyApr.AprShort object")
        success = False

    try:
        pars = pyApr.APRParameters()
    except:
        print("Could not instantiate pyApr.APRParameters object")
        success = False

    if success:
        successes += 1
        print("success!")

    print("Testing read APR...")
    try:
        apr = pyApr.AprShort()
        apr.read_apr( os.path.join(myPath, 'files/Apr/sphere_120/sphere_apr.h5') )
        successes += 1
        print("success!")
    except:
        print("**** failed ****")


    print("Testing write APR...")
    try:
        outpath = os.path.join(myPath, 'writeaprtest')

        if apr:
            apr.write_apr(outpath)
        else:
            apr = pyApr.AprShort()
            apr.write_apr(outpath)

        successes += 1
        print("success!")

        outpath = os.path.join(myPath, 'writeaprtest_apr.h5')
        if os.path.exists(outpath):
            print('removing temporarily created APR file')
            os.remove(outpath)
        else:
            print('could not find temporarily created APR file for removal')
    except:
        print("**** failed ****")


    print("Testing piecewise constant pixel image reconstruction...")
    try:
        img = np.array(apr.reconstruct(), copy=False)
        assert img.shape == (120, 120, 120)
        successes += 1
        print("success!")
    except:
        print("**** failed ****")


    print("Testing smooth pixel image reconstruction...")
    try:
        img = np.array(apr.reconstruct_smooth(), copy=False)
        assert img.shape == (120, 120, 120)
        successes += 1
        print("success!")
    except:
        print("**** failed ****")


    print("Testing set APRParameters...")
    try:
        if not pars:
            pars = pyApr.APRParameters()

        pars.Ip_th = 0
        pars.sigma_th = 0
        pars.sigma_th_max = 0
        pars.rel_error = 0.1
        pars.lmbda = 1
        pars.auto_parameters = False

        apr.set_parameters(pars)

        successes += 1
        print("success!")
    except:
        print("**** failed ****")


    print("Testing compute APR from image file")
    try:
        imgpath = os.path.join(myPath, 'files/Apr/sphere_120/sphere_original.tif')
        apr = pyApr.AprShort()

        pars = pyApr.APRParameters()
        pars.rel_error = 0.1
        pars.auto_parameters = True
        apr.set_parameters(pars)

        apr.get_apr_from_file(imgpath)
        successes += 1
        print("success!")
    except:
        print("**** failed ****")


    print("Testing compute APR from numpy array")
    try:
        apr = pyApr.AprShort()

        pars = pyApr.APRParameters()
        pars.rel_error = 0.1
        pars.auto_parameters = True
        apr.set_parameters(pars)

        img = np.zeros((30, 30, 30), dtype=np.uint16)
        img[4:9, 13:21, 22:27] = 1200

        apr.get_apr_from_array(img)
        successes += 1
        print("success!")
    except:
        print("**** failed ****")

    print('PASSED {}/8 TESTS'.format(successes))


if __name__ == '__main__':
    main()

