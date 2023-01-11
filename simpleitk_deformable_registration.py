
import SimpleITK as sitk
import time
import numpy as np



##########################################################################
# Calc back functions to monitor the registration
def on_update_multiresolution_event():
    print('Next resolution')

def on_def_reg_iteration_event(registration):
    global file_def_reg
    file_def_reg.write('{}   {}\n'.format(registration.GetOptimizerIteration(), registration.GetMetricValue()))
    print(registration.GetOptimizerIteration(), registration.GetMetricValue())

def on_rigid_iteration_event(registration):
    global file_rigid
    file_rigid.write('{}   {}\n'.format(registration.GetOptimizerIteration(), registration.GetMetricValue()))
    print(registration.GetOptimizerIteration(), registration.GetMetricValue())
    

##########################################################################
def determine_bspline_dimensions(image : sitk.Image, resolution : float) -> tuple[float, float, float]:
    """ 
    Determine the dimensions of the bspline given a resolution such that the bspline 
    grid will cover the entire image.
    """
    
    if image.GetDirection() != (1, 0, 0, 0, 1, 0, 0, 0, 1):
        raise Exception('Image direction is not unit transform {}.'.format(image.GetDirection()))

    spacing = np.array(image.GetSpacing())
    dim = np.array(image.GetSize())
    size = spacing * dim

    bspline_dim = np.array(size / resolution + 0.5).astype(int)

    return bspline_dim.tolist()

########################################################################################
#
# Deformable image registration to align the reference CT (moving) extended CBCT (fixed)
# 
# No need for a rigid pre-reg since that has already been done.
#
def deformable_registration(ct_ref_float, cbct_extended):
    
    print('ACCURATE Deformable registration used!')

    start_time = time.time()

    fixed_image = cbct_extended
    moving_image = ct_ref_float
    

    deformable_registration = sitk.ImageRegistrationMethod()

    #
    # Similarity metric settings.
    #
    # CROSS CORRELATION
    deformable_registration.SetMetricAsCorrelation()
    deformable_registration.SetMetricSamplingPercentage(0.10) # number of pixels that are sampled


    # use mask to only evaluate pixels that are inside patient
    #deformable_registration.SetMetricFixedMask(fixed_mask)
    #deformable_registration.SetMetricMovingMask(fixed_mask)


    #
    # Different optimisation methods can be explored, here are three (common) alternatives
    # GradientDescent (simplest, i.e take a step in the negative direction ofthe gradient)
    # ConjugateGradient (slightly better, taking more intelligent directions)
    # LBFGSB (advanced, also taking the (approx) 2nd derivateives (Hessian) into account)
    #
    # It is not a given that the more advanced gives better results (but likely faster)
    # The parameters need to be tuned somewhat to get the best result (speed)
    #
    deformable_registration.SetOptimizerAsConjugateGradientLineSearch(
        learningRate=1,
        numberOfIterations=50,
        convergenceMinimumValue=1e-5,
        convergenceWindowSize=10)


    deformable_registration.SetInterpolator(sitk.sitkLinear)
    deformable_registration.SetOptimizerScalesFromPhysicalShift()

    transformDomainMeshSize = determine_bspline_dimensions(fixed_image, 40) # 40 is the initial bspline dim.
    tx = sitk.BSplineTransformInitializer(fixed_image,
                                          transformDomainMeshSize )
    print("Initial transform:")
    print(tx)
    print()
    print()

    # this is where we specify the transformation model (i.e. deformable registration using B-Splines)
    deformable_registration.SetInitialTransformAsBSpline(tx,
                                                    inPlace=True,
                                                    scaleFactors=[1,2,4])

    # Setup the multi-resolution framework.
    deformable_registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    deformable_registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    deformable_registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    
    # call back functions to monitor registration
    deformable_registration.AddCommand(
    sitk.sitkMultiResolutionIterationEvent, lambda: on_update_multiresolution_event
    )
    deformable_registration.AddCommand(
        sitk.sitkIterationEvent, lambda: on_def_reg_iteration_event(deformable_registration)
    )

    final_dir_transform = deformable_registration.Execute( fixed_image, moving_image)
    
    print("Final Number of Parameters:", final_dir_transform.GetNumberOfParameters())
    print('final transform', final_dir_transform)

    synthetic_ct = sitk.Resample(
        ct_ref_float,
        cbct_extended,
        final_dir_transform,
        sitk.sitkLinear,
        -1000,
        ct_ref_float.GetPixelID(),
    )

    print(time.time() - start_time)

    return synthetic_ct


def deformable_registration_fast(ct_ref_float, cbct_extended):
    
    print('FAST INACCURATE Deformable registration used!')

    import time
    start_time = time.time()

    fixed_image = cbct_extended
    moving_image = ct_ref_float

    deformable_registration = sitk.ImageRegistrationMethod()

    #
    # Similarity metric settings.
    #
    # CROSS CORRELATION
    deformable_registration.SetMetricAsCorrelation()
    deformable_registration.SetMetricSamplingPercentage(0.01) # number of pixels that are sampled

    
    #
    # Different optimisation methods can be explored, here are three (common) alternatives
    # GradientDescent (simplest, i.e take a step in the negative direction ofthe gradient)
    # ConjugateGradient (slightly better, taking more intelligent directions)
    # LBFGSB (advanced, also taking the (approx) 2nd derivateives (Hessian) into account)
    #
    # It is not a given that the more advanced gives better results (but likely faster)
    # The parameters need to be tuned somewhat to get the best result (speed)
    #
    deformable_registration.SetOptimizerAsConjugateGradientLineSearch(
        learningRate=1,
        numberOfIterations=2,
        convergenceMinimumValue=1e-4,
        convergenceWindowSize=5)

    deformable_registration.SetInterpolator(sitk.sitkLinear)
    deformable_registration.SetOptimizerScalesFromPhysicalShift()

    transformDomainMeshSize = determine_bspline_dimensions(fixed_image, 200) # 200 is the initial bspline dim.
    tx = sitk.BSplineTransformInitializer(fixed_image,
                                          transformDomainMeshSize )

    coeff_images = tx.GetCoefficientImages()
    print(coeff_images[0].GetSpacing(), coeff_images[0].GetSize())
    print("Initial transform:")
    print(tx)
    print()
    print()

    # this is where we specify the transformation model (i.e. deformable registration using B-Splines)
    deformable_registration.SetInitialTransformAsBSpline(tx,
                                                    inPlace=True,
                                                    scaleFactors=[1,2])

    # Setup the multi-resolution framework.
    deformable_registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])
    deformable_registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1])
    deformable_registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    
    # call back functions to monitor registration
    deformable_registration.AddCommand(
    sitk.sitkMultiResolutionIterationEvent, on_update_multiresolution_event
    )
    deformable_registration.AddCommand(
        sitk.sitkIterationEvent, lambda: on_def_reg_iteration_event(deformable_registration)
    )

    final_dir_transform = deformable_registration.Execute( fixed_image, moving_image)

    print("Final Number of Parameters:", final_dir_transform.GetNumberOfParameters())
    print('final transform', final_dir_transform)
    
    synthetic_ct = sitk.Resample(
        ct_ref_float,
        cbct_extended,
        final_dir_transform,
        sitk.sitkLinear,
        -1000,
        ct_ref_float.GetPixelID(),
    )

    print(time.time() - start_time)

    return synthetic_ct