
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import distance_transform_edt
import os, glob
from datetime import datetime
from .anonymise import write_image_series
from .elastix_deformable_registration import deformable_registration as elx_deformable_registration
from .elastix_deformable_registration import extract_contour_points, output_rigid_points
#from .simpleitk_deformable_registration import deformable_registration as sitk_deformable_registration

import pydicom
from TreatmentSimulation.DicomIO.RtStruct import RtStruct
from TreatmentSimulation.Patient.Roi import Roi

image_path = r'P:\TERAPI\FYSIKER\David_Tilly\Rect-5-study\PatientData'
image_path = r'/home/david/work/recti/datasets'
image_path = r'C:\Users\tid003\Dropbox\Work\5-day-recti\datasets'

#image_path = r'C:\temp\5-days-recti\PatientData'

file_rigid = None
file_def_reg = None

##########################################################################
def read_ref_ct(patient_id):
    ct_path = os.path.join(image_path, '{}_anonymized'.format(patient_id), 'Reference')
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ct_path)

    # remove all files starting with filename 'R'
    dicom_names = list(filter( lambda fname: not os.path.basename(fname).startswith('R'), dicom_names))

    print('{} CT files found'.format(len(dicom_names)))
    reader.SetFileNames(dicom_names)
    ct_ref = reader.Execute()

    ct_ref_float = sitk.Cast(ct_ref, sitk.sitkFloat32)

    return ct_ref_float

##########################################################################
def read_ref_rs(patient_id):
    rs_path = os.path.join(image_path, '{}_anonymized'.format(patient_id), 'Reference', 'RS*dcm')
    files = glob.glob(rs_path)
    if len(files) != 1:
        raise Exception('Could not read reference structure set {}'.format(rs_path))

    rtss = RtStruct(files[0])
    rtss.parse()
    
    return rtss


##########################################################################
def crop_to_bounding_box(image : sitk.Image, bounding_box) -> sitk.Image:
    """ Cropt the image using a bounding box defined by [low corner, high corner] """

    if image.GetDirection() != (1, 0, 0, 0, 1, 0, 0, 0, 1):
        raise Exception('Image direction is not unit transform {}.'.format(image.GetDirection()))
    
    # find indices of bounding box corner 
    low_corner, high_corner = bounding_box
    low_index = image.TransformPhysicalPointToIndex(low_corner)
    high_index = image.TransformPhysicalPointToIndex(high_corner)

    # crop and update origin
    cropped = image[low_index[0]:high_index[0], low_index[1]:high_index[1], low_index[2]:high_index[2]]
    new_origin = image.TransformIndexToPhysicalPoint(low_index)
    cropped.SetOrigin(new_origin)

    return cropped


##########################################################################
def read_pixel_info(patient_id):

    ct_path = os.path.join(image_path, '{}_anonymized'.format(patient_id), 'Reference')
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ct_path)

    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_names[0])
    _ = reader.Execute()

    rescale_intercept = reader.GetMetaData("0028|1052")
    rescale_slope = reader.GetMetaData("0028|1053")
    
    return rescale_intercept, rescale_slope 

##########################################################################
def read_cbct(patient_id, fraction):

    cbct_path = os.path.join(image_path, '{}_anonymized'.format(patient_id), 'CBCT', fraction)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(cbct_path)
    print('{} CBCT files found'.format(len(dicom_names)))
    
    reader.SetFileNames(dicom_names)
    cbct = reader.Execute()

    cbct_float = sitk.Cast(cbct, sitk.sitkFloat32)

    return cbct_float



##########################################################################
#
# Match cthe CBCT image pixel histogram to the CT pixel histogram
#
def histogram_matching(cbct, ct):

    ct_np = sitk.GetArrayFromImage(ct)
    cbct_np = sitk.GetArrayFromImage(cbct)
    ct_pixels = ct_np[ct_np > -700].flatten()
    cbct_pixels = cbct_np[cbct_np > -700].flatten()
    
    ct_histogram, bin_edges = np.histogram(ct_pixels, bins=141, range=[-700, 1000])
    cbct_histogram, bin_edges = np.histogram(cbct_pixels, bins=141, range=[-700, 1000])
    bin_centers = list(map(lambda i: 0.5*(bin_edges[i] + bin_edges[i+1]), range(len(bin_edges)-1)))
    
    probabilities = np.interp(cbct_np, bin_centers, np.cumsum(cbct_histogram) / np.sum(cbct_histogram))
    cbct_t_np = np.interp(probabilities, np.cumsum(ct_histogram) / np.sum(ct_histogram), bin_centers) 
    
    cbct_t = sitk.GetImageFromArray(cbct_t_np)
    cbct_t.SetDirection(cbct.GetDirection())
    cbct_t.SetOrigin(cbct.GetOrigin())
    cbct_t.SetSpacing(cbct.GetSpacing())

    cbct_t = sitk.Cast(cbct_t, sitk.sitkFloat32)

    return cbct_t


##########################################################################
#
# Rigid registration to align the cbct with the reference CT
#
def register_cbct_to_ct(ct_ref_float, cbct_float):

    # here we move around the cbct to match the CT
    fixed_image = ct_ref_float
    moving_image = cbct_float

    # start by aligning centers of gravity 
    initial_cog_transform = sitk.CenteredTransformInitializer(
        ct_ref_float,
        cbct_float,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    cbct_cog_resampled = sitk.Resample(
        cbct_float,
        ct_ref_float,
        initial_cog_transform,
        sitk.sitkLinear,
        -1000,
        cbct_float.GetPixelID(),
    )


    params = initial_cog_transform.GetParameters()
    print('rigid transform params', params)
    

    #
    # define the registration method
    #
    rigid_registration = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    rigid_registration.SetMetricAsCorrelation()
    #rigid_registration.SetMetricAsMeanSquares()
    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    #registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    #registration_method.SetMetricSamplingPercentage(0.01)

    rigid_registration.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    rigid_registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    rigid_registration.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    #rigid_registration.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
    #rigid_registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1, 0])
    rigid_registration.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2])
    rigid_registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 2])
    #rigid_registration.SetShrinkFactorsPerLevel(shrinkFactors=[4])
    #rigid_registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2])
    rigid_registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    rigid_registration.SetInitialTransform(initial_cog_transform, inPlace=False)

    # call back functions to monitor registration
    rigid_registration.AddCommand(
        sitk.sitkMultiResolutionIterationEvent, on_update_multiresolution_event
    )
    rigid_registration.AddCommand(
        sitk.sitkIterationEvent, lambda: on_rigid_iteration_event(rigid_registration)
    )

    final_rigid_transform = rigid_registration.Execute( fixed_image, moving_image)

    # finally resample the cbct according to rigid registration
    cbct_rigid_resampled = sitk.Resample(
        cbct_float,
        ct_ref_float,
        final_rigid_transform,
        sitk.sitkLinear,
        -1000,
        cbct_float.GetPixelID(),
    )

    return cbct_cog_resampled, cbct_rigid_resampled

##########################################################################
#
# Use the CT as a template to extend the anatomy of the CBCT
# Copy pixels from the CBCT (previously registered) to CT for all pixels 
# fullfilling that both CT and CBCT pixel value is > -250 HU.
#
# This means that there is an assumption that the patient external do not 
# change during the course of treatment.
#
def create_extended_cbct(ct_ref_float, cbct_in_ct_for):

    #
    # Copy CBCT pixels to CT template where there is a CBCT pixel value > -800
    # convert to numpy -> over write pixels
    #
    cbct_extended_np = sitk.GetArrayFromImage(ct_ref_float)
    cbct_in_ct_for_np = sitk.GetArrayFromImage(cbct_in_ct_for)

    ct_ref_inside = cbct_extended_np > -250
    cbct_inside = cbct_in_ct_for_np > -250
    ct_cbct_inside = ct_ref_inside * cbct_inside
    cbct_extended_np[ct_cbct_inside] = cbct_in_ct_for_np[ct_cbct_inside]

    #
    # Create a simpleitk image from numpy array containing the new pixels
    # Same geometry (voxelisation) as the reference CT
    #
    cbct_extended = sitk.GetImageFromArray(cbct_extended_np)
    cbct_extended.SetDirection(ct_ref_float.GetDirection())
    cbct_extended.SetOrigin(ct_ref_float.GetOrigin())
    cbct_extended.SetSpacing(ct_ref_float.GetSpacing())

    return cbct_extended


##########################################################################
# Create a mask based on the reference CT 
#       Set all pixels with HU > -500 = 1
#       Set all pixels with HU <= -500 = 0
#
# The procedure is performed by 
#   1. converting pixel data to numpy matrix 
#   2. Perform the thresholding according to the above
#   3. Create SimpleITK image from the result -> mask
#
def create_registration_mask(ct_ref_float):

    ct_fixed_mask_np = sitk.GetArrayFromImage(ct_ref_float)
    ct_fixed_mask_np[ct_fixed_mask_np > -500] = 1
    ct_fixed_mask_np[ct_fixed_mask_np <= -500] = 0
    ct_fixed_mask_np = ct_fixed_mask_np.astype(np.uint8)

    # create an SimpeITK image from the data
    fixed_mask = sitk.GetImageFromArray(ct_fixed_mask_np)
    fixed_mask.SetDirection(ct_ref_float.GetDirection())
    fixed_mask.SetOrigin(ct_ref_float.GetOrigin())
    fixed_mask.SetSpacing(ct_ref_float.GetSpacing())

    return fixed_mask

##########################################################################
def create_post_processing_mask(image_template, rtstruct):
    """ Create a mask based on the external ROI and a band around it

    """

    pos_000 = image_template.GetOrigin()
    dim = image_template.GetSize()
    spacing = image_template.GetSpacing()
    grid = [np.array(pos_000), np.array(spacing), np.array(dim)]

    external = Roi.from_rtstruct(rtstruct, 'External', mask_margin=0, grid=grid)
    
    #
    # Create distance map such a narrow band can be constructed
    #
    dt_inside = distance_transform_edt(external.mask, sampling = spacing)
    dt_outside = distance_transform_edt(1 - external.mask, sampling = spacing)
    dt = dt_inside + dt_outside
    mask = np.zeros(dim, dtype=np.int8)
    mask[dt < 20] = 1
    mask = np.swapaxes(mask, 0, 2)
    fixed_mask = sitk.GetImageFromArray(mask)

    #ct_fixed_mask_np = sitk.GetArrayFromImage(ct_ref_float)
    #ct_fixed_mask_np[ct_fixed_mask_np > -500] = 1
    #ct_fixed_mask_np[ct_fixed_mask_np <= -500] = 0
    #ct_fixed_mask_np = ct_fixed_mask_np.astype(np.uint8)

    # create an SimpeITK image from the data
    #fixed_mask = sitk.GetImageFromArray(image_template)
    fixed_mask.SetDirection(image_template.GetDirection())
    fixed_mask.SetOrigin(pos_000)
    fixed_mask.SetSpacing(spacing)

    return fixed_mask


#########################################################################
def erosion(image):

    image_float = sitk.Cast(image, sitk.sitkFloat32)
    image_np = sitk.GetArrayFromImage(image_float)
    image_eroded_np = binary_erosion(image_np, iterations=3)
    
    image_eroded_np = binary_fill_holes(image_eroded_np)
    image_eroded_np = image_eroded_np.astype(np.float32)

    image_eroded = sitk.GetImageFromArray(image_eroded_np)
    image_eroded.SetDirection(image.GetDirection())
    image_eroded.SetOrigin(image.GetOrigin())
    image_eroded.SetSpacing(image.GetSpacing())

    return image_eroded

#########################################################################
# post processing of the synthetic CT 
# - Copy all pixels from reference CT for all pixels outside CBCT FOV
def post_processing(cbct_hu_mapped, ct_ref_float, fixed_mask):
    
    fixed_mask_np = sitk.GetArrayFromImage(fixed_mask)
    cbct_hu_mapped_np = sitk.GetArrayFromImage(cbct_hu_mapped)
    ct_ref_float_np = sitk.GetArrayFromImage(ct_ref_float)
    
    # use reference CT outside of CBCT
    synthetic_ct_np = cbct_hu_mapped_np
    synthetic_ct_np[fixed_mask_np < 1] = ct_ref_float_np[fixed_mask_np < 1] 

    # create an SimpeITK image from the data
    synthetic_ct = sitk.GetImageFromArray(synthetic_ct_np)
    synthetic_ct.SetDirection(ct_ref_float.GetDirection())
    synthetic_ct.SetOrigin(ct_ref_float.GetOrigin())
    synthetic_ct.SetSpacing(ct_ref_float.GetSpacing())

    return synthetic_ct

#########################################################################
# post processing of the synthetic CT 
# - Copy all pixels from reference CT for all pixels outside CBCT FOV
def post_processing2(cbct_hu_mapped, ct_ref_float, fixed_mask):
       
    fixed_mask_np = sitk.GetArrayFromImage(fixed_mask)
    cbct_hu_mapped_np = sitk.GetArrayFromImage(cbct_hu_mapped)
    ct_ref_float_np = sitk.GetArrayFromImage(ct_ref_float)
    
    # use reference CT inside mask
    synthetic_ct_np = cbct_hu_mapped_np
    synthetic_ct_np[fixed_mask_np > 1] = ct_ref_float_np[fixed_mask_np > 1] 

    # create an SimpeITK image from the data
    synthetic_ct = sitk.GetImageFromArray(synthetic_ct_np)
    synthetic_ct.SetDirection(ct_ref_float.GetDirection())
    synthetic_ct.SetOrigin(ct_ref_float.GetOrigin())
    synthetic_ct.SetSpacing(ct_ref_float.GetSpacing())

    return synthetic_ct



#########################################################################
# outout an SimpleITK image as a nifti file for easy IO
def output_image(image, filename):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filename)
    writer.SetImageIO("NiftiImageIO")
    writer.Execute(image)


#########################################################################
#
# Main unction to create a synthetic CT based on CBCT and utilising the 
# reference CT to extend FOV as well as HU values
#
def create_sct(patient_id, fraction):

    global file_def_reg, file_rigid

    output_dir = os.path.join(image_path, '{}_anonymized'.format(patient_id), 'synthetic_CT', fraction)
    print('output dir', output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
     
    # 0. read data
    print('Read data CT and CBCT')
    ct_ref_float = read_ref_ct(patient_id)
    rescale_intercept, rescale_slope = read_pixel_info(patient_id) 
    cbct_float = read_cbct(patient_id, fraction)
    output_image(ct_ref_float, os.path.join(output_dir, 'ct_ref.nii'))
    output_image(cbct_float, os.path.join(output_dir, 'cbct.nii'))
    print('CT ref dimenions', ct_ref_float.GetSize())

    # 1. Perform histogram matching on the CBCT pixels for easier alignment
    cbct_matched = histogram_matching(cbct_float, ct_ref_float)
    output_image(cbct_matched, os.path.join(output_dir, 'cbct_matched.nii'))

    # 2. Align CBCT to CT using rigid registration
    print('Rigid Registration of CBCT to CT')
    file_rigid = open(os.path.join(output_dir, 'rigid_registration.txt'), 'w')
    cbct_cog, cbct_in_ct_for = register_cbct_to_ct(ct_ref_float, cbct_matched)
    output_image(cbct_cog, os.path.join(output_dir, 'cbct_cog.nii'))
    output_image(cbct_in_ct_for, os.path.join(output_dir, 'cbct_in_ct_for.nii'))
    file_rigid.close()

    # 3. Extend the CBCT (now in CT for) using the pixels from the CT
    print('Extend CBCT to CT')
    cbct_extended = create_extended_cbct(ct_ref_float, cbct_in_ct_for)
    output_image(cbct_extended, os.path.join(output_dir, 'cbct_extended.nii')) 

    # 4. Map the pixel values from the reference CT to the extended CBCT using deformable registration 
    # create a mask to only use pixels inside patient for registration
    print('Deformable mappng of HU.')
    file_def_reg = open(os.path.join(output_dir, 'def_reg_registration.txt'), 'w')
    cbct_hu_mapped = deformable_hu_mapping_fast(ct_ref_float, cbct_extended)
    output_image(cbct_hu_mapped, os.path.join(output_dir, 'cbct_hu_mapped.nii'))

    # 5. post processing to make sure the very outermost part of the patient is not deformed
    cbct_in_ct_mask = create_registration_mask(cbct_in_ct_for)
    cbct_in_ct_mask_eroded = erosion(cbct_in_ct_mask)
    output_image(cbct_in_ct_mask_eroded, os.path.join(output_dir, 'cbct_in_ct_mask_eroded.nii'))
    synthetic_ct_float = post_processing(cbct_hu_mapped, ct_ref_float, cbct_in_ct_mask_eroded)
    synthetic_ct = sitk.Cast(synthetic_ct_float, sitk.sitkInt16)
    output_image(synthetic_ct, os.path.join(output_dir, 'synthetic_ct.nii'))


    # output the final synthetic CT
    print('Output of synthetic CT')
    date_time = datetime.now()
    study_date = date_time.strftime("%Y%m%d")
    study_time = date_time.strftime("%H%M%S")
    study_id = fraction

    write_image_series( synthetic_ct, output_dir, 'CT', 
                        'Recti{}'.format(patient_id), patient_id, study_date, study_time, study_id, patient_position = 'HFS',
                        rescale_intercept = rescale_intercept, rescale_slope = rescale_slope)



#########################################################################
#
# Main unction to create a synthetic CT based on CBCT and utilising the 
# reference CT to extend FOV as well as HU values
#
def create_sct_from_rigid(patient_id, fraction):

    global file_def_reg, file_rigid

    output_dir = os.path.join(image_path, '{}_anonymized'.format(patient_id), 'synthetic_CT', fraction)
    print('output dir', output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
     
    # 1. read image data
    print('Start reading images')
    ct_ref_float = sitk.ReadImage(os.path.join(output_dir, 'ct_ref.nii'))
    rescale_intercept, rescale_slope = read_pixel_info(patient_id) 
    print('CT ref size', ct_ref_float.GetSize())
    print('dimension', ct_ref_float.GetDimension())

    #2. Read in the manually registered CBCT image  
    cbct_in_ct_for = sitk.ReadImage(os.path.join(output_dir, 'cbct_in_ct_for_manual.nii'))
    #cbct_in_ct_for = sitk.ReadImage(os.path.join(output_dir, 'cbct_in_ct_for.nii'))

    # 2.5 crop images vs external to improve the speed
    ref_rtstruct = read_ref_rs(patient_id)
    bounding_box = ref_rtstruct.roi_bounding_box('External')
    ct_ref_float = crop_to_bounding_box(ct_ref_float, bounding_box)
    cbct_in_ct_for = crop_to_bounding_box(cbct_in_ct_for, bounding_box)

    # 3. Extend the CBCT (now in CT for) using the pixels from the CT
    print('Extend CBCT to CT')
    cbct_extended = create_extended_cbct(ct_ref_float, cbct_in_ct_for)
    output_image(cbct_extended, os.path.join(output_dir, 'cbct_extended.nii')) 


    # 4. Map the pixel values from the reference CT to the extended CBCT using deformable registration 
    # create a mask to only use pixels inside patient for registration
    # extract contour points in ref structure set to kepp bones intact.
    print('Deformable mappng of HU.')
    #file_def_reg = open(os.path.join(output_dir, 'def_reg_registration.txt'), 'w')
    rigid_points = extract_contour_points(ref_rtstruct, roi_names = ['Sacrum', 'Fermoral Head_R', 'Fermoral Head_L'], num_points_per_roi = [30, 30, 60])
    rigid_points_filename = os.path.join(output_dir, 'rigid_points.txt')
    output_rigid_points(rigid_points, rigid_points_filename)
    cbct_hu_mapped = elx_deformable_registration(ct_ref_float, cbct_extended, rigid_points_filename)
    output_image(cbct_hu_mapped, os.path.join(output_dir, 'cbct_hu_mapped.nii'))

    # 5. post processing to make sure the very outermost part of the patient is not deformed
    post_processing_mask = create_post_processing_mask(ct_ref_float, ref_rtstruct)
    output_image(post_processing_mask, os.path.join(output_dir, 'post_processing_mask.nii'))
    synthetic_ct_float = post_processing2(cbct_hu_mapped, ct_ref_float, post_processing_mask)

    #cbct_in_ct_mask = create_registration_mask(cbct_in_ct_for)
    #cbct_in_ct_mask_eroded = erosion(cbct_in_ct_mask)
    #output_image(cbct_in_ct_mask_eroded, os.path.join(output_dir, 'cbct_in_ct_mask_eroded.nii'))
    #synthetic_ct_float = post_processing(cbct_hu_mapped, ct_ref_float, cbct_in_ct_mask_eroded)
    
    synthetic_ct = sitk.Cast(synthetic_ct_float, sitk.sitkInt16)
    output_image(synthetic_ct, os.path.join(output_dir, 'synthetic_ct.nii'))


    # output the final synthetic CT
    print('Output of synthetic CT')
    date_time = datetime.now()
    study_date = date_time.strftime("%Y%m%d")
    study_time = date_time.strftime("%H%M%S")
    study_id = fraction

    write_image_series( synthetic_ct, output_dir, 'CT', 
                        'Recti{}'.format(patient_id), patient_id, study_date, study_time, study_id, patient_position = 'HFS',
                        rescale_intercept = rescale_intercept, rescale_slope = rescale_slope)

#########################################################################
#
# Align CBCT ro refereference CT
#
def align_cbct(patient_id, fraction):

    global file_rigid

    output_dir = os.path.join(image_path, '{}_anonymized'.format(patient_id), 'synthetic_CT', fraction)
    print('output dir', output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
     
    # 0. read data
    print('Read data CT and CBCT')
    ct_ref_float = read_ref_ct(patient_id)
    cbct_float = read_cbct(patient_id, fraction)
    output_image(ct_ref_float, os.path.join(output_dir, 'ct_ref.nii'))
    output_image(cbct_float, os.path.join(output_dir, 'cbct.nii'))
    print('CT ref dimenions', ct_ref_float.GetSize())

    # 0.5 Perform histogram matching on the CBCT pixels for easier alignment
    cbct_matched = histogram_matching(cbct_float, ct_ref_float)
    output_image(cbct_matched, os.path.join(output_dir, 'cbct_matched.nii'))


    # 1. Align CBCT to CT using rigid registration
    print('Rigid Registration of CBCT to CT')
    file_rigid = open(os.path.join(output_dir, 'rigid_registration.txt'), 'w')
    cbct_cog, cbct_in_ct_for = register_cbct_to_ct(ct_ref_float, cbct_matched)
    output_image(cbct_cog, os.path.join(output_dir, 'cbct_cog.nii'))
    output_image(cbct_in_ct_for, os.path.join(output_dir, 'cbct_in_ct_for.nii'))
    file_rigid.close()

    return
