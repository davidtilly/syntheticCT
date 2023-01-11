import SimpleITK as sitk
import time
import numpy as np
from TreatmentSimulation.DicomIO.RtStruct import RtStruct

########################################################################################
def extract_contour_points(rtstruct : RtStruct, roi_names : list[str], num_points_per_roi : list[int]) -> np.array:
    """ 
    Extract controur points to include as regularisation in the deformable image registration 
    :param rtstruct: Dicom rtstruct containing all contour points
    :param roi_names: list of roi names
    :param num_points_per_roi: number of points to extract per roi
    
    """

    if len(roi_names) != len(num_points_per_roi):
        raise Exception('Iumber of rois not consistent btwn names and num samples.')

    all_points = []
    for name, n in zip(roi_names, num_points_per_roi):
        points = np.array(rtstruct.point_cloud_n(name, n))
        all_points.append(points)
    
    # make a single list of points
    all_points = np.concatenate(all_points)

    return all_points

########################################################################################
def output_rigid_points(points : np.array, filename : str):
    """ Write points to a file that Elastix can read. """

    f = open(filename, 'w')
    f.write('points')
    f.write('\n')
    f.write('{}'.format(len(points)))
    f.write('\n')
    for p in points:
        f.write('{}  {}  {}'.format(p[0], p[1], p[2]))
        f.write('\n')
        
    f.close()

########################################################################################
def deformable_registration(ct_ref_float : sitk.Image, cbct_extended: sitk.Image, points_filename) -> sitk.Image:
    """
    Deformable image registration to align the reference CT (moving) extended CBCT (fixed)
    No need for a rigid pre-reg since that has already been done.
    The points in the 'points filename' is used as regulariser to not destroy the rigid registration
    """

    print('ACCURATE Elastix Deformable registration used!')

    start_time = time.time()

    fixed_image = cbct_extended
    moving_image = ct_ref_float

    selx = sitk.ElastixImageFilter() 
    selx.LogToFileOn()
    selx.SetFixedImage(fixed_image)
    selx.SetMovingImage(moving_image)

    parameter_map = sitk.GetDefaultParameterMap('bspline')
    parameter_map['MaximumNumberOfIterations'] = ('250',)
    parameter_map["Metric"] = ('AdvancedMattesMutualInformation', 'TransformBendingEnergyPenalty', 'CorrespondingPointsEuclideanDistanceMetric')
    parameter_map['Metric2Weight'] = ('1.0',)

    parameter_map['FinalGridSpacingInPhysicalUnits'] = ('10.000000', )
    selx.SetParameterMap(parameter_map)
    selx.SetFixedPointSetFileName(points_filename)
    selx.SetMovingPointSetFileName(points_filename)    
    
    selx.Execute()

    result_image = selx.GetResultImage()
    #transform_parameter_mMap = s elx.GetTransformParameterMap()

    print(time.time() - start_time)

    return result_image