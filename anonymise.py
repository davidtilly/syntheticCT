
import os, glob
import SimpleITK as sitk
import pydicom
from datetime import datetime

def get_for_uid(rtplan_filename_input):
    ds = pydicom.read_file(rtplan_filename_input, force=True)
    return ds.FrameOfReferenceUID

def psuedo_anonymise_ds(ds, patient_name, patient_id, study_date, study_time, study_id):
    """ Psuedo anonymise an RT file """
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = ''
    ds.PatientSex = ''
    ds.OtherPatientIDs = patient_id
    ds.OtherPatientNames = patient_name
    ds.StudyID = study_id
    ds.StudyDate = study_date
    ds.StudyTime = study_time

def set_references_rtdose(ds, referenced_rtplan_uid, referenced_rtss_uid):
    ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = referenced_rtplan_uid
    ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID = referenced_rtss_uid


def psuedo_anonymise_rtdose(filename_input, filename_output, patient_name, patient_id,
                            study_date, study_time, study_id, rtdose_uid, rtplan_uid, rtss_uid):
    """ Psuedo anonymise an RT Dose file """
    
    ds = pydicom.read_file(filename_input, force=True)
    ds.SOPInstanceUID = rtdose_uid
    psuedo_anonymise_ds(ds, patient_name, patient_id, study_date, study_time, study_id)
    set_references_rtdose(ds, rtplan_uid, rtss_uid)

    pydicom.write_file(filename_output, ds)


def set_references_rtplan(ds, referenced_rtss_uid):
    ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID = referenced_rtss_uid

def psuedo_anonymise_rtplan(filename_input, filename_output, patient_name, patient_id, 
                            study_date, study_time, study_id, rtplan_uid, rtss_uid):
    """ Psuedo anonymise an RT file """
    ds = pydicom.read_file(filename_input, force=True)
    ds.SOPInstanceUID = rtplan_uid
    psuedo_anonymise_ds(ds, patient_name, patient_id, study_date, study_time, study_id)
    set_references_rtplan(ds, rtss_uid)
    pydicom.write_file(filename_output, ds)

def psuedo_anonymise_rtss(filename_input, filename_output, patient_name, patient_id, 
                          study_date, study_time, study_id, rtss_uid):
    """ Psuedo anonymise an RT file """
    ds = pydicom.read_file(filename_input, force=True)
    ds.SOPInstanceUID = rtss_uid
    psuedo_anonymise_ds(ds, patient_name, patient_id, study_date, study_time, study_id)
    pydicom.write_file(filename_output, ds)


def psuedo_anonymise_image_series(input_dir, output_dir, patient_name, patient_id, 
                                  study_date, study_time, study_id):
    """ 
    Read an image series, apply the required tags (as psuedo anonymisation) and write to the output directory
    """
    image_filenames_input = glob.glob(os.path.join(input_dir, '*image*.dcm'))
    #image_filenames_input = glob.glob(os.path.join(input_dir, 'CT*'))
    for filename in image_filenames_input:
        ds = pydicom.read_file(filename, force=True)
        ds.ReferringPhysiciansName = ''
        psuedo_anonymise_ds(ds, patient_name, patient_id, study_date, study_time, study_id)
    
        output_filename = os.path.join(output_dir, 'CT{}.dcm'.format(ds.SOPInstanceUID))
        pydicom.write_file(output_filename, ds)

def psuedo_anonymise_case(input_dir, output_dir, patient_name, patient_id, study_id):
    """ 
    Read a whole dicom set from the input directory (rtplan, rtss, rtdose and images), perform psuedo-anonymisation, 
    and write the result to the output directory
    """
    rtplan_uid =  pydicom.uid.generate_uid()
    rtss_uid =  pydicom.uid.generate_uid()
    rtdose_uid =  pydicom.uid.generate_uid()
    date_time = datetime.now()
    study_date = date_time.strftime("%Y%m%d")
    study_time = date_time.strftime("%H%M%S")

    rtplan_filename_input = glob.glob(os.path.join(input_dir, '*vmat.dcm'))[0]
    print('Rtplan filename', rtplan_filename_input)
    rtplan_filename_output = os.path.join(output_dir, 'RP_{}.dcm'.format(patient_id))

    psuedo_anonymise_rtplan(rtplan_filename_input, rtplan_filename_output, patient_name, patient_id, 
                            study_date, study_time, study_id, rtplan_uid, rtss_uid)

    rtss_filename_input = glob.glob(os.path.join(input_dir, '*StrctrSets.dcm'))[0]
    print('RtStruct filename', rtss_filename_input)    
    rtss_filename_output = os.path.join(output_dir, 'RS_{}.dcm'.format(patient_id))
    psuedo_anonymise_rtss(rtss_filename_input, rtss_filename_output, patient_name, patient_id,
                          study_date, study_time, study_id, rtss_uid)

    rtdose_filename_input = glob.glob(os.path.join(input_dir, '*Dose*.dcm'))[0]
    print('RTDose filename', rtdose_filename_input)
    rtdose_filename_output = os.path.join(output_dir, 'RD_{}.dcm'.format(patient_id))
    psuedo_anonymise_rtdose(rtdose_filename_input, rtdose_filename_output, patient_name, patient_id, 
                            study_date, study_time, study_id, rtdose_uid, rtplan_uid, rtss_uid)

    # anonymise the images
    print(os.path.join(input_dir, 'CT*.dcm'))
    psuedo_anonymise_image_series(input_dir, output_dir, patient_name, patient_id, 
                                  study_date=study_date, study_time=study_time, study_id=study_id)


def read_image_series(path, pixel_type = sitk.sitkFloat32):
    """ 
    Read an image series that exist in the specified path 
    Convert to float pixels is default
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    
    if len(dicom_names) is 0:
        raise Exception('Directory does not contain images {}'.format(path))
    
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    print('read pixel type', image.GetPixelIDTypeAsString())

    if image.GetPixelID() != pixel_type:
        print('casting from', image.GetPixelIDTypeAsString())
        image = sitk.Cast(image, pixel_type)

    return image

def read_image_series_pixel_rescaling(path):

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    
    if len(dicom_names) is 0:
        raise Exception('Directory does not contain images {}'.format(path))     
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_names[0])
    reader.Execute()

    rescale_intercept = reader.GetMetaData('0028|1052')
    rescale_slope = reader.GetMetaData('0028|1053')

    return rescale_intercept, rescale_slope

def write_image_series( image, output_path, modality, 
                        patient_name, patient_id, study_date, study_time, study_id, patient_position = 'HFS',
                        rescale_intercept = -1024, rescale_slope = 1,  
                        study_instance_uid = None, series_instance_uid = None, frame_of_reference_uid = None):
    """ 
    Write the resulting image as a dicom image series (and apply the supplied info)
    """
    
    if image.GetPixelID() is not sitk.sitkInt16:
        raise Exception('Image must be signed integer 32')
        
    print('3D pixel type', image.GetPixelIDTypeAsString())
    writer = sitk.ImageFileWriter()

    manufacturer = 'UAS'

    if study_instance_uid is None:
        study_instance_uid = pydicom.uid.generate_uid()

    if series_instance_uid is None:
        series_instance_uid = pydicom.uid.generate_uid()
    
    if frame_of_reference_uid is None:
        frame_of_reference_uid = pydicom.uid.generate_uid()

    tags_values_to_write = [["0008|0008", 'DERIVED\\SECONDARY\\AXIAL'],
                            ["0010|0010",  patient_name],   # Patient Name
                            [ "0010|0020", patient_id],     # Patient ID
                            ["0010|0030",  ''],             # Patient Birth Date
                            ["0020|0011", '1'],             # Series Number
                            ["0020|0010",  study_id],       # Study ID, for human consumption
                            ["0008|0022",  study_date],     # Acquisition Date
                            ["0008|0020",  study_date],     # Study Date
                            ["0008|0030",  study_time],     # Study Time
                            ["0008|0021",  study_date],     # Series Date
                            ["0008|0031",  study_time],     # Series Time
                            ["0008|0023",  study_date],     # Content date
                            ["0008|0033",  study_time],     # Content time
                            ["0020|000d",  study_instance_uid],
                            ["0020|000e",  series_instance_uid],
                            ["0020|0052",  frame_of_reference_uid],
                            ["0018|5100",  patient_position],
                            ["0008|0070",  manufacturer],
                            ["0008|0060", modality],            # Modality
                            ["0028|1052", rescale_intercept],
                            ["0028|1053", rescale_slope],
                            ["0028|0103", '1']]


    # write all image slices to disc while writinh the tags in the list 
    for i in range(image.GetDepth()):

        image_slice = image[:, :, i]

        # Tags shared by the series.
        for tag, value in tags_values_to_write:
            image_slice.SetMetaData(tag, value)

        sop_instance_uid = pydicom.uid.generate_uid()
        image_slice.SetMetaData("0008|0018", sop_instance_uid)

        image_slice.SetMetaData("0020|0032", '\\'.join(
            map(str, image.TransformIndexToPhysicalPoint((0, 0, i)))))

        # Instace Number
        image_slice.SetMetaData("0020|0013", str(i))

        # Write to the output directory and add the extension dcm, to force writing
        # in DICOM format.

        if i == 0:
            print('2d image pixel type', image_slice.GetPixelIDTypeAsString())

        output_filename = os.path.join(output_path, '{}{}.dcm'.format(modality, sop_instance_uid))
        writer.KeepOriginalImageUIDOn()
        writer.SetFileName(output_filename)
        writer.Execute(image_slice)

