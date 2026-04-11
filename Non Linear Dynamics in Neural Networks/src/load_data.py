import nilearn
from nilearn import image
from nilearn.image import load_img
from nilearn.image import clean_img
from nilearn import datasets
import xml.etree.ElementTree as ET

def load_fmri(fmri_path):
    #loading resting state fMRI data (NIfTI file)
    fmri_img = load_img(fmri_path) 
    return fmri_img

def preprocess_fmri(fmri_img):
    #minimal preprocessing
    fmri_cleaned = clean_img(
        fmri_img, 
        detrend = True, 
        standardize = True, 
        low_pass = 0.1, 
        high_pass = 0.01, 
        t_r = 2.0)

    return fmri_cleaned

def load_atlas():
    #loading AAL atlas 
    atlas_path = "dataset/atlas/AAL.nii"
    atlas_img = load_img(atlas_path)

    #loading atlas labels from XML file
    tree = ET.parse("dataset/atlas/AAL.xml")
    root = tree.getroot()

    atlas_labels = [region.find("name").text for region in root.findall(".//label")]                       
    
    return atlas_img, atlas_labels