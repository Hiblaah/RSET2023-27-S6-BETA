import nilearn
from nilearn.maskers import NiftiLabelsMasker

def extract_bold_timeseries(fmri_clean, atlas_maps):
    masker = NiftiLabelsMasker(
        labels_img = atlas_maps,
        standardize = True
    )

    time_series = masker.fit_transform(fmri_clean)
    return time_series

    #output: (time points, no. of ROIs)