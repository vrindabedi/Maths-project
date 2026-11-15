# Maths-project

Standard CT scans use voxels to model the brain and make 2D cross sections which blurs the edges and makes tiny tumors hard to see. 
This project uses calculus and continuity to convert a continuous function that represents the organ being scanned into a piece wise function according to thresholds and attenuation value jumps. Then it converts the raw data from the CT scan into a more clearer model.

## Supported files
- DICOM (.dcm files or folders)
- NIfTI (.nii, .nii.gz, .seg.nii)
- Excel/CSV (raw CT scanner ray tables)

## To use install
- numpy
- pandas
- scipy
- nibabel
- pydicom
- openpyxl
