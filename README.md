# DCM Normalization and PNG Conversion

Overview

This repository contains the scripts and utilities for processing DICOM (DCM) files from lung CT scans. The project focuses on normalizing the data and converting the DICOM files into PNG format for easier visualization and further analysis.

Features
Data Preprocessing:
Normalize DICOM files.
Extract and save image patches.
Utilities:
Convert DICOM files to PNG format.
Segmentation utilities for lung CT scans.
Environment Setup:
Easily reproducible with requirements.txt and environment.yml.

1. Clone the Repository
bash
Copy code
git clone https://github.com/piyushsamant1/DCM_normalization_and_png_conversion.git
cd DCM_normalization_and_png_conversion
2. Set Up the Environment
Using conda:
bash
Copy code
conda env create -f environment.yml
conda activate pipeline_env
Using pip:
bash
Copy code
pip install -r requirements.txt
3. Run the Scripts
Run the Jupyter notebook:
bash
Copy code
jupyter notebook Data_preprocessing.ipynb
Use the utility scripts for DICOM to PNG conversion:
bash
Copy code
python dicom_utils.py
Requirements
Python 3.10 or later
Libraries specified in requirements.txt
Optional: Use the conda environment described in environment.yml
