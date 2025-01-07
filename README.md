# DCM Normalization and PNG Conversion

## Overview

This repository contains the scripts and utilities for processing DICOM (DCM) files from lung CT scans. 
The repository focuses on normalizing the data and converting the DICOM files into PNG format for easier visualization and further analysis.

## Features

### Data Preprocessing:
Normalize DICOM files.
Extract and save image patches.

### Utilities:
Convert DICOM files to PNG format.
Segmentation utilities for lung CT scans.

## Environment Setup:
Reproducible with requirements.txt and environment.yml.

### Set Up the Environment
Using conda:
conda env create -f environment.yml
conda activate pipeline_env

### Using pip

pip install -r requirements.txt

## Run the Scripts (Jupyter notebook):
Data_preprocessing.ipynb

### Requirements
Python 3.10 or later
Libraries specified in requirements.txt
Optional: Use the conda environment described in environment.yml
