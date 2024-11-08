# import_data/processing.py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas 
import ipywidgets
import mne
import os
import boto3
import sagemaker
from sagemaker import get_execution_role
from botocore.exceptions import NoCredentialsError
from mne import Epochs, compute_covariance, find_events, make_ad_hoc_cov
from mne.datasets import sample
from mne.preprocessing import annotate_movement, compute_average_dev_head_t, annotate_muscle_zscore, find_eog_events
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf, read_raw_edf
from mne.viz import plot_alignment, set_3d_view
from mne.simulation import (
    add_ecg,
    add_eog,
    add_noise,
    simulate_raw,
    simulate_sparse_stc,
)


# EDFInputPath = 'EDFFiles'
# EDFOutputPath = 'OutputFiles'

def preprocess_eeg(file_path):
    # Placeholder function to simulate preprocessing
    print(f"Preprocessing the EEG file at: {file_path}")
    # processed_data = "processed_data_placeholder"  # Simulated processed data


    def AllEDFProcess(EDFFolder):
        # if not os.path.exists(EDFOutputPath):
        #     os.makedirs(EDFOutputPath)
    
        # for FileName in os.listdir(EDFFolder):
        #   if FileName.endswith('.edf'):
        #      EDFFilePath = os.path.join(EDFFolder, FileName)
        processed_data, PSD_data, EEG_image = EDFProcess(EDFFolder)
        return processed_data, PSD_data


    def EDFProcess(EDFFilePath):
        RawEEGDataFile = mne.io.read_raw_edf(EDFFilePath, preload=True)
        RawEEGDataFile.interpolate_bads();

        BPEEGDataFile = BPFilter(RawEEGDataFile)

        # OutputFileName = f"filtered_{os.path.splitext(os.path.basename(EDFFilePath))[0]}.fif"
        # OutputFile = os.path.join(EDFOutputPath, OutputFileName)
        # BPEEGDataFile.save(OutputFile, overwrite=True)

        EEG_image = RawEEGDataFile
        #EEG_image = 'peepeepoopoo'
        

        ADRatioDF = AlphaDeltaProcess(BPEEGDataFile)
    
        # PSDOutputFileName = f"PSD_{os.path.splitext(os.path.basename(EDFFilePath))[0]}.csv"
        # PSDOutputFile = os.path.join(EDFOutputPath, PSDOutputFileName)
        # ADRatioDF.to_csv(PSDOutputFile, index=False)

        #print(f"Finished and saved file {EDFFilePath} to {OutputFile}")
        #print(f"Finished and saved PSD data to {PSDOutputFile}")
        return BPEEGDataFile, ADRatioDF, EEG_image

    def BPFilter(RawEEGDataFile):
        BPEEGDataFile = RawEEGDataFile.copy().filter(l_freq=0.5, h_freq=40.0, fir_design='firwin')
        return BPEEGDataFile


    ## ALPHA DELTA PSD ANALYSIS AND DATA FRAMING ##
    def AlphaDeltaProcess(EEGFile):
        AlphaComp = EEGFile.compute_psd(method='welch', fmin=8, fmax=12, tmin=None, tmax=None, picks='eeg', exclude=(), proj=False, remove_dc=True, reject_by_annotation=True, n_jobs=1, verbose=None)
        AlphaPSD, AlphaFreq = AlphaComp.get_data(return_freqs=True)
        #display(AlphaComp)
        DeltaComp = EEGFile.compute_psd(method='welch', fmin=0.5, fmax=4, tmin=None, tmax=None, picks='eeg', exclude=(), proj=False, remove_dc=True, reject_by_annotation=True, n_jobs=1, verbose=None)
        DeltaPSD, DeltaFreq = DeltaComp.get_data(return_freqs=True)
        #DeltaComp.plot()
        #raw_csd = mne.preprocessing.compute_current_source_density(RawEEGDataFile);

        ChanLab = EEGFile.ch_names

        AlphaMean = AlphaPSD.mean(axis=1)
        DeltaMean = DeltaPSD.mean(axis=1)

        AlDeRat = AlphaMean / DeltaMean

        PSDRatDF = pandas.DataFrame({'Channel': ChanLab,'Alpha Power': AlphaMean,'Delta Power': DeltaMean,'Alpha/Delta Ratio': AlDeRat})

        #display(PSDRatDF)
    
        return PSDRatDF



    processed_data, PSD_data = AllEDFProcess(file_path)

    return processed_data, PSD_data

def run_ml_model(processed_data, PSD_data):
    # Placeholder function to simulate running the ML model
    print(f"Running ML model on data: {processed_data}")
    #display(processed_data)
    #display(PSD_data)
    result = "Stroke: 98%"  # Simulated result
    return result


