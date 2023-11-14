
import os
import glob
from os.path import join as opj
import h5py  
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
import json

TRIAL_PER_SESS = 750
SESS_NUM = 37
MAX_IDX = TRIAL_PER_SESS * SESS_NUM
SUBJ=sub.split("subj")[-1]

class NSDDataset(Dataset):
    
      """
        A PyTorch dataset used to load images and related betas (neural data) from the NSD dataset.
        
        Parameters:
            stim_order_path (str): The path to the .mat file containing information about the stimulus order.
            stim_file_path (str): The path to the .hdf5 file containing the image data.
            stim_info_path (str): The path to the .csv file containing information about the images.
            stim_captions_train_path (str): The path to the .json file containing captions for the training images.
            stim_captions_val_path (str): The path to the .json file containing captions for the validation images.
            fmri_path (str): The path to the directory containing the fMRI data.
            roi (bool): If True, load data from the ROI data (.pt files), else load from the full brain data (.hdf5 files).
        """
    
    def __init__(self,stim_order_path=stim_order_path,
                 stim_file_path=stim_file_path,
                 stim_info_path=stim_info_path,
                 stim_captions_train_path=stim_captions_train_path,
                 stim_captions_val_path=stim_captions_val_path,
                 fmri_path=subj_betas_path,
                roi=True):

        
        self.fmri_dir = fmri_path
        
        self.roi=roi
        if roi:
            self.fmri_files = [f for f in os.listdir(self.fmri_dir) if
                           os.path.isfile(os.path.join(self.fmri_dir, f)) and
                           f[-3:] == '.pt']
        
        else:
            self.fmri_files = [f for f in os.listdir(self.fmri_dir) if
                           os.path.isfile(os.path.join(self.fmri_dir, f)) and
                           f[-5:] == '.hdf5']
        
        stim_order = loadmat(stim_order_path)

        #for indexing (remember to apply -1 because of MATLAB)
        self.subjectim = stim_order['subjectim']
        self.masterordering = stim_order['masterordering'].squeeze()


        #for images
        
        self.stim_file = stim_file_path
        
        #for captions
        self.stim_info=pd.read_csv(stim_info_path)
        
        with open(stim_captions_train_path,'rb') as f:
            train_cap=json.load(f)

        with open(stim_captions_val_path,'rb') as f:
            val_cap=json.load(f)

        
        self.caption_train_df=pd.DataFrame.from_dict(train_cap["annotations"])
        self.caption_val_df=pd.DataFrame.from_dict(val_cap["annotations"])
            
    def __len__(self):
        return TRIAL_PER_SESS * len(self.fmri_files)
    
    def __getitem__(self,idx):

        #retrieve nsdId accounting for Matlab indexig for subject, masterordering and id
        nsdId = self.subjectim[int(SUBJ)-1, self.masterordering[idx] - 1] - 1 
        
        #load relative image
        with h5py.File(self.stim_file, 'r') as f:
            img_sample = f['imgBrick'][nsdId]
        
        
        #load caption (only first?)
        #get codoId
        cocoId=self.stim_info[self.stim_info.nsdId==nsdId].cocoId.values[0]
        split=self.stim_info[self.stim_info.nsdId==nsdId].cocoSplit.values[0]
        
        if split=="train2017":
            captions=self.caption_train_df[self.caption_train_df.image_id==cocoId].caption.values
        
        elif split=="val2017":
            captions=self.caption_val_df[self.caption_val_df.image_id==cocoId].caption.values
        
        
        # retrieve the related betas
        
        sess = idx // TRIAL_PER_SESS + 1
        
        if self.roi:
            base_name= self.fmri_files[0][:-5] #something like betas_sessions
            fmri_file = os.path.join(self.fmri_dir, f'{base_name}{sess:02}.pt')
            fmri_sample=torch.load(fmri_file)[:,idx % TRIAL_PER_SESS]  
        else:
            base_name= self.fmri_files[0][:-7] #something like betas_sessions
            fmri_file = os.path.join(self.fmri_dir, f'{base_name}{sess:02}.hdf5')
            with h5py.File(fmri_file, 'r') as f:
                fmri_sample = f['betas'][idx % TRIAL_PER_SESS]       

        #
        
        return fmri_sample,img_sample, captions