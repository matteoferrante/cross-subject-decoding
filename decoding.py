import pickle
import sys
sys.path.append("../")
from hparams import HParams
from hps import Hyperparams
from vae import VAE

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.manifold import TSNE
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
import os
import glob
from os.path import join as opj
import h5py  
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import json
from PIL import Image
# from diffusers import VersatileDiffusionPipeline
# from diffusers import VersatileDiffusionDualGuidedPipeline
from diffusers.models import AutoencoderKL, DualTransformer2DModel, Transformer2DModel, UNet2DConditionModel
from versatile_diffusion_dual_guided import VersatileDiffusionDualGuidedPipeline
from versatile_diffusion_dual_guided_fake_images import VersatileDiffusionDualGuidedFromCLIPEmbeddingPipeline
from autoencoder import *
from torchsummary import summary
import torchvision
import tqdm
from sklearn.linear_model import Ridge
import pickle
import wandb
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

to_pil=torchvision.transforms.ToPILImage()


class NSDDataset(Dataset):
    

    
    def __init__(self, fmri_data,imgs_data,caption_data,transforms=None):
        self.fmri_data=np.load(fmri_data)
        self.imgs_data=np.load(imgs_data).astype(np.uint8)
        self.caption_data=np.load(caption_data,allow_pickle=True)
        self.transforms=transforms
        
    def __len__(self):
        return  len(self.fmri_data)
    
    def __getitem__(self,idx):
        fmri=torch.tensor(self.fmri_data[idx])
        img=Image.fromarray(self.imgs_data[idx])
        
        if self.transforms:
            img=self.transforms(img)
        
        caption=self.caption_data[idx][0] #cambiare se ne voglio altre
        
        return fmri,img,caption



class BrainDiffuserPretrainedDecoder:
    def __init__(self,vae_weights="/home/matteo/models/vdvae/vae2.pt",
                 vae_hyper='/home/matteo/models/vdvae/H.sav', 
                 pretrained=True,
                 subj_path=None,
                 device="cpu"):
        super().__init__()
        self.keep=31
        self.device=device
        self.pretrained=pretrained
        self.subj_path=subj_path

        print("Loading pretrained deep learning backbones")

        with open(vae_hyper, 'rb') as fp:
            d = pickle.load(fp)

        H=Hyperparams()
        for k,v in d.items():
            H[k]=v
            
        vae=VAE(H)    
        state_dict = torch.load(vae_weights)
        new_state_dict = {}
        l = len('module.')
        for k in state_dict:
            if k.startswith('module.'):
                new_state_dict[k[l:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
        state_dict = new_state_dict
        vae.load_state_dict(state_dict)


        self.vae=vae.to(device)


        self.pipe_embed= VersatileDiffusionDualGuidedFromCLIPEmbeddingPipeline.from_pretrained("shi-labs/versatile-diffusion",)

        self.pipe_embed.remove_unused_weights()
        self.pipe_embed.to(self.device)

        if self.pretrained:

            ### aggiungere qua 
            self.train_fmri_mean=torch.load(opj(self.subj_path,"train_fmri_mean.pt"))
            self.train_fmri_std=torch.load(opj(self.subj_path,"train_fmri_std.pt"))

            assert self.subj_path is not None, "Please provide a valid subject path, whith decoding dir and related files"
            print("Loading pretrained brain to feature models")

            keys=np.arange(self.keep)
            # filename='brain_to_latent_ridge.sav'
            self.brain_to_latent = {}
        #     pickle.load(open(opj(f"models/{sub}/decoding",filename), 'rb'))


            self.brain_to_img_emb=[]
            self.brain_to_txt_emb=[]          
            
            print("loading brain to latent models")
            for k in keys:
                filename = f'brain_to_vdvae_latent_ridge_{k}.sav'
                p=pickle.load(open(opj(self.subj_path,"decoding",filename), 'rb'))
                self.brain_to_latent[k]=p

            print("loading brain to img embeddings models")
            for i in range(257):
                filename = f'brain_to_img_emb_ridge_{i}.sav'
                p=pickle.load(open(opj(self.subj_path,"decoding",filename), 'rb'))
                self.brain_to_img_emb.append(p)
            
            print("loading brain to txt embeddings models")
            for i in range(77):
                filename = f'brain_to_txt_emb_ridge_{i}.sav'
                p=pickle.load(open(opj(self.subj_path,"decoding",filename), 'rb'))
                self.brain_to_txt_emb.append(p)

            print("loading adjust values")
            filename = f'latent_adjust_values.sav'
            with open(opj(self.subj_path,filename), 'rb') as f:
                self.latent_adjust_values=pickle.load(f)

            self.clip_img_embeds_mean=torch.load(opj(self.subj_path,"clip_img_embeds_mean.pt"))
            self.clip_img_embeds_std=torch.load(opj(self.subj_path,"clip_img_embeds_std.pt"))


            self.clip_txt_embeds_mean=torch.load(opj(self.subj_path,"clip_txt_embeds_mean.pt"))
            self.clip_txt_embeds_std=torch.load(opj(self.subj_path,"clip_txt_embeds_std.pt"))
            
            print("loading predicted values for adjusting")
            
            img_emb_mean_path = opj(self.subj_path,"predicted_img_emb_mean.pt")
            img_emb_std_path = opj(self.subj_path,"predicted_img_emb_std.pt")
            txt_emb_mean_path = opj(self.subj_path,"predicted_txt_emb_mean.pt")
            txt_emb_std_path = opj(self.subj_path,"predicted_txt_emb_std.pt")

            # Load the tensors
            self.predicted_img_emb_mean = torch.load(img_emb_mean_path)
            self.predicted_img_emb_std = torch.load(img_emb_std_path)
            self.predicted_txt_emb_mean = torch.load(txt_emb_mean_path)
            self.predicted_txt_emb_std = torch.load(txt_emb_std_path)
            
            
            with open(opj(self.subj_path,"predicted_latent_stats.sav"),"rb") as f:
                self.predicted_latent_stats=pickle.load(f)
            

    def get_latents(self,data):
        
        
        shapes={0:(16,1,1),
                1: (16, 1, 1),
                 2: (16, 4, 4),
                 3: (16, 4, 4),
                 4: (16, 4, 4),
                 5: (16, 4, 4),
                 6: (16, 8, 8),
                 7: (16, 8, 8),
                 8: (16, 8, 8),
                 9: (16, 8, 8),
                 10: (16, 8, 8),
                 11: (16, 8, 8),
                 12: (16, 8, 8),
                 13: (16, 8, 8),
                 14: (16, 16, 16),
                 15: (16, 16, 16),
                 16: (16, 16, 16),
                 17: (16, 16, 16),
                 18: (16, 16, 16),
                 19: (16, 16, 16),
                 20: (16, 16, 16),
                 21: (16, 16, 16),
                 22: (16, 16, 16),
                 23: (16, 16, 16),
                 24: (16, 16, 16),
                 25: (16, 16, 16),
                 26: (16, 16, 16),
                 27: (16, 16, 16),
                 28: (16, 16, 16),
                 29: (16, 16, 16),
                 30: (16, 32, 32)}
        
        
        adjust=self.latent_adjust_values
        latents={}
        bs=data.shape[0]
        for k,v in self.brain_to_latent.items():
            s=shapes[k]
            z=torch.tensor(v.predict(data)).reshape(-1,*s)


            if adjust is not None and bs>1:
                #compute actual mean and std
                                
                z_mean=self.predicted_latent_stats[k]["mean"]  
                z_std=self.predicted_latent_stats[k]["std"] 
                
                
                
                #standardize 
                z = (z - z_mean)/(1e-9+z_std)

                #replace with latent mean and std
                z = z*adjust[k]["std"]+adjust[k]["mean"]

            latents[k]=z

        return latents
    
    def decode_with_partial_sampling(self,latents,keep=None):
        xs = {a.shape[2]: a for a in self.vae.decoder.bias_xs}
        
        decoder=self.vae.decoder.to(self.device)
        out=decoder.forward_manual_latents(keep,latents.values(),t=None)

        xs=decoder.out_net.sample(out)
        xs=torch.tensor(xs).permute(0,3,1,2)/255
        return xs
                                             
    def decode_features(self,fmri):
        
        #get latents
        z=self.get_latents(fmri.numpy())
        
        adjust=self.latent_adjust_values
        
        img_emb=[]
        txt_emb=[]
        for i in tqdm.tqdm(range(257)):
            emb=torch.tensor(self.brain_to_img_emb[i].predict(fmri.numpy()))
            # print(emb.shape)
            if adjust and len(fmri)>1:
                #compute actual mean and std
                emb_mean=self.predicted_img_emb_mean[i]
                emb_std=self.predicted_img_emb_std[i]

                emb= (emb-emb_mean)/emb_std
                emb = emb*self.clip_img_embeds_std[i]+self.clip_img_embeds_mean[i]

            img_emb.append(emb)

        for i in tqdm.tqdm(range(77)):


            emb=torch.tensor(self.brain_to_txt_emb[i].predict(fmri.numpy()))

            if adjust and len(fmri)>1:
                #compute actual mean and std
                
                emb_mean=self.predicted_txt_emb_mean[i]
                emb_std=self.predicted_txt_emb_std[i]
                
                emb= (emb-emb_mean)/emb_std

                emb = emb*self.clip_txt_embeds_std[i]+self.clip_txt_embeds_mean[i]
            txt_emb.append(emb)
                                             
        img_emb=torch.stack(img_emb,1)
        txt_emb=torch.stack(txt_emb,1)
        
        return z, img_emb, txt_emb
        
        
    def reconstruct_guess(self,fmri):
        upsample=torchvision.transforms.Resize(512,interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        
        z, img_emb, txt_emb = self.decode_features(fmri)
        
        with torch.no_grad():

            latents={k:v.to(self.device).float() for k,v in z.items()}
            # guess_img=upsample(autoencoder.decoder.double()(z.to(device)).cpu())
            guess_img=self.decode_with_partial_sampling(latents=latents,keep=len(fmri))
            # img_out=pipe_embed.vae.float().decode(z.float().to(device)).sample.cpu()
            print(guess_img.max())
            guess_img=upsample(guess_img).clamp(0,1)
        
        
        return guess_img, z, img_emb, txt_emb
    
    
    def decode(self,fmri,strength=7.5,text_to_image_strength=0.4, num_inference_steps=37,how_many=1, use_latents=True,scale=False):
        
        to_pil=torchvision.transforms.ToPILImage()

        if scale:
            frmi= (fmri- self.train_fmri_mean)/self.train_fmri_std
            fmri= torch.nan_to_num(fmri)
        
        # decode initial guess and featuers
        guess_img, z, img_emb, txt_emb=self.reconstruct_guess(fmri)
        
        
        # encode null img and null prompt
        null_prompt=""
        null_img=Image.fromarray(np.zeros((425,425,3),dtype=np.uint8))
        uimg=self.pipe_embed._encode_image_prompt([null_img],device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False).cpu()
        utxt=self.pipe_embed._encode_text_prompt([null_prompt],device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False).cpu()
        
        
        #decode the final images
        
        scale=self.pipe_embed.vae.config.scaling_factor
        images=[]
        for i in range(len(fmri)):
            with torch.no_grad():
                print(f"[INFO] Final reconstrution {i+1}/{len(fmri)}")
                encoded_latents=scale*self.pipe_embed.vae.encode((2*guess_img[i:i+1]-1).to(self.device)).latent_dist.sample()
                noise = randn_tensor((how_many,encoded_latents.shape[1],encoded_latents.shape[2],encoded_latents.shape[3]), device=self.device, dtype=encoded_latents.dtype)
                encoded_latents_norm=(encoded_latents-encoded_latents.mean())//(1e-8+encoded_latents.std())
                #final_latents=pipe_embed.scheduler.add_noise(0.0*(encoded_latents_norm.clamp(-3,3)),noise,torch.tensor(50).long().to(device))

                #final_latents=noise+0.18*encoded_latents_norm.clamp(-3,3)
                final_latents=noise+scale*encoded_latents.clamp(-3,3)
                final_latents = (final_latents - final_latents.mean())/final_latents.std()
                
                if use_latents:
                    final_latents=noise+scale*encoded_latents.clamp(-3,3)
                    final_latents = (final_latents - final_latents.mean())/final_latents.std()
                 
                else:
                    final_latents=noise
                

                if strength>1:
                    txt_cond=torch.cat([utxt.repeat(how_many,1,1),txt_emb[i:i+1].float().repeat(how_many,1,1)],0)

                    img_cond=torch.cat([uimg.repeat(how_many,1,1),img_emb[i:i+1].float().repeat(how_many,1,1)],0)
                else:
                    txt_cond=txt_emb[i:i+1].float().repeat(how_many,1,1)
                    img_cond=img_emb[i:i+1].float().repeat(how_many,1,1)

                # print(txt_emb[i:i+1].float().repeat(how_many,1,1).shape,img_emb[i:i+1].float().repeat(how_many,1,1).shape,final_latents.shape)

                # image_generated = pipe_embed([null_prompt]*bs,guessed,txt_cond.to(device), img_cond.to(device), text_to_image_strength=0.4,num_inference_steps=37,guidance_scale=strength,latents=final_latents).images
                image_generated = self.pipe_embed([null_prompt]*how_many,[null_img]*how_many,txt_cond.to(self.device), img_cond.to(self.device), text_to_image_strength=text_to_image_strength,num_inference_steps=num_inference_steps,guidance_scale=strength,latents=final_latents).images
                images+=image_generated
    
        guessed=[to_pil(i) for i in guess_img]
        
        
        return images, guessed
            
                                                    
            
                                            
                
                


class BrainDiffuserDecoder:
    def __init__(self,vae_weights="/home/matteo/models/vdvae/vae2.pt",
                 vae_hyper='/home/matteo/models/vdvae/H.sav', 
                 pretrained=True,
                 subj_path=None,
                 device="cpu", sub="subj02",save=True):
        super().__init__()
        self.keep=31
        self.device=device
        self.pretrained=pretrained
        self.subj_path=subj_path
        self.sub=sub
        
        self.shapes={0:(16,1,1),
                1: (16, 1, 1),
                 2: (16, 4, 4),
                 3: (16, 4, 4),
                 4: (16, 4, 4),
                 5: (16, 4, 4),
                 6: (16, 8, 8),
                 7: (16, 8, 8),
                 8: (16, 8, 8),
                 9: (16, 8, 8),
                 10: (16, 8, 8),
                 11: (16, 8, 8),
                 12: (16, 8, 8),
                 13: (16, 8, 8),
                 14: (16, 16, 16),
                 15: (16, 16, 16),
                 16: (16, 16, 16),
                 17: (16, 16, 16),
                 18: (16, 16, 16),
                 19: (16, 16, 16),
                 20: (16, 16, 16),
                 21: (16, 16, 16),
                 22: (16, 16, 16),
                 23: (16, 16, 16),
                 24: (16, 16, 16),
                 25: (16, 16, 16),
                 26: (16, 16, 16),
                 27: (16, 16, 16),
                 28: (16, 16, 16),
                 29: (16, 16, 16),
                 30: (16, 32, 32)}
        

        print("Loading pretrained deep learning backbones")

        with open(vae_hyper, 'rb') as fp:
            d = pickle.load(fp)

        H=Hyperparams()
        for k,v in d.items():
            H[k]=v
            
        vae=VAE(H)    
        state_dict = torch.load(vae_weights)
        new_state_dict = {}
        l = len('module.')
        for k in state_dict:
            if k.startswith('module.'):
                new_state_dict[k[l:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
        state_dict = new_state_dict
        vae.load_state_dict(state_dict)


        self.vae=vae.to(device)


        self.pipe_embed= VersatileDiffusionDualGuidedFromCLIPEmbeddingPipeline.from_pretrained("shi-labs/versatile-diffusion",)

        self.pipe_embed.remove_unused_weights()
        self.pipe_embed.to(self.device)
        self.transform=torchvision.transforms.Compose([to_pil,torchvision.transforms.Resize(64),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=110/255,std=69/255)])
        
    def compute_train_dataset(self,train_dataloader,save=True):
        train_fmri=[]
        train_imgs=[]
        train_captions=[]
        train_z={}
        train_clip_img_embeds=[]
        train_clip_txt_embeds=[]
        train_clip_pool_txt=[]
        to_pil=torchvision.transforms.ToPILImage()
        
        first=True
        guidance_scale = 7.5
        num_images_per_prompt =1
        do_classifier_free_guidance = False
        keep=self.keep
        device=self.device
        
        for x,y,c in tqdm.tqdm(train_dataloader):

            #save fMRI data
            train_fmri.append(x)

            #save img data
            train_imgs.append(y)

            train_captions+=list(c)

            #encode images in autoencoder and save z representation
            with torch.no_grad():
                T=torch.stack([self.transform(i) for i in y])
                act=self.vae.encoder.forward(T.to(self.device))
                px_z, stats = self.vae.decoder.forward(act, get_latents=True)

                latents=[i["z"] for i in stats[:keep]]

                if first:
                    z={k:v.cpu().clamp(-10,10) for k,v in zip(np.arange(keep),latents)}
                    train_z.update(z)
                    first=False
                else:
                    z={k:v.cpu().clamp(-10,10) for k,v in zip(np.arange(keep),latents)}

                    for k in train_z.keys():
                        train_z[k]=torch.cat([train_z[k],z[k]],axis=0)





                #encode images in CLIP
                image_features=self.pipe_embed._encode_image_prompt([to_pil(i) for i in y],device=device,num_images_per_prompt=num_images_per_prompt,do_classifier_free_guidance=do_classifier_free_guidance).cpu()
                train_clip_img_embeds.append(image_features)

                #encode text in clip
                text_features=self.pipe_embed._encode_text_prompt(c,device=device,num_images_per_prompt=num_images_per_prompt,do_classifier_free_guidance=do_classifier_free_guidance).cpu()
                train_clip_txt_embeds.append(text_features)

                #txt pool
                # text = clip.tokenize(c).to(device  )
                # text_pool_features = model.encode_text(text).cpu()
                # train_clip_pool_txt.append(text_pool_features)

        train_clip_txt_embeds = torch.cat(train_clip_txt_embeds,axis=0)
        train_clip_img_embeds = torch.cat(train_clip_img_embeds,axis=0)

        train_fmri = torch.cat(train_fmri,axis=0)
        # train_z = torch.cat(train_z,axis=0)  
        # train_z={k:torch.cat(v,axis=0) for k,v in train_z.items()}
        train_imgs = torch.cat(train_imgs,axis=0)
        # train_clip_pool_txt = torch.cat(train_clip_pool_txt,axis=0)
        
        

        for k in train_z.keys():
            train_z[k]=torch.nan_to_num(train_z[k])

        
        if save:
            sub=self.sub
            os.makedirs(f"models/{sub}",exist_ok=True)
    
            ## train
            torch.save(train_fmri,f"models/{sub}/train_fmri.pt")
            torch.save(train_clip_txt_embeds,f"models/{sub}/train_clip_txt_embeds.pt")
            torch.save(train_clip_img_embeds,f"models/{sub}/train_clip_img_embeds.pt")
            torch.save(train_imgs,f"models/{sub}/train_imgs.pt")
            with open(f"models/{sub}/train_z.sav","wb") as f:
                pickle.dump(train_z,f)

            with open(f"models/{sub}/train_captions.sav","wb") as f:
                pickle.dump(train_captions,f)

            print("saved training stuff")
        
        return train_fmri,train_imgs,train_captions,train_z,train_clip_img_embeds,train_clip_txt_embeds,train_clip_pool_txt
    
    
    # fix nan

    
    def compute_test_dataset(self,test_dataloader,save=True):
        test_fmri=[]
        test_imgs=[]
        test_captions=[]
        test_z={}
        test_clip_img_embeds=[]
        test_clip_txt_embeds=[]
        test_clip_pool_txt=[]
        to_pil=torchvision.transforms.ToPILImage()
        
        first=True
        guidance_scale = 7.5
        num_images_per_prompt =1
        do_classifier_free_guidance = False
        keep=self.keep
        device=self.device
        
        for x,y,c in tqdm.tqdm(test_dataloader):

            #save fMRI data
            test_fmri.append(x)

            #save img data
            test_imgs.append(y)

            test_captions+=list(c)

            #encode images in autoencoder and save z representation
            with torch.no_grad():
                T=torch.stack([self.transform(i) for i in y])
                act=self.vae.encoder.forward(T.to(self.device))
                px_z, stats = self.vae.decoder.forward(act, get_latents=True)

                latents=[i["z"] for i in stats[:keep]]

                if first:
                    z={k:v.cpu().clamp(-10,10) for k,v in zip(np.arange(keep),latents)}
                    test_z.update(z)
                    first=False
                else:
                    z={k:v.cpu().clamp(-10,10) for k,v in zip(np.arange(keep),latents)}

                    for k in test_z.keys():
                        test_z[k]=torch.cat([test_z[k],z[k]],axis=0)





                #encode images in CLIP
                image_features=self.pipe_embed._encode_image_prompt([to_pil(i) for i in y],device=device,num_images_per_prompt=num_images_per_prompt,do_classifier_free_guidance=do_classifier_free_guidance).cpu()
                test_clip_img_embeds.append(image_features)

                #encode text in clip
                text_features=self.pipe_embed._encode_text_prompt(c,device=device,num_images_per_prompt=num_images_per_prompt,do_classifier_free_guidance=do_classifier_free_guidance).cpu()
                test_clip_txt_embeds.append(text_features)

                #txt pool
                # text = clip.tokenize(c).to(device  )
                # text_pool_features = model.encode_text(text).cpu()
                # test_clip_pool_txt.append(text_pool_features)
                
        for k in test_z.keys():
            test_z[k]=torch.nan_to_num(test_z[k])

        test_clip_txt_embeds = torch.cat(test_clip_txt_embeds,axis=0)
        test_clip_img_embeds = torch.cat(test_clip_img_embeds,axis=0)

        test_fmri = torch.cat(test_fmri,axis=0)
        # test_z = torch.cat(test_z,axis=0)  
        # test_z={k:torch.cat(v,axis=0) for k,v in test_z.items()}
        test_imgs = torch.cat(test_imgs,axis=0)
        # test_clip_pool_txt = torch.cat(test_clip_pool_txt,axis=0)
        
        if save:
            sub=self.sub
            os.makedirs(f"models/{sub}",exist_ok=True)
    
            ## test
            torch.save(test_fmri,f"models/{sub}/test_fmri.pt")
            torch.save(test_clip_txt_embeds,f"models/{sub}/test_clip_txt_embeds.pt")
            torch.save(test_clip_img_embeds,f"models/{sub}/test_clip_img_embeds.pt")
            torch.save(test_imgs,f"models/{sub}/test_imgs.pt")
            with open(f"models/{sub}/test_z.sav","wb") as f:
                pickle.dump(test_z,f)

            with open(f"models/{sub}/test_captions.sav","wb") as f:
                pickle.dump(test_captions,f)

            print("saved testing stuff")
        
        return test_fmri,test_imgs,test_captions,test_z,test_clip_img_embeds,test_clip_txt_embeds,test_clip_pool_txt
    
    
    def fit_brain_to_latent(self,train_fmri_norm,train_z):
        brain_to_latent ={}
        keys=train_z.keys()
        alphas=[5e4]*len(keys)
        for k,alpha in tqdm.tqdm(list(zip(keys,alphas))):
            brain_vdvae_latent=Ridge(alpha, max_iter=10000, fit_intercept=True)
            brain_vdvae_latent.fit(train_fmri_norm.numpy(),train_z[k].reshape(train_z[k].shape[0],-1).numpy())
            brain_to_latent[k]=brain_vdvae_latent
        return brain_to_latent
    
    def fit_brain_to_img_emb(self,train_fmri_norm,train_clip_img_embeds):
        max_len_img=257
        brain_to_img_emb=[]

        for i in tqdm.tqdm(range(max_len_img)):
            m=Ridge(alpha=6e4)
            m.fit(train_fmri_norm.numpy(),train_clip_img_embeds[:,i,:].numpy())
            brain_to_img_emb.append(m)
            
        
        return brain_to_img_emb
    
    def fit_brain_to_txt_emb(self,train_fmri_norm,train_clip_txt_embeds):
        max_len_txt=77
        brain_to_txt_emb=[]

        for i in tqdm.tqdm(range(max_len_txt)):
            m=Ridge(alpha=1e5)
            m.fit(train_fmri_norm.numpy(),train_clip_txt_embeds[:,i,:].numpy())
            brain_to_txt_emb.append(m)
            
        
        return brain_to_txt_emb
    
    
    

    def fit(self,train_dataloader,save=True):
        
        
        sub=self.sub
        shapes=self.shapes
        ## extract latents
        print("Extracting latent space for training set")
        train_fmri,train_imgs,train_captions,train_z,train_clip_img_embeds,train_clip_txt_embeds,train_clip_pool_txt= self.compute_train_dataset(train_dataloader, save=save)
        
        self.train_fmri_mean=torch.mean(train_fmri,axis=0)
        self.train_fmri_std=torch.std(train_fmri,axis=0)
        
        train_fmri_norm=(train_fmri-self.train_fmri_mean)/self.train_fmri_std
        train_fmri_norm=torch.nan_to_num(train_fmri_norm)
        
        ## train brain to latent model
        print("Fit brain to latents model")
        self.brain_to_latent=self.fit_brain_to_latent(train_fmri_norm,train_z)
        
        print("Fit brain to img embeds model")
        self.brain_to_img_emb=self.fit_brain_to_img_emb(train_fmri_norm,train_clip_img_embeds)
        
        print("Fit brain to txt embeds model")
        self.brain_to_txt_emb=self.fit_brain_to_txt_emb(train_fmri_norm,train_clip_txt_embeds)
        

        
        stats={}

        ## compute adjusting values
        print("Computing adjust values")
        
        for k,v in self.brain_to_latent.items():
            s=shapes[k]
            z=torch.tensor(v.predict(train_fmri_norm.numpy())).reshape(-1,*s)

            stats[k]={"mean":z.mean(0),"std":z.std(0)}
        

                
        self.predicted_latent_stats=stats
        
        latent_adjust_values={}
        for i in range(self.keep):
            latent_adjust_values[i]={"mean":train_z[i].mean(0), "std": train_z[i].std(0)}
        
        self.latent_adjust_values=latent_adjust_values
        
        
        
        img_emb=[]
        txt_emb=[]

        for i in tqdm.tqdm(range(257)):
            emb=torch.tensor(self.brain_to_img_emb[i].predict(train_fmri_norm.numpy()))
            img_emb.append(emb)


        for i in tqdm.tqdm(range(77)):
            emb=torch.tensor(self.brain_to_txt_emb[i].predict(train_fmri_norm.numpy()))
            txt_emb.append(emb)

        img_emb=torch.stack(img_emb,1)
        txt_emb=torch.stack(txt_emb,1)
        predicted_img_emb_mean=img_emb.mean(0)
        predicted_img_emb_std=img_emb.std(0)

        predicted_txt_emb_mean=txt_emb.mean(0)
        predicted_txt_emb_std=txt_emb.std(0)
        
        
        ## true values
        self.clip_img_embeds_mean=train_clip_img_embeds.mean(0)
        self.clip_img_embeds_std=train_clip_img_embeds.std(0)


        self.clip_txt_embeds_mean=train_clip_txt_embeds.mean(0)
        self.clip_txt_embeds_std=train_clip_txt_embeds.std(0)
        
        self.predicted_img_emb_mean=predicted_img_emb_mean
        self.predicted_img_emb_std=predicted_img_emb_std
        
        self.predicted_txt_emb_mean=predicted_txt_emb_mean
        self.predicted_txt_emb_std=predicted_txt_emb_std
        

        if save:
            filename="predicted_latent_stats.sav"

            with open(opj(f"models/{sub}",filename),"wb") as f:
                pickle.dump(stats,f)
        
            filename = f'latent_adjust_values.sav'
            with open(opj(f"models/{sub}",filename), 'wb') as f:
                pickle.dump(latent_adjust_values, f)

            # Define the file paths
            img_emb_mean_path = f"models/{sub}/predicted_img_emb_mean.pt"
            img_emb_std_path = f"models/{sub}/predicted_img_emb_std.pt"
            txt_emb_mean_path = f"models/{sub}/predicted_txt_emb_mean.pt"
            txt_emb_std_path = f"models/{sub}/predicted_txt_emb_std.pt"

            # Save the tensors
            torch.save(predicted_img_emb_mean, img_emb_mean_path)
            torch.save(predicted_img_emb_std, img_emb_std_path)
            torch.save(predicted_txt_emb_mean, txt_emb_mean_path)
            torch.save(predicted_txt_emb_std, txt_emb_std_path)

            torch.save(self.train_fmri_mean,f"models/{sub}/train_fmri_mean.pt")
            torch.save(self.train_fmri_std,f"models/{sub}/train_fmri_std.pt")
            
            torch.save(self.clip_img_embeds_mean, opj(f"models/{sub}","clip_img_embeds_mean.pt"))
            torch.save(self.clip_img_embeds_std, opj(f"models/{sub}","clip_img_embeds_std.pt"))
            torch.save(self.clip_txt_embeds_mean, opj(f"models/{sub}","clip_txt_embeds_mean.pt"))
            torch.save(self.clip_txt_embeds_std, opj(f"models/{sub}","clip_txt_embeds_std.pt"))

        
        #eventually save models separately
        if save:
            
            print("saving all models separately")
            
            os.makedirs(f"models/{sub}/decoding",exist_ok=True)
            for i in train_z.keys():
                filename = f'brain_to_vdvae_latent_ridge_{i}.sav'
                with open(opj(f"models/{sub}/decoding",filename), 'wb') as f:
                    pickle.dump(self.brain_to_latent[i], f)

            for i in range(257):
                filename = f'brain_to_img_emb_ridge_{i}.sav'
                with open(opj(f"models/{sub}/decoding",filename), 'wb') as f:
                    pickle.dump(self.brain_to_img_emb[i], f)

            for i in range(77):
                filename = f'brain_to_txt_emb_ridge_{i}.sav'
                with open(opj(f"models/{sub}/decoding",filename), 'wb') as f:
                    pickle.dump(self.brain_to_txt_emb[i], f)
    
        
    def get_latents(self,data):
        shapes=self.shapes
        
        adjust=self.latent_adjust_values
        latents={}
        bs=data.shape[0]
        for k,v in self.brain_to_latent.items():
            s=shapes[k]
            z=torch.tensor(v.predict(data)).reshape(-1,*s)


            if adjust is not None and bs>1:
                #compute actual mean and std
                                
                z_mean=self.predicted_latent_stats[k]["mean"]  
                z_std=self.predicted_latent_stats[k]["std"] 
                
                
                
                #standardize 
                z = (z - z_mean)/(1e-9+z_std)

                #replace with latent mean and std
                z = z*adjust[k]["std"]+adjust[k]["mean"]

            latents[k]=z

        return latents
    
    def decode_with_partial_sampling(self,latents,keep=None):
        xs = {a.shape[2]: a for a in self.vae.decoder.bias_xs}
        
        decoder=self.vae.decoder.to(self.device)
        out=decoder.forward_manual_latents(keep,latents.values(),t=None)

        xs=decoder.out_net.sample(out)
        xs=torch.tensor(xs).permute(0,3,1,2)/255
        return xs
                                             
    def decode_features(self,fmri):
        
        #get latents
        z=self.get_latents(fmri.numpy())
        
        adjust=self.latent_adjust_values
        
        img_emb=[]
        txt_emb=[]
        for i in tqdm.tqdm(range(257)):
            emb=torch.tensor(self.brain_to_img_emb[i].predict(fmri.numpy()))
            # print(emb.shape)
            if adjust and len(fmri)>1:
                #compute actual mean and std
                emb_mean=self.predicted_img_emb_mean[i]
                emb_std=self.predicted_img_emb_std[i]

                emb= (emb-emb_mean)/emb_std
                emb = emb*self.clip_img_embeds_std[i]+self.clip_img_embeds_mean[i]

            img_emb.append(emb)

        for i in tqdm.tqdm(range(77)):


            emb=torch.tensor(self.brain_to_txt_emb[i].predict(fmri.numpy()))

            if adjust and len(fmri)>1:
                #compute actual mean and std
                
                emb_mean=self.predicted_txt_emb_mean[i]
                emb_std=self.predicted_txt_emb_std[i]
                
                emb= (emb-emb_mean)/emb_std

                emb = emb*self.clip_txt_embeds_std[i]+self.clip_txt_embeds_mean[i]
            txt_emb.append(emb)
                                             
        img_emb=torch.stack(img_emb,1)
        txt_emb=torch.stack(txt_emb,1)
        
        return z, img_emb, txt_emb
        
        
    def reconstruct_guess(self,fmri):
        upsample=torchvision.transforms.Resize(512,interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        
        z, img_emb, txt_emb = self.decode_features(fmri)
        
        with torch.no_grad():

            latents={k:v.to(self.device).float() for k,v in z.items()}
            # guess_img=upsample(autoencoder.decoder.double()(z.to(device)).cpu())
            guess_img=self.decode_with_partial_sampling(latents=latents,keep=len(fmri))
            # img_out=pipe_embed.vae.float().decode(z.float().to(device)).sample.cpu()
            print(guess_img.max())
            guess_img=upsample(guess_img).clamp(0,1)
        
        
        return guess_img, z, img_emb, txt_emb
    
    
    def decode(self,fmri,strength=7.5,text_to_image_strength=0.4, num_inference_steps=37,how_many=1, use_latents=True, scale=False):
        
        if scale:
            frmi= (fmri- self.train_fmri_mean)/self.train_fmri_std
            fmri= torch.nan_to_num(fmri)
        
        to_pil=torchvision.transforms.ToPILImage()

        
        # decode initial guess and featuers
        guess_img, z, img_emb, txt_emb=self.reconstruct_guess(fmri)
        
        
        # encode null img and null prompt
        null_prompt=""
        null_img=Image.fromarray(np.zeros((425,425,3),dtype=np.uint8))
        uimg=self.pipe_embed._encode_image_prompt([null_img],device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False).cpu()
        utxt=self.pipe_embed._encode_text_prompt([null_prompt],device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False).cpu()
        
        
        #decode the final images
        
        scale=self.pipe_embed.vae.config.scaling_factor
        images=[]
        for i in range(len(fmri)):
            with torch.no_grad():
                print(f"[INFO] Final reconstrution {i+1}/{len(fmri)}")
                encoded_latents=scale*self.pipe_embed.vae.encode((2*guess_img[i:i+1]-1).to(self.device)).latent_dist.sample()
                noise = randn_tensor((how_many,encoded_latents.shape[1],encoded_latents.shape[2],encoded_latents.shape[3]), device=self.device, dtype=encoded_latents.dtype)
                encoded_latents_norm=(encoded_latents-encoded_latents.mean())//(1e-8+encoded_latents.std())
                #final_latents=pipe_embed.scheduler.add_noise(0.0*(encoded_latents_norm.clamp(-3,3)),noise,torch.tensor(50).long().to(device))

                #final_latents=noise+0.18*encoded_latents_norm.clamp(-3,3)
                final_latents=noise+scale*encoded_latents.clamp(-3,3)
                final_latents = (final_latents - final_latents.mean())/final_latents.std()
                
                if use_latents:
                    final_latents=noise+scale*encoded_latents.clamp(-3,3)
                    final_latents = (final_latents - final_latents.mean())/final_latents.std()
                 
                else:
                    final_latents=noise
                

                if strength>1:
                    txt_cond=torch.cat([utxt.repeat(how_many,1,1),txt_emb[i:i+1].float().repeat(how_many,1,1)],0)

                    img_cond=torch.cat([uimg.repeat(how_many,1,1),img_emb[i:i+1].float().repeat(how_many,1,1)],0)
                else:
                    txt_cond=txt_emb[i:i+1].float().repeat(how_many,1,1)
                    img_cond=img_emb[i:i+1].float().repeat(how_many,1,1)

                # print(txt_emb[i:i+1].float().repeat(how_many,1,1).shape,img_emb[i:i+1].float().repeat(how_many,1,1).shape,final_latents.shape)

                # image_generated = pipe_embed([null_prompt]*bs,guessed,txt_cond.to(device), img_cond.to(device), text_to_image_strength=0.4,num_inference_steps=37,guidance_scale=strength,latents=final_latents).images
                image_generated = self.pipe_embed([null_prompt]*how_many,[null_img]*how_many,txt_cond.to(self.device), img_cond.to(self.device), text_to_image_strength=text_to_image_strength,num_inference_steps=num_inference_steps,guidance_scale=strength,latents=final_latents).images
                images+=image_generated
    
        guessed=[to_pil(i) for i in guess_img]
        
        
        return images, guessed