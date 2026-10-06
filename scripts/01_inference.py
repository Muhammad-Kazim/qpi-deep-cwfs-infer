import os
import sys
sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-1]))

from qpi_deep_cwfs import utils
from qpi_deep_cwfs import hRAFT, hSea_RAFT

import numpy as np
from tifffile import tifffile
from matplotlib import pyplot as plt

import torch
import torchvision
from torchvision.models.optical_flow import raft_large

from torchvision.utils import flow_to_image
from scipy.ndimage import gaussian_filter

from datetime import datetime
import json


def run_model_inference(model:torch.nn.Module, img1_batch:torch.Tensor, img2_batch:torch.Tensor, 
                        model_name:str, model_settings:dict):

    model_name = model_name.lower()
    if 'sea_raft_medium' in model_name:
        list_of_flows = model(img1_batch, img2_batch,**model_settings)['final']
        # if 'sea_raft' in model_name:
        #     list_of_flows = list_of_flows['flow']
    else:
        list_of_flows = model(img1_batch, img2_batch)[-1]
    
    return list_of_flows


if __name__=='__main__':
    """
    Inference on SynthELlips test set using ckpt and cwfs add method based on json file in configs/infer....json
    """
    JSON_PATH = '../configs/args.json'
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r') as file:
            params = json.load(file)
    else:
        print(f'{JSON_PATH} does not exist.')
        raise
    
    date = f'{datetime.today()}'.split()[0]

    PX_SIZE = float(params['PX_SIZE'])
    DIST_PM_IM = float(params['DIST_PM_IM'])
    
    DEVICE = params['DEVICE']
    
    REF_PATH = params['REF_PATH']
    OBJ_PATH = params['OBJ_PATH']
    CROP_REGION = params['CROP_REGION']
    
    SAVE_PATH = params['SAVE_PATH']
    SAVE_NAME = params['SAVE_NAME']
    
    MODEL_NAME = params['MODEL_NAME']
    CKPT_PATH = params['CKPT_PATH']
    
    PATCH_SIZE = int(params['PATCH_SIZE'])
    OVERLAP = int(params['OVERLAP'])
    
    CMAP = params['CMAP']
    
    if DEVICE == 'cuda':
            assert torch.cuda.is_available(), "CUDA not available. Switch to CPU."
    print(DEVICE)
    
    assert os.path.exists(REF_PATH), f"reference image does not exist at {REF_PATH}"
    assert os.path.exists(OBJ_PATH), f"object image does not exist at {OBJ_PATH}"
    assert os.path.exists(SAVE_PATH), f"file already exists at {SAVE_PATH}"
    assert os.path.exists(CKPT_PATH), f"checkpoint does not exist at {CKPT_PATH}"
    
    
    model, transforms, model_settings = utils.get_model_and_transforms(MODEL_NAME, DEVICE, checkpoint=CKPT_PATH)

    preprocess_map = {'sea_raft': hSea_RAFT.preprocess, 'raft': hRAFT.preprocess, 'gmflow': '', 'gma': ''}
    preprocess_fn = next((preprocess_map[key] for key in preprocess_map if key in MODEL_NAME.lower()), hRAFT.preprocess)

    model = model.to(DEVICE)
    model.eval()
    
    I0 = torch.tensor(tifffile.imread(REF_PATH)).float()[CROP_REGION[0][0]:CROP_REGION[1][0], CROP_REGION[0][1]:CROP_REGION[1][1]]
    I1 = torch.tensor(tifffile.imread(OBJ_PATH)).float()[CROP_REGION[0][0]:CROP_REGION[1][0], CROP_REGION[0][1]:CROP_REGION[1][1]]
    
    shape = I0.shape
    
    with torch.no_grad():
        if shape[0] == PATCH_SIZE:
            ref_imgs = [I0/I0.max()]
            obj_imgs = [I1/I0.max()]
            
            img1_batch, img2_batch = preprocess_fn(ref_imgs, obj_imgs, transforms)
            list_of_flows = run_model_inference(model, img1_batch.to(DEVICE), img2_batch.to(DEVICE), MODEL_NAME, model_settings)

            flow0, flow1 = torchvision.transforms.functional.resize(list_of_flows.detach().cpu(), size=[PATCH_SIZE, PATCH_SIZE])[0]
            nn_opd = -1e6*PX_SIZE**2/DIST_PM_IM*utils.integrate_flow_field_torch(torch.stack([flow1, flow0], dim=2))
            
        elif shape[0] < PATCH_SIZE or shape[1] < PATCH_SIZE:
            if PATCH_SIZE > shape[0]:
                nx = int((PATCH_SIZE - shape[0])/2)
                nI0 = torch.nn.functional.pad(I0.unsqueeze(0), [0, 0, 0, 2*nx], 'reflect').squeeze()
                nI1 = torch.nn.functional.pad(I1.unsqueeze(0), [0, 0, 0, 2*nx], 'reflect').squeeze()

            if PATCH_SIZE > shape[1]:
                nx = int((PATCH_SIZE - shape[1])/2)
                nI0 = torch.nn.functional.pad(nI0.unsqueeze(0), [0, 2*nx, 0, 0], 'reflect').squeeze()
                nI1 = torch.nn.functional.pad(nI1.unsqueeze(0), [0, 2*nx, 0, 0], 'reflect').squeeze()

            ref_imgs = [nI0/nI0.max()]
            obj_imgs = [nI1/nI0.max()]

            img1_batch, img2_batch = preprocess_fn(ref_imgs, obj_imgs, transforms)
            list_of_flows = run_model_inference(model, img1_batch.to(DEVICE), img2_batch.to(DEVICE), MODEL_NAME, model_settings)

            flow0, flow1 = torchvision.transforms.functional.resize(list_of_flows.detach().cpu(), size=[PATCH_SIZE, PATCH_SIZE])[0]
            nn_opd = -1e6*PX_SIZE**2/DIST_PM_IM*utils.integrate_flow_field_torch(torch.stack([flow1, flow0], dim=2))[:shape[0], :shape[1]]
                    
        else:
            crop_array = utils.make_crop_array(shape, PATCH_SIZE, OVERLAP)
            
            grad_0s = []
            grad_1s = []
            
            for idx, crop in enumerate(crop_array):
                print(f'{idx} of {len(crop_array)} crops')
                ref_imgs = []
                obj_imgs = []
                
                X, Y = crop

                nI0 = I0[X:X+PATCH_SIZE, Y:Y+PATCH_SIZE]
                nI1 = I1[X:X+PATCH_SIZE, Y:Y+PATCH_SIZE]

                if X + PATCH_SIZE >= shape[0]:
                    nx = int((X + PATCH_SIZE - shape[0])/2)
                    nI0 = torch.nn.functional.pad(nI0.unsqueeze(0), [0, 0, 0, 2*nx], 'reflect').squeeze()
                    nI1 = torch.nn.functional.pad(nI1.unsqueeze(0), [0, 0, 0, 2*nx], 'reflect').squeeze()

                if Y + PATCH_SIZE >= shape[1]:
                    nx = int((Y + PATCH_SIZE - shape[1])/2)
                    nI0 = torch.nn.functional.pad(nI0.unsqueeze(0), [0, 2*nx, 0, 0], 'reflect').squeeze()
                    nI1 = torch.nn.functional.pad(nI1.unsqueeze(0), [0, 2*nx, 0, 0], 'reflect').squeeze()

                ref_imgs = [nI0/nI0.max()]
                obj_imgs = [nI1/nI0.max()]
                
                img1_batch, img2_batch = preprocess_fn(ref_imgs, obj_imgs, transforms)
                list_of_flows = run_model_inference(model, img1_batch.to(DEVICE), img2_batch.to(DEVICE), MODEL_NAME, model_settings)

                flow0, flow1 = torchvision.transforms.functional.resize(list_of_flows.detach().cpu(), size=[PATCH_SIZE, PATCH_SIZE])[0]

                grad_0s.append(flow0)
                grad_1s.append(flow1)


            grad0_tot, grad1_tot = utils.grad_stitch_n([grad_0s, grad_1s], overlap=OVERLAP)
            nn_opd = -1e6*PX_SIZE**2/DIST_PM_IM*utils.integrate_flow_field_torch(torch.stack([grad1_tot[:shape[0], :shape[1]], grad0_tot[:shape[0], :shape[1]]], dim=2))
    
    
    fig, axs = plt.subplots(figsize=(10, 10))
    cm = axs.imshow(nn_opd, cmap=CMAP)
    plt.colorbar(cm, ax=axs)
    
    if SAVE_PATH:
        plt.savefig(f'{SAVE_PATH}/{date}_{SAVE_NAME}.png', dpi=200, bbox_inches='tight')
    
    plt.show()
