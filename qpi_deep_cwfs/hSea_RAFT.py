import os 
import torch
import torchvision.transforms.functional as F
from typing import Optional, Union
from torch import nn, Tensor

class SeaRAFTSettings:
    INPUT_SIZE = [540, 960]  # height, width # not 520,960 but 540x960 to be divisible by 8 after downsampling in the model


def init_model_Sea_RAFT(model, device='cuda', checkpoint=None,model_name='sea_raft_medium'):

    assert model_name in ['sea_raft_medium', 'sea_raft_small'], "model_name must be either 'sea_raft_medium' or 'sea_raft_small'"

    print(f'Loading model with checkpoint: {checkpoint}')
    weights = torch.load(checkpoint, map_location=torch.device(device))
    if 'model' in weights:
        weights = weights['model']
    encoder_output_channels = weights['fnet.conv1.weight'].shape[0]
    
    model.fnet.conv1 = torch.nn.Conv2d(1, encoder_output_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)) # 3 before
    model.cnet.conv1 = torch.nn.Conv2d(2, encoder_output_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)) # 6 before
    model.load_state_dict(weights,strict=True)
    
    return model.to(device)


class Sea_RAFT_loss:

    def __init__(self, device='cuda', iter=12):
        self.iter = iter

    def __call__(self,flow_preds, flow_gt, gamma=0.8,device='cpu'):
        """ Loss function defined over sequence of flow predictions """
        n_predictions = len(flow_preds)    
        flow_loss = 0.0

        # exclude invalid pixels and extremely large displacements
        valid = torch.ones_like(flow_gt[:, 0, :, :], dtype=torch.bool) # no valid mask for now, so all valid

        for i in range(n_predictions):
            i_weight = gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return flow_loss, metrics


def process_labels(flows, size=None,antialias=True):

    if size is None:
        size = SeaRAFTSettings.INPUT_SIZE    

    imgs = []
    
    for im in range(len(flows)):
        if isinstance(flows[im], list):
            image = torch.stack(flows[im], dim=0)
        else:
            image = flows[im]
        imgs.append(F.resize(image.float(), size=size, antialias=antialias))

    return torch.stack(imgs)


class OpticalFlowTransformSeaRAFT(nn.Module):
    def forward(self, img1, img2):
        if not isinstance(img1, Tensor):
            img1 = F.pil_to_tensor(img1)
        if not isinstance(img2, Tensor):
            img2 = F.pil_to_tensor(img2)

        img1 = F.convert_image_dtype(img1, torch.float)
        img2 = F.convert_image_dtype(img2, torch.float)

        # map [0, 1] into [-1, 1]
        img1 = F.normalize(img1, mean=[0.5], std=[0.5])
        img2 = F.normalize(img2, mean=[0.5], std=[0.5])

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        return img1, img2

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            "The images are rescaled to ``[-1.0, 1.0]``."
        )
        
def preprocess(ref_img, obj_img, transforms):
    if isinstance(ref_img, list):
        img1_batch = torch.stack(ref_img)
    else:
        img1_batch = ref_img
    if isinstance(obj_img, list):
        img2_batch = torch.stack(obj_img)
    else:
        img2_batch = obj_img

    img1_batch = F.resize(img1_batch, size=SeaRAFTSettings.INPUT_SIZE, antialias=True)
    img2_batch = F.resize(img2_batch, size=SeaRAFTSettings.INPUT_SIZE, antialias=True)

    return transforms(img1_batch.unsqueeze(1), img2_batch.unsqueeze(1))


if __name__ == "__main__":
    pass
