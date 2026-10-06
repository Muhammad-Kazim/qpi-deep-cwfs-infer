# pm26
import argparse
import torch 
from torchvision.models.optical_flow import raft_large, raft_small

from qpi_deep_cwfs import hRAFT, hSea_RAFT
from qpi_deep_cwfs.sea_raft.core.raft import RAFT as Sea_RAFT


def get_model_and_transforms(model_name: str, device: str, checkpoint: str = None,args=None) -> torch.nn.Module:

    if model_name.lower() == 'raft_large' or model_name.lower() == 'raft_small':
        if model_name == 'raft_large':
            model = raft_large(progress=False)
        else:
            model = raft_small(progress=False)

        model = hRAFT.init_model_RAFT(model, device=device, checkpoint=checkpoint,model_name=model_name)
        transforms = hRAFT.OpticalFlowTransformRAFT()
        model_settings = dict()

    elif model_name.lower().__contains__('sea_raft'):

        class SeaArgs(argparse.Namespace):
            def __init__(self, model_name):
                self.name = model_name
                self.gma = False      
                self.restore_ckpt = None
                self.image_size = hSea_RAFT.SeaRAFTSettings.INPUT_SIZE

                self.use_var = True,
                self.var_min = 0
                self.var_max = 10
                self.initial_dim = 64
                self.radius = 4

                self.num_heads = 4    
                self.hidden_dim = 128
                self.context_dim = 128
                self.dropout = 0.0
                self.alternate_corr = False
                self.mixed_precision = False
                self.num_blocks = 2

                if model_name.lower() == 'sea_raft_large':
                    raise ValueError('not available yet')
                    self.dim = 64              
                    self.num_heads = 8          
                    self.block_dims = [128, 256, 512] 
                    self.pretrain = 'resnet34' # experimental
                elif model_name.lower() == 'sea_raft_medium':
                    self.pretrain = 'resnet34'
                    self.block_dims = [64, 128, 256]
                    self.dim = 128   
                    self.iters = 4 #  better use 12 !      
                elif model_name.lower() == 'sea_raft_small':
                    self.pretrain = 'resnet34'
                    self.block_dims = [64,128,256]
                    self.dim = 128   
                    self.iters = 4      
                else:
                    raise ValueError(f'Model name {model_name} not recognized for SeaRAFT.')

        sea_args = SeaArgs(model_name)
        model = Sea_RAFT(sea_args)
        model = hSea_RAFT.init_model_Sea_RAFT(model=model, device=device, checkpoint=checkpoint, model_name=model_name)
        transforms = hSea_RAFT.OpticalFlowTransformSeaRAFT()
        model_settings = dict(test_mode=True) # faster... no nf computation


    elif model_name == 'gmflow':
        upsample_factor = 4 # FIXME: upsample_factor=2
        model = UniMatch(num_scales=2,
                         feature_channels=128, 
                         upsample_factor=upsample_factor,
                         num_head=1,
                         ffn_dim_expansion=4,
                         num_transformer_layers=6,
                         reg_refine=True, #False,
                         task='flow',
                         input_dtype='float') # needed for scaling in model.forward()

        attn_type = 'swin'
        if model.num_scales == 2:
            attn_splits_list = [4,4]#[8,8]#[16,16]#[8,8]
            corr_radius_list = [2,2] #[4, 4]
            prop_radius_list = [2, 2] # [1,1]
        else:
            raise ValueError('num_scales other than 2 not implemented yet for GMFlow in this script.')

        model_settings = dict(attn_type=attn_type,
                                  attn_splits_list=attn_splits_list,
                                  corr_radius_list=corr_radius_list,
                                  prop_radius_list=prop_radius_list
                            )

        model = hGMFlow.init_model_gmflow(model, device=device, checkpoint=checkpoint)
        transforms = hGMFlow.OpticalFlowTransformGMFlow()

    
    elif model_name.lower() == 'gma':
        # from helper.gma.GMA.core.train args => used in model.
        args.mixed_precision = False # True
        args.dropout=0.0
        args.num_heads = 1
        args.position_only = False 
        args.position_and_content = False
        args.hidden_dim = 128
        args.context_dim = 128
        args.cnet_relation = False 
        model = RAFTGMA(args=args)
        model = hGMA.init_model_GMA(model=model, device=device, checkpoint=checkpoint,model_name=model_name)
        transforms = hGMA.OpticalFlowTransformGMA()
        model_settings = dict()

    else:
        raise NotImplementedError(f'Model {model_name} not implemented yet.')
    
    return model, transforms, model_settings


def grad_stitch_n(grad_list, overlap):
    grad_0s, grad_1s = grad_list
    n = int((len(grad_0s))**0.5)
    dimX, dimY = grad_0s[0].shape

    stride_x = dimX - overlap
    stride_y = dimY - overlap

    H = n * dimX - (n - 1) * overlap
    W = n * dimY - (n - 1) * overlap

    grad0_tot = torch.zeros((H, W), dtype=grad_0s[0].dtype)
    grad1_tot = torch.zeros((H, W), dtype=grad_1s[0].dtype)
    weight = torch.zeros((H, W), dtype=grad_0s[0].dtype)

    for i in range(n):
        for j in range(n):
            y0 = i * stride_x
            y1 = y0 + dimX
            x0 = j * stride_y
            x1 = x0 + dimY

            idx = j + n * i

            grad0_tot[y0:y1, x0:x1] += grad_0s[idx]
            grad1_tot[y0:y1, x0:x1] += grad_1s[idx]
            weight[y0:y1, x0:x1] += 1.0

    # avoid division by zero
    weight[weight == 0] = 1.0

    grad0_tot = grad0_tot / weight
    grad1_tot = grad1_tot / weight

    return grad0_tot, grad1_tot


def make_crop_array(image_size, patch_size, overlap):
    H, W = image_size
    stride = patch_size - overlap

    rows = torch.arange(0, H, stride)
    cols = torch.arange(0, W, stride)

    row, col = torch.meshgrid(rows, cols, indexing="ij")

    return torch.stack((row.flatten(), col.flatten()), dim=1)


def integrate_flow_field_torch(flow, n_iter=2500, tol=1e-15):
    """ faster, conjugate gradient based solver ... Ax=b"""
    if flow.ndim == 3:
        flow = flow.permute(2, 0, 1).unsqueeze(0)
    
    orig_dtype = flow.dtype
    flow = flow.double() 
    
    device = flow.device
    B, C, H, W = flow.shape

    Ix = flow[:, 0:1, :, :]
    Iy = flow[:, 1:2, :, :]

    dIx_inner = Ix[:, :, :, 1:] - Ix[:, :, :, :-1]
    dIx_last = dIx_inner[:, :, :, -1:]
    dIx = torch.cat([dIx_inner, dIx_last], dim=3)
    dIy_inner = Iy[:, :, 1:, :] - Iy[:, :, :-1, :]
    dIy_last = dIy_inner[:, :, -1:, :]
    dIy = torch.cat([dIy_inner, dIy_last], dim=2)

    rhs = dIx + dIy
    b = -rhs 

    laplace_kernel = torch.tensor([[0, -1, 0], 
                                   [-1, 4, -1], 
                                   [0, -1, 0]], 
                                  device=device, dtype=torch.float64).view(1, 1, 3, 3)

    mask = torch.ones((1, 1, H, W), device=device, dtype=torch.float64)
    mask[:, :, [0, -1], :] = 0
    mask[:, :, :, [0, -1]] = 0

    def A_op(x):
        return torch.nn.functional.conv2d(x, laplace_kernel, padding=1) * mask

    x = torch.zeros_like(b)
    r = b - A_op(x)
    r = r * mask
    p = r.clone()
    rsold = torch.sum(r * r, dim=(1, 2, 3))

    for i in range(n_iter):
        Ap = A_op(p)        
        pAp = torch.sum(p * Ap, dim=(1, 2, 3))
        alpha = rsold / (pAp + 1e-20)
        alpha = alpha.view(B, 1, 1, 1)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        rsnew = torch.sum(r * r, dim=(1, 2, 3))
        
        max_err = torch.max(torch.sqrt(rsnew))
        if max_err < tol:
            break
            
        beta = rsnew / (rsold + 1e-20)
        beta = beta.view(B, 1, 1, 1)
        
        p = r + beta * p
        rsold = rsnew
        
    if orig_dtype == torch.float32:
        return x.float().squeeze()
        
    return x.squeeze()