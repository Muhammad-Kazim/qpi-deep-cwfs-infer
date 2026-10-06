import numpy as np
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from corr import CorrBlock
from extractor import ResNetFPN
from sea_raft_layer import conv1x1, conv3x3, ConvNextBlock

from huggingface_hub import PyTorchModelHubMixin


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=4):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class BasicMotionEncoder(nn.Module):
    def __init__(self, args, dim=128):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_channel
        self.convc1 = nn.Conv2d(cor_planes, dim*2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim*2, dim+dim//2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim//2, 3, padding=1)
        self.conv = nn.Conv2d(dim*2, dim-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)
    
class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hdim=128, cdim=128):
        #net: hdim, inp: cdim
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, dim=cdim)
        self.refine = []
        for i in range(args.num_blocks):
            self.refine.append(ConvNextBlock(2*cdim+hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        return net


class RAFT(
    nn.Module,
    PyTorchModelHubMixin, 
    # optionally, you can add metadata which gets pushed to the model card
    repo_url="https://github.com/princeton-vl/SEA-RAFT",
    pipeline_tag="optical-flow-estimation",
    license="bsd-3-clause",
):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.output_dim = args.dim * 2
        
        self.args.corr_levels = 4
        self.args.corr_radius = args.radius
        self.args.corr_channel = args.corr_levels * (args.radius * 2 + 1) ** 2
        self.cnet = ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=True)

        # conv for iter 0 results
        self.init_conv = conv3x3(2 * args.dim, 2 * args.dim)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, 6, 3, padding=1)
        )
        if args.iters > 0:
            self.fnet = ResNetFPN(args, input_dim=3, output_dim=self.output_dim, norm_layer=nn.BatchNorm2d, init_weight=True)
            self.update_block = BasicUpdateBlock(args, hdim=args.dim, cdim=args.dim)

            self.iters = args.iters # pm26
    
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords2 - coords1"""
        N, C, H, W = img.shape
        coords1 = coords_grid(N, H//8, W//8, device=img.device)
        coords2 = coords_grid(N, H//8, W//8, device=img.device)
        return coords1, coords2

    def upsample_data(self, flow, info, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 8*H, 8*W), up_info.reshape(N, C, 8*H, 8*W)

    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        N, _, H, W = image1.shape
        if iters is None:
            iters = self.args.iters
        if flow_gt is None:
            flow_gt = torch.zeros(N, 2, H, W, device=image1.device)

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        flow_predictions = []
        info_predictions = []

        # padding
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        N, _, H, W = image1.shape
        dilation = torch.ones(N, 1, H//8, W//8, device=image1.device)
        # run the context network
        cnet = self.cnet(torch.cat([image1, image2], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        # init flow
        flow_update = self.flow_head(net)
        weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)
            
        if self.args.iters > 0:
            # run the feature network
            fmap1_8x = self.fnet(image1)
            fmap2_8x = self.fnet(image2)
            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.args)

        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (coords_grid(N, H, W, device=image1.device) + flow_8x).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        if test_mode == False:
            # exlude invalid pixels and extremely large diplacements
            nf_predictions = []
            for i in range(len(info_predictions)):
                if not self.args.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.args.var_max
                    var_min = self.args.var_min
                    
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                # Large b Component                
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                # Small b Component
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                # term2: [N, 2, m, H, W]
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                # term1: [N, m, H, W]
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)

            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': None}