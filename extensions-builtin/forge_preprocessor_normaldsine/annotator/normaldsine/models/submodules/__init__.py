import torch
import torch.nn as nn
import torch.nn.functional as F
import os


INPUT_CHANNELS_DICT = {
    0: [1280, 112, 40, 24, 16],
    1: [1280, 112, 40, 24, 16],
    2: [1408, 120, 48, 24, 16],
    3: [1536, 136, 48, 32, 24],
    4: [1792, 160, 56, 32, 24],
    5: [2048, 176, 64, 40, 24],
    6: [2304, 200, 72, 40, 32],
    7: [2560, 224, 80, 48, 32]
}


class Encoder(nn.Module):
    def __init__(self, B=5, pretrained=True):
        super(Encoder, self).__init__()

        basemodel_name = 'tf_efficientnet_b%s_ap' % B
        print('Loading base model ()...'.format(basemodel_name), end='')
        repo_path = os.path.join(os.path.dirname(__file__), 'efficientnet_repo')
        basemodel = torch.hub.load(repo_path, basemodel_name, pretrained=False, source='local')
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, ks=3):
        super(ConvGRU, self).__init__()
        p = (ks - 1) // 2
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, ks, padding=p)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, ks, padding=p)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, ks, padding=p)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h
    

class RayReLU(nn.Module):
    def __init__(self, eps=1e-2):
        super(RayReLU, self).__init__()
        self.eps = eps

    def forward(self, pred_norm, ray):
        # angle between the predicted normal and ray direction
        cos = torch.cosine_similarity(pred_norm, ray, dim=1).unsqueeze(1) # (B, 1, H, W)

        # component of pred_norm along view
        norm_along_view = ray * cos

        # cos should be bigger than eps
        norm_along_view_relu = ray * (torch.relu(cos - self.eps) + self.eps)

        # difference        
        diff = norm_along_view_relu - norm_along_view

        # updated pred_norm
        new_pred_norm = pred_norm + diff
        new_pred_norm = F.normalize(new_pred_norm, dim=1)

        return new_pred_norm    


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features, align_corners=True):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())
        self.align_corners = align_corners

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=self.align_corners)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class Conv2d_WS(nn.Conv2d):
    """ weight standardization
    """ 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class UpSampleGN(nn.Module):
    """ UpSample with GroupNorm
    """
    def __init__(self, skip_input, output_features, align_corners=True):
        super(UpSampleGN, self).__init__()
        self._net = nn.Sequential(Conv2d_WS(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(8, output_features),
                                  nn.LeakyReLU(),
                                  Conv2d_WS(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(8, output_features),
                                  nn.LeakyReLU())
        self.align_corners = align_corners

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=self.align_corners)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


def upsample_via_bilinear(out, up_mask, downsample_ratio):
    """ bilinear upsampling (up_mask is a dummy variable)
    """
    return F.interpolate(out, scale_factor=downsample_ratio, mode='bilinear', align_corners=True)


def upsample_via_mask(out, up_mask, downsample_ratio):
    """ convex upsampling
    """
    # out: low-resolution output (B, o_dim, H, W)
    # up_mask: (B, 9*k*k, H, W)
    k = downsample_ratio

    N, o_dim, H, W = out.shape
    up_mask = up_mask.view(N, 1, 9, k, k, H, W)
    up_mask = torch.softmax(up_mask, dim=2)             # (B, 1, 9, k, k, H, W)

    up_out = F.unfold(out, [3, 3], padding=1)       # (B, 2, H, W) -> (B, 2 X 3*3, H*W)
    up_out = up_out.view(N, o_dim, 9, 1, 1, H, W)   # (B, 2, 3*3, 1, 1, H, W)
    up_out = torch.sum(up_mask * up_out, dim=2)     # (B, 2, k, k, H, W)

    up_out = up_out.permute(0, 1, 4, 2, 5, 3)       # (B, 2, H, k, W, k)
    return up_out.reshape(N, o_dim, k*H, k*W)   # (B, 2, kH, kW)


def convex_upsampling(out, up_mask, k):
    # out: low-resolution output    (B, C, H, W)
    # up_mask:                      (B, 9*k*k, H, W)
    B, C, H, W = out.shape
    up_mask = up_mask.view(B, 1, 9, k, k, H, W)
    up_mask = torch.softmax(up_mask, dim=2)             # (B, 1, 9, k, k, H, W)

    out = F.pad(out, pad=(1,1,1,1), mode='replicate')
    up_out = F.unfold(out, [3, 3], padding=0)           # (B, C, H, W) -> (B, C X 3*3, H*W)
    up_out = up_out.view(B, C, 9, 1, 1, H, W)           # (B, C, 9, 1, 1, H, W)

    up_out = torch.sum(up_mask * up_out, dim=2)         # (B, C, k, k, H, W)
    up_out = up_out.permute(0, 1, 4, 2, 5, 3)           # (B, C, H, k, W, k)
    return up_out.reshape(B, C, k*H, k*W)               # (B, C, kH, kW)


def get_unfold(pred_norm, ps, pad):
    B, C, H, W = pred_norm.shape
    pred_norm = F.pad(pred_norm, pad=(pad,pad,pad,pad), mode='replicate')       # (B, C, h, w)
    pred_norm_unfold = F.unfold(pred_norm, [ps, ps], padding=0)                 # (B, C X ps*ps, h*w)
    pred_norm_unfold = pred_norm_unfold.view(B, C, ps*ps, H, W)                 # (B, C, ps*ps, h, w)
    return pred_norm_unfold


def get_prediction_head(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Conv2d(input_dim, hidden_dim, 3, padding=1), 
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, 1), 
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, output_dim, 1),
    )
