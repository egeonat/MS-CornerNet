import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.cpool import TopPool, BottomPool, LeftPool, RightPool


class pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(pool, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, 3, padding=1, bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        pool1 = self.pool1(self.p1_conv1(x))
        pool2 = self.pool2(self.p2_conv1(x))

        p_bn1 = self.p_bn1(self.p_conv1(pool1 + pool2))
        bn1 = self.bn1(self.conv1(x))

        out = self.conv2(F.relu(p_bn1 + bn1, inplace=True))
        return out


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = nn.Sequential(nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
                                  nn.BatchNorm2d(out_dim)) \
            if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


def make_layer(kernel_size, inp_dim, out_dim, modules, layer, stride=1):
    layers = [layer(kernel_size, inp_dim, out_dim, stride=stride)]
    layers += [layer(kernel_size, out_dim, out_dim) for _ in range(modules - 1)]
    return nn.Sequential(*layers)


def make_layer_revr(kernel_size, inp_dim, out_dim, modules, layer):
    layers = [layer(kernel_size, inp_dim, inp_dim) for _ in range(modules - 1)]
    layers.append(layer(kernel_size, inp_dim, out_dim))
    return nn.Sequential(*layers)


def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False),
                         nn.Conv2d(curr_dim, out_dim, (1, 1)))


class kp_module(nn.Module):
    def __init__(self, n, dims, modules):
        super(kp_module, self).__init__()

        self.n = n

        curr_modules = modules[0]
        next_modules = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.top = make_layer(3, curr_dim, curr_dim, curr_modules, layer=residual)
        self.down = nn.Sequential()
        self.low1 = make_layer(3, curr_dim, next_dim, curr_modules, layer=residual, stride=2)
        if self.n > 1:
            self.low2 = kp_module(n - 1, dims[1:], modules[1:])
        else:
            self.low2 = make_layer(3, next_dim, next_dim, next_modules, layer=residual)
        self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_modules, layer=residual)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1 = self.top(x)
        down = self.down(x)
        low1 = self.low1(down)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up(low3)
        return up1 + up2


class exkp(nn.Module):
    def __init__(self, n, nstack, dims, modules, num_classes=1, cnv_dim=256):
        super(exkp, self).__init__()

        # Fixing nstack for fusion architecture
        # self.nstack = nstack
        self.nstack = 2

        curr_dim = dims[0]

        self.pre_rgb = nn.Sequential(convolution(7, 3, 128, stride=2),
                                     residual(3, 128, curr_dim, stride=2))
        self.pre_ir = nn.Sequential(convolution(7, 1, 128, stride=2),
                                    residual(3, 128, curr_dim, stride=2))

        # Hourglass kp for rgb and ir
        self.kps_rgb = nn.ModuleList([kp_module(n, dims, modules)])
        self.kps_ir = nn.ModuleList([kp_module(n, dims, modules)])

        # Hourglass cnvs for rgb and ir
        self.cnvs_rgb = nn.ModuleList([convolution(3, curr_dim, cnv_dim)])
        self.cnvs_ir = nn.ModuleList([convolution(3, curr_dim, cnv_dim)])

        self.convs_rgb_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                                                       nn.BatchNorm2d(curr_dim))])
        self.convs_ir_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                                                      nn.BatchNorm2d(curr_dim))])
        self.inters_rgb_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                                                        nn.BatchNorm2d(curr_dim))])
        self.inters_ir_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                                                       nn.BatchNorm2d(curr_dim))])

        # Fusion convolution for multispectral
        self.fus = nn.ModuleList([convolution(3, curr_dim*2, curr_dim)])

        self.inters = nn.ModuleList([residual(3, curr_dim, curr_dim)])

        self.kp_ms = nn.ModuleList([kp_module(n, dims, modules)])
        self.cnvs_ms = nn.ModuleList([convolution(3, curr_dim, cnv_dim)])

        self.cnvs_tl = nn.ModuleList([pool(cnv_dim, TopPool, LeftPool) for _ in range(3)])
        self.cnvs_br = nn.ModuleList([pool(cnv_dim, BottomPool, RightPool) for _ in range(3)])

        # heatmap layers
        self.hmap_tl = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(3)])
        self.hmap_br = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(3)])

        # embedding layers
        self.embd_tl = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(3)])
        self.embd_br = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(3)])

        for hmap_tl, hmap_br in zip(self.hmap_tl, self.hmap_br):
            hmap_tl[-1].bias.data.fill_(-2.19)
            hmap_br[-1].bias.data.fill_(-2.19)

        # regression layers
        self.regs_tl = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(3)])
        self.regs_br = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(3)])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        img_rgb = inputs[0]
        img_ir = inputs[1]
        outs = []
        # Pre and HG for RGB
        img_rgb = self.pre_rgb(img_rgb)
        skip_rgb = img_rgb
        img_rgb = self.kps_rgb[0](img_rgb)
        img_rgb = self.cnvs_rgb[0](img_rgb)
        inter_rgb = img_rgb

        # Pre and HG for IR
        img_ir = self.pre_ir(img_ir)
        skip_ir = img_ir
        img_ir = self.kps_ir[0](img_ir)
        img_ir = self.cnvs_ir[0](img_ir)
        inter_ir = img_ir

        # Intermediate supervision if training
        if self.training:
            inter_rgb_tl = self.cnvs_tl[0](inter_rgb)
            inter_rgb_br = self.cnvs_br[0](inter_rgb)

            hmap_tl_rgb, hmap_br_rgb = self.hmap_tl[0](inter_rgb_tl), self.hmap_br[0](inter_rgb_br)
            embd_tl_rgb, embd_br_rgb = self.embd_tl[0](inter_rgb_tl), self.embd_br[0](inter_rgb_br)
            regs_tl_rgb, regs_br_rgb = self.regs_tl[0](inter_rgb_tl), self.regs_br[0](inter_rgb_br)

            outs.append([hmap_tl_rgb, hmap_br_rgb, embd_tl_rgb, embd_br_rgb, regs_tl_rgb, regs_br_rgb])

        if self.training:
            inter_ir_tl = self.cnvs_tl[1](inter_ir)
            inter_ir_br = self.cnvs_br[1](inter_ir)

            hmap_tl_ir, hmap_br_ir = self.hmap_tl[1](inter_ir_tl), self.hmap_br[1](inter_ir_br)
            embd_tl_ir, embd_br_ir = self.embd_tl[1](inter_ir_tl), self.embd_br[1](inter_ir_br)
            regs_tl_ir, regs_br_ir = self.regs_tl[1](inter_ir_tl), self.regs_br[1](inter_ir_br)

            outs.append([hmap_tl_ir, hmap_br_ir, embd_tl_ir, embd_br_ir, regs_tl_ir, regs_br_ir])

        # Inter for rgb
        img_rgb = self.inters_rgb_[0](skip_rgb) + self.convs_rgb_[0](img_rgb)
        img_rgb = self.relu(img_rgb)

        # Inter for ir
        img_ir = self.inters_ir_[0](skip_ir) + self.convs_ir_[0](img_ir)
        img_ir = self.relu(img_ir)

        # Merge RGB and IR
        img_ms = torch.cat((img_rgb, img_ir), 1)

        # Apply conv to reduce dim
        img_ms = self.fus[0](img_ms)
        img_ms = self.inters[0](img_ms)

        # Multispectral hourglass
        img_ms = self.kp_ms[0](img_ms)
        img_ms = self.cnvs_ms[0](img_ms)

        # Final predictions
        final_tl = self.cnvs_tl[2](img_ms)
        final_br = self.cnvs_br[2](img_ms)

        hmap_tl, hmap_br = self.hmap_tl[2](final_tl), self.hmap_br[2](final_br)
        embd_tl, embd_br = self.embd_tl[2](final_tl), self.embd_br[2](final_br)
        regs_tl, regs_br = self.regs_tl[2](final_tl), self.regs_br[2](final_br)

        outs.append([hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br])

        return outs


# tiny hourglass is for only debug
get_hourglass = \
    {'large_hourglass':
         exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4]),
     'small_hourglass':
         exkp(n=5, nstack=1, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4]),
     'tiny_hourglass':
         exkp(n=5, nstack=1, dims=[256, 128, 256, 256, 256, 384], modules=[2, 2, 2, 2, 2, 4])}
