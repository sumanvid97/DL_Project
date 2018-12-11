import torch
from torch import nn
from torchvision import models, transforms



class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.vgg = nn.Sequential(*(list(vgg.features.children())[:36])).eval()
        self.vgg = nn.DataParallel(self.vgg, device_ids=[0, 1])

        self.mse = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        return self.mse(self.vgg(input), self.vgg(target).detach())


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LRTransformTest(object):
    def __init__(self, downscale_factor):
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.downscale_factor = downscale_factor

    def __call__(self, hr_tensor):
        hr_img = self.to_pil(hr_tensor)
        w, h = hr_img.size
        lr_scale = transforms.Resize(int(min(w, h) / self.downscale_factor), interpolation=3)
        hr_scale = transforms.Resize(min(w, h), interpolation=3)
        lr_img = lr_scale(hr_img)
        hr_interpolated_img = hr_scale(lr_img)
        return self.to_tensor(lr_img), self.to_tensor(hr_interpolated_img)


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        return (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        # return (((x[:, :, :-1, :] - x[:, :, 1:, :]) ** 2 + (x[:, :, :, :-1] - x[:, :, :, 1:]) ** 2) ** 1.25).mean()
