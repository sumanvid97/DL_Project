# coding: utf-8
import os
import time
import torch
import numpy as np
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from misc import PerceptualLoss, AvgMeter, LRTransformTest, TotalVariationLoss
from models import Generator, Discriminator

train_args = {
    'train_batch_size': 64,
    'hr_size': 64,  # make sure that hr_size can be divided by scale_factor exactly
    'scale_factor': 4,  # should be power of 2
    'g_snapshot': 'epoch_40_set14_g_mse_loss_0.00320_psnr_25.87172_g.pth',
    'd_snapshot': 'epoch_40_set14_g_mse_loss_0.00320_psnr_25.87172_d.pth',
    'g_lr': 1e-4,
    'd_lr': 1e-4,
    'train_set_path': './datasets/VOC2012',
    'set5_path': './datasets/Set5',
    'set14_path': './datasets/Set14',
    'start_epoch': 1,
    'epoch_num': 40,
    'ckpt_path': './checkpoint',
    'c': 0.01
}

g_pretrain_args = {
    'pretrain': True,
    'epoch_num': 100,
    'lr': 1e-4,
}

writer = SummaryWriter()

train_ori_transform = transforms.Compose([
    transforms.RandomCrop(train_args['hr_size']),
    transforms.ToTensor()
])
train_lr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(int(train_args['hr_size'] / train_args['scale_factor']), interpolation=3),
    transforms.ToTensor()
])
val_ori_transform = transforms.ToTensor()
val_lr_transform = LRTransformTest(train_args['scale_factor'])
val_display_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(400),
    transforms.CenterCrop(400),
    transforms.ToTensor()
])

train_set = datasets.ImageFolder(train_args['train_set_path'], train_ori_transform)
train_loader = DataLoader(train_set, batch_size=train_args['train_batch_size'], shuffle=True, num_workers=12,
                          pin_memory=True)

set5 = datasets.ImageFolder(train_args['set5_path'], val_ori_transform)
set14 = datasets.ImageFolder(train_args['set14_path'], val_ori_transform)
set5_loader = DataLoader(set5, batch_size=1, num_workers=12, pin_memory=True)
set14_loader = DataLoader(set14, batch_size=1, num_workers=12, pin_memory=True)
val_loader = {'set5': set5_loader, 'set14': set14_loader}


def train():
    g = Generator(scale_factor=train_args['scale_factor']).cuda().train()
    g = nn.DataParallel(g, device_ids=[0, 1])
    if len(train_args['g_snapshot']) > 0:
        print ('load generator snapshot ' + train_args['g_snapshot'])
        g.load_state_dict(torch.load(os.path.join(train_args['ckpt_path'], train_args['g_snapshot'])))

    mse_criterion = nn.MSELoss().cuda()
    tv_criterion = TotalVariationLoss().cuda()
    g_mse_loss_record, g_tv_loss_record, g_loss_record, psnr_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    iter_nums = len(train_loader)

    if g_pretrain_args['pretrain']:
        g_optimizer = optim.Adam(g.parameters(), lr=g_pretrain_args['lr'])
        scheduler = optim.lr_scheduler.MultiStepLR(g_optimizer, milestones=[10,20,30,40,50], gamma=0.5)
        for epoch in range(g_pretrain_args['epoch_num']):
            scheduler.step()
            start = time.time()

            for i, data in enumerate(train_loader):
                hr_imgs, _ = data
                batch_size = hr_imgs.size(0)
                lr_imgs = Variable(torch.stack([train_lr_transform(img) for img in hr_imgs], 0)).cuda()
                hr_imgs = Variable(hr_imgs).cuda()

                g.zero_grad()
                gen_hr_imgs = g(lr_imgs)

                g_mse_loss = mse_criterion(gen_hr_imgs, hr_imgs)
                # g_tv_loss = tv_criterion(gen_hr_imgs)
                g_tv_loss = 0
                g_loss = g_mse_loss + 2e-8 * g_tv_loss
                g_loss.backward()
                g_optimizer.step()

                g_mse_loss_record.update(g_mse_loss.item(), batch_size)
                # g_tv_loss_record.update(g_tv_loss.item(), batch_size)
                g_loss_record.update(g_loss.item(), batch_size)
                psnr_record.update(10 * np.log10(1 / g_mse_loss.item()), batch_size)

                print ('[pretrain]: [epoch %d], [iter %d / %d], [loss %.5f], [psnr %.5f]' % (
                    epoch + 1, i + 1, iter_nums, g_loss_record.avg, psnr_record.avg))

                writer.add_scalar('pretrain_g_loss', g_loss_record.avg, epoch * iter_nums + i + 1)
                writer.add_scalar('pretrain_psnr', psnr_record.avg, epoch * iter_nums + i + 1)

            torch.save(g.state_dict(), os.path.join(
                train_args['ckpt_path'], 'pretrain_g_epoch_%d_loss_%.5f_psnr_%.5f.pth' % (
                    epoch + 1, g_loss_record.avg, psnr_record.avg)))

            end = time.time()

            print ('[time for last epoch: %.5f] [pretrain]: [epoch %d], [iter %d / %d], [loss %.5f], [psnr %.5f]' % (
                    end - start, epoch + 1, i + 1, iter_nums, g_loss_record.avg, psnr_record.avg))

            g_mse_loss_record.reset()
            psnr_record.reset()

            validate(g, epoch)

    d = Discriminator().cuda().train()
    d = nn.DataParallel(d, device_ids=[0, 1])
    if len(train_args['d_snapshot']) > 0:
        print ('load discriminator snapshot ' + train_args['d_snapshot'])
        d.load_state_dict(torch.load(os.path.join(train_args['ckpt_path'], train_args['d_snapshot'])))

    g_optimizer = optim.Adam(g.parameters(), lr=train_args['g_lr'])
    d_optimizer = optim.Adam(d.parameters(), lr=train_args['d_lr'])
    g_scheduler = optim.lr_scheduler.MultiStepLR(g_optimizer, milestones=[10,20,30,40], gamma=0.5)
    d_scheduler = optim.lr_scheduler.MultiStepLR(g_optimizer, milestones=[10,20,30,40], gamma=0.5)
    perceptual_criterion, tv_criterion = PerceptualLoss().cuda(), TotalVariationLoss().cuda()

    g_mse_loss_record, g_perceptual_loss_record, g_tv_loss_record = AvgMeter(), AvgMeter(), AvgMeter()
    psnr_record, g_ad_loss_record, g_loss_record, d_loss_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    for epoch in range(train_args['start_epoch'] - 1, train_args['epoch_num']):
        g_scheduler.step()
        d_scheduler.step()
        start = time.time()

        for i, data in enumerate(train_loader):
            hr_imgs, _ = data
            batch_size = hr_imgs.size(0)
            lr_imgs = Variable(torch.stack([train_lr_transform(img) for img in hr_imgs], 0)).cuda()
            hr_imgs = Variable(hr_imgs).cuda()
            gen_hr_imgs = g(lr_imgs)

            # update d
            d.zero_grad()

            # gen_hr_imgs.detach() because we don't want to update the gradients for g when d is being updated
            # d_ad_loss = - torch.log10(1 - d(gen_hr_imgs.detach())).mean() - torch.log10(d(hr_imgs)).mean() 
            d_ad_loss = d(gen_hr_imgs.detach()).mean() - d(hr_imgs).mean()
            d_ad_loss.backward()
            d_optimizer.step()

            d_loss_record.update(d_ad_loss.item(), batch_size)

            for p in d.parameters():
                p.data.clamp_(-train_args['c'], train_args['c'])

            # update g
            g.zero_grad()
            g_mse_loss = mse_criterion(gen_hr_imgs, hr_imgs)
            g_perceptual_loss = perceptual_criterion(gen_hr_imgs, hr_imgs)
            g_tv_loss = tv_criterion(gen_hr_imgs)
            # g_ad_loss = -torch.log10(d(gen_hr_imgs)).mean()
            g_ad_loss = -d(gen_hr_imgs).mean()
            g_loss = g_mse_loss + 0.006 * g_perceptual_loss + 0.001 * g_ad_loss + 2e-8 * g_tv_loss
            g_loss.backward()
            g_optimizer.step()

            g_mse_loss_record.update(g_mse_loss.item(), batch_size)
            g_perceptual_loss_record.update(g_perceptual_loss.item(), batch_size)
            g_tv_loss_record.update(g_tv_loss.item(), batch_size)
            psnr_record.update(10 * np.log10(1 / g_mse_loss.item()), batch_size)
            g_ad_loss_record.update(g_ad_loss.item(), batch_size)
            g_loss_record.update(g_loss.item(), batch_size)

            print ('[train]: [epoch %d], [iter %d / %d], [d_ad_loss %.5f], [g_ad_loss %.5f], [psnr %.5f], ' \
                  '[g_mse_loss %.5f], [g_perceptual_loss %.5f], [g_tv_loss %.5f] [g_loss %.5f]' % \
                  (epoch + 1, i + 1, iter_nums, d_loss_record.avg, g_ad_loss_record.avg, psnr_record.avg,
                   g_mse_loss_record.avg, g_perceptual_loss_record.avg, g_tv_loss_record.avg, g_loss_record.avg))

            writer.add_scalar('d_loss', d_loss_record.avg, epoch * iter_nums + i + 1)
            writer.add_scalar('g_mse_loss', g_mse_loss_record.avg, epoch * iter_nums + i + 1)
            writer.add_scalar('g_perceptual_loss', g_perceptual_loss_record.avg, epoch * iter_nums + i + 1)
            writer.add_scalar('g_tv_loss', g_tv_loss_record.avg, epoch * iter_nums + i + 1)
            writer.add_scalar('psnr', psnr_record.avg, epoch * iter_nums + i + 1)
            writer.add_scalar('g_ad_loss', g_ad_loss_record.avg, epoch * iter_nums + i + 1)
            writer.add_scalar('g_loss', g_loss_record.avg, epoch * iter_nums + i + 1)

        end = time.time()

        print ('[time for last epoch: %.5f][train]: [epoch %d], [iter %d / %d], [d_ad_loss %.5f], [g_ad_loss %.5f], [psnr %.5f], ' \
              '[g_mse_loss %.5f], [g_perceptual_loss %.5f], [g_tv_loss %.5f] [g_loss %.5f]' % \
              (end - start, epoch + 1, i + 1, iter_nums, d_loss_record.avg, g_ad_loss_record.avg, psnr_record.avg,
               g_mse_loss_record.avg, g_perceptual_loss_record.avg, g_tv_loss_record.avg, g_loss_record.avg))

        d_loss_record.reset()
        g_mse_loss_record.reset()
        g_perceptual_loss_record.reset()
        g_tv_loss_record.reset()
        psnr_record.reset()
        g_ad_loss_record.reset()
        g_loss_record.reset()

        validate(g, epoch, d)


def validate(g, curr_epoch, d=None):
    g.eval()

    mse_criterion = nn.MSELoss()
    g_mse_loss_record, psnr_record = AvgMeter(), AvgMeter()

    for name, loader in val_loader.items():

        val_visual = []
        # note that the batch size is 1
        for i, data in enumerate(loader):
            hr_img, _ = data

            lr_img, hr_interpolated_img = val_lr_transform(hr_img.squeeze(0))

            lr_img = Variable(lr_img.unsqueeze(0), volatile=True).cuda()
            hr_interpolated_img = hr_interpolated_img
            hr_img = Variable(hr_img, volatile=True).cuda()

            gen_hr_img = g(lr_img)

            g_mse_loss = mse_criterion(gen_hr_img, hr_img)

            g_mse_loss_record.update(g_mse_loss.item())
            psnr_record.update(10 * np.log10(1 / g_mse_loss.item()))

            val_visual.extend([val_display_transform(hr_interpolated_img),
                               val_display_transform(hr_img.cpu().data.squeeze(0)),
                               val_display_transform(gen_hr_img.cpu().data.squeeze(0))])

        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)

        snapshot_name = 'epoch_%d_%s_g_mse_loss_%.5f_psnr_%.5f' % (
            curr_epoch + 1, name, g_mse_loss_record.avg, psnr_record.avg)

        if d is None:
            snapshot_name = 'pretrain_' + snapshot_name
            writer.add_scalar('pretrain_validate_%s_psnr' % name, psnr_record.avg, curr_epoch + 1)
            writer.add_scalar('pretrain_validate_%s_g_mse_loss' % name, g_mse_loss_record.avg, curr_epoch + 1)

            print ('[pretrain validate %s]: [epoch %d], [g_mse_loss %.5f], [psnr %.5f]' % (
                name, curr_epoch + 1, g_mse_loss_record.avg, psnr_record.avg))
        else:
            writer.add_scalar('validate_%s_psnr' % name, psnr_record.avg, curr_epoch + 1)
            writer.add_scalar('validate_%s_g_mse_loss' % name, g_mse_loss_record.avg, curr_epoch + 1)

            print ('[validate %s]: [epoch %d], [g_mse_loss %.5f], [psnr %.5f]' % (
                name, curr_epoch + 1, g_mse_loss_record.avg, psnr_record.avg))

            torch.save(d.state_dict(), os.path.join(train_args['ckpt_path'], snapshot_name + '_d.pth'))

        torch.save(g.state_dict(), os.path.join(train_args['ckpt_path'], snapshot_name + '_g.pth'))

        writer.add_image(snapshot_name, val_visual)

        g_mse_loss_record.reset()
        psnr_record.reset()

    g.train()


if __name__ == '__main__':
    train()
