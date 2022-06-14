import os

import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from mask import Masker
from modules import *
from utils import save_weights, print_msg, inverse_normalize


def train(model, discriminator, train_config, data_config, train_dataset, val_dataset, save_path, estimator, device, logger=None):
    recon_optimizer = initialize_optimizer(train_config, model)
    discr_optimizer = initialize_optimizer(train_config, discriminator)
    weighted_sampler = initialize_sampler(data_config, train_dataset)
    lr_scheduler, warmup_scheduler = initialize_lr_scheduler(train_config, recon_optimizer)
    loss_function, loss_weight_scheduler = initialize_loss(train_config, train_dataset, device)
    train_loader, val_loader = initialize_dataloader(train_config, train_dataset, val_dataset, weighted_sampler)

    adv_loss_function = nn.MSELoss()
    masker = Masker(width=train_config['grid_size'], pixel_size=train_config['patch_size'], mode='random')

    # start training
    model.train()
    min_indicator = 999
    for epoch in range(1, train_config['epochs'] + 1):
        # resampling weight update
        if weighted_sampler:
            weighted_sampler.step()

        # update loss weights
        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step().to(device)
            loss_function.weight = weight

        # warmup scheduler update
        if warmup_scheduler and not warmup_scheduler.is_finish():
            warmup_scheduler.step()

        epoch_dis_loss = 0
        epoch_adv_loss = 0
        epoch_recon_loss = 0
        estimator.reset()
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            X, fov_mask = train_data
            X, fov_mask = X.to(device), fov_mask.to(device)
            noisy, mask = masker.mask(X, step, mode='training')
            mask = mask * fov_mask

            # forward
            recon_x = model(noisy)
            true_pred = discriminator(X.detach()).squeeze(1)
            recon_pred = discriminator(recon_x.detach()).squeeze(1)

            true_y = torch.ones_like(true_pred).float()
            fake_y = torch.zeros_like(recon_pred).float()

            # Train discriminator
            true_loss = adv_loss_function(true_pred, true_y)
            fake_loss = adv_loss_function(recon_pred, fake_y)
            discr_loss = (true_loss + fake_loss) / 2

            discr_optimizer.zero_grad()
            discr_loss.backward()
            discr_optimizer.step()

            # Train reconstruction
            recon_pred = discriminator(recon_x).squeeze(1)
            adv_loss = adv_loss_function(recon_pred, true_y)
            recon_loss = loss_function(recon_x * mask, X.detach() * mask)

            gen_loss = recon_loss + adv_loss * 0.1

            # backward
            recon_optimizer.zero_grad()
            gen_loss.backward()
            recon_optimizer.step()

            # metrics
            epoch_dis_loss += discr_loss.item()
            epoch_adv_loss += adv_loss.item()
            epoch_recon_loss += recon_loss.item()
            avg_dis_loss = epoch_dis_loss / (step + 1)
            avg_adv_loss = epoch_adv_loss / (step + 1)
            avg_recon_loss = epoch_recon_loss / (step + 1)

            # visualize samples
            if train_config['sample_view'] and step % train_config['sample_view_interval'] == 0:
                samples = torchvision.utils.make_grid(X.detach())
                samples = inverse_normalize(samples, data_config['mean'], data_config['std'])
                logger.add_image('input samples', samples, epoch, dataformats='CHW')

                samples = torchvision.utils.make_grid(noisy.detach())
                samples = inverse_normalize(samples, data_config['mean'], data_config['std'])
                logger.add_image('noisy', samples, epoch, dataformats='CHW')

                samples = torchvision.utils.make_grid(recon_x.detach())
                samples = inverse_normalize(samples, data_config['mean'], data_config['std'])
                logger.add_image('reconstructed samples', samples, epoch, dataformats='CHW')

            progress.set_description(
                'epoch: [{} / {}], recon_loss: {:.6f}, adv_loss: {:.6f}, dis_loss: {:.6f}'
                .format(epoch, train_config['epochs'], avg_recon_loss, avg_adv_loss, avg_dis_loss)
            )

        # validation performance
        if epoch % train_config['eval_interval'] == 0:
            val_loss = eval(model, val_loader, loss_function, estimator, device, masker)
            print('validation loss: {}'.format(val_loss))
            if logger:
                logger.add_scalar('validation loss', val_loss, epoch)

        # save model
        indicator = val_loss
        if indicator < min_indicator:
            save_weights(model, os.path.join(save_path, 'best_validation_weights.pt'))
            min_indicator = indicator
            print_msg('Best in validation set. Model save at {}'.format(save_path))

        if epoch % train_config['save_interval'] == 0:
            save_weights(model, os.path.join(save_path, 'epoch_{}.pt'.format(epoch)))

        # update learning rate
        curr_lr = recon_optimizer.param_groups[0]['lr']
        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            if train_config['lr_scheduler'] == 'reduce_on_plateau':
                lr_scheduler.step(avg_recon_loss)
            else:
                lr_scheduler.step()

        # record
        if logger:
            logger.add_scalar('training reconstruction loss', avg_recon_loss, epoch)
            logger.add_scalar('training adversial loss', avg_adv_loss, epoch)
            logger.add_scalar('training discriminator loss', avg_dis_loss, epoch)
            logger.add_scalar('learning rate', curr_lr, epoch)

    # save final model
    save_weights(model, os.path.join(save_path, 'final_weights.pt'))
    if logger:
        logger.close()


def evaluate(model, checkpoint, train_config, test_dataset, estimator, device):
    weights = torch.load(checkpoint)
    model.load_state_dict(weights, strict=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        num_workers=train_config['num_workers'],
        shuffle=False,
        pin_memory=train_config['pin_memory']
    )

    print('Running on Test set...')
    eval(model, test_loader, train_config['criterion'], estimator, device)

    print('========================================')
    print('Finished! test acc: {}'.format(estimator.get_accuracy(6)))
    print('Confusion Matrix:')
    print(estimator.conf_mat)
    print('quadratic kappa: {}'.format(estimator.get_kappa(6)))
    print('========================================')


def eval(model, dataloader, loss_function, estimator, device, masker):
    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()

    steps = 0
    val_loss = 0
    for step, test_data in enumerate(dataloader):
        X = test_data
        X = X.to(device)
        noisy, mask = masker.mask(X, step, mode='training')

        recon_x = model(noisy)
        loss = loss_function(recon_x * mask, X.detach() * mask)

        steps += 1
        val_loss += loss.item()

    val_loss = val_loss / steps

    model.train()
    torch.set_grad_enabled(True)
    return val_loss


def initialize_sampler(data_config, train_dataset):
    sampling_strategy = data_config['sampling_strategy']
    if sampling_strategy == 'balance':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, 1)
    elif sampling_strategy == 'dynamic':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, data_config['sampling_weights_decay_rate'])
    else:
        weighted_sampler = None
    return weighted_sampler


# define data loader
def initialize_dataloader(train_config, train_dataset, val_dataset, weighted_sampler):
    batch_size = train_config['batch_size']
    num_workers = train_config['num_workers']
    pin_memory = train_config['pin_memory']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(weighted_sampler is None),
        sampler=weighted_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


# define loss and loss weights scheduler
def initialize_loss(train_config, train_dataset, device):
    criterion = train_config['criterion']
    criterion_config = train_config['criterion_config']

    weight = None
    loss_weight_scheduler = None
    loss_weight = train_config['loss_weight']
    if criterion == 'cross_entropy':
        if loss_weight == 'balance':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)
        elif loss_weight == 'dynamic':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, train_config['loss_weight_decay_rate'])
        elif isinstance(loss_weight, list):
            assert len(loss_weight) == len(train_dataset.classes)
            weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=device)
        loss_function = nn.CrossEntropyLoss(weight=weight, **criterion_config)
    elif criterion == 'mean_square_root':
        loss_function = nn.MSELoss(**criterion_config)
    elif criterion == 'L1':
        loss_function = nn.L1Loss(**criterion_config)
    elif criterion == 'smooth_L1':
        loss_function = nn.SmoothL1Loss(**criterion_config)
    elif criterion == 'kappa_loss':
        loss_function = KappaLoss(**criterion_config)
    elif criterion == 'focal_loss':
        loss_function = FocalLoss(**criterion_config)
    else:
        raise NotImplementedError('Not implemented loss function.')

    return loss_function, loss_weight_scheduler


# define optmizer
def initialize_optimizer(train_config, model):
    optimizer_strategy = train_config['optimizer']
    learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    momentum = train_config['momentum']
    nesterov = train_config['nesterov']
    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer


# define learning rate scheduler
def initialize_lr_scheduler(train_config, optimizer):
    learning_rate = train_config['learning_rate']
    warmup_epochs = train_config['warmup_epochs']
    scheduler_strategy = train_config['lr_scheduler']
    scheduler_config = train_config['lr_scheduler_config']

    if scheduler_strategy in [None, 'None']:
        lr_scheduler = None
    elif scheduler_strategy == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
    elif scheduler_strategy == 'multiple_steps':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
    elif scheduler_strategy == 'reduce_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
    elif scheduler_strategy == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_config)
    elif scheduler_strategy == 'clipped_cosine':
        lr_scheduler = ClippedCosineAnnealingLR(optimizer, **scheduler_config)
    else:
        raise NotImplementedError('Not implemented learning rate scheduler.')

    if warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(optimizer, warmup_epochs, learning_rate)
    else:
        warmup_scheduler = None

    return lr_scheduler, warmup_scheduler
