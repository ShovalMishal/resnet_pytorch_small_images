# train.py
# !/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os

import time
from datetime import datetime
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from resnet_pytorch_small_images.utils import get_network, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
device = torch.device("cuda" if torch.cuda.is_available() else "")


def train(epoch, train_dataloader, net, args, optimizer, loss_function, writer, warmup_scheduler):
    start = time.time()
    net.train()
    for batch_index, batch in enumerate(train_dataloader):
        if args.gpu:
            labels = batch["labels"].to(device)
            images = batch["pixel_values"].to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * train_dataloader.batch_size + len(images),
            total_samples=len(train_dataloader.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(net, val_dataloader, args, loss_function, writer, epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for batch_ind, batch in enumerate(val_dataloader):
        if args.gpu:
            labels = batch["labels"].to(device)
            images = batch["pixel_values"].to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(val_dataloader.dataset),
        correct.float() / len(val_dataloader.dataset),
        finish - start
    ))
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(val_dataloader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(val_dataloader.dataset), epoch)

    return correct.float() / len(val_dataloader.dataset)


def create_and_train_model(train_dataloader, val_dataloader, checkpoint, log_dir, logger, resume=False, max_epoch=100,
                           milestones=[30, 60, 90], save_epoch=10, loss_class_weights=False):
    args = SimpleNamespace()
    args.net = "resnet18"
    args.gpu = True
    args.lr = 0.001
    args.resume = resume
    args.warm = 1
    args.num_classes = len(train_dataloader.dataset.classes)

    net = get_network(args)

    # time of we run the script
    time_now = datetime.now().strftime(DATE_FORMAT)
    training_weights = torch.tensor(list(train_dataloader.dataset.class_weights.values())).to(device) \
        if loss_class_weights else None
    validation_weights = torch.tensor(list(val_dataloader.dataset.class_weights.values())).to(device) \
        if loss_class_weights else None
    train_loss_function = nn.CrossEntropyLoss(weight=training_weights)
    val_loss_function = nn.CrossEntropyLoss(weight=validation_weights)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)  # learning rate decay
    iter_per_epoch = len(train_dataloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(checkpoint, fmt=DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(checkpoint, recent_folder)

    else:
        checkpoint_path = os.path.join(checkpoint, time_now)

    # use tensorboard
    os.makedirs(log_dir, exist_ok=True)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
        log_dir, time_now))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.to(device)
    writer.add_graph(net, input_tensor)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(checkpoint, recent_folder))
        if best_weights:
            weights_path = os.path.join(checkpoint, args.net, recent_folder, best_weights)
            logger.info('found best acc weights file:{}'.format(weights_path))
            logger.info('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            logger.info('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(checkpoint, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(checkpoint, recent_folder, recent_weights_file)
        logger.info('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(checkpoint, recent_folder))

    for epoch in tqdm(range(1, max_epoch + 1)):
        logger.info('Starting epoch {}\n'.format(epoch))
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch=epoch,
              train_dataloader=train_dataloader,
              net=net,
              args=args,
              optimizer=optimizer,
              loss_function=train_loss_function,
              writer=writer,
              warmup_scheduler=warmup_scheduler)
        acc = eval_training(net=net, val_dataloader=val_dataloader, args=args, loss_function=val_loss_function,
                            writer=writer, epoch=epoch, tb=True)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > milestones[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            logger.info('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % save_epoch:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            logger.info('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
