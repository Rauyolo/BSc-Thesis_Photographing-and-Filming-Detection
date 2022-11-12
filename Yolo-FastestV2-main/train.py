'''
https://github.com/dog-qiuqiu/Yolo-FastestV2
'''

import os
import math
import time
import argparse
import numpy as np
from tqdm import tqdm
from numpy.testing._private.utils import print_assert_equal

import torch
from torch import optim
from torch.utils.data import dataset
from numpy.core.fromnumeric import shape

from torchsummary import summary

import utils.loss
import utils.utils
import utils.datasets
import model.detector


if __name__ == '__main__':
    # Specify training profile
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='Specify training profile *.data')
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    print("训练配置:")
    print(cfg)

    # dataset loading
    train_dataset = utils.datasets.TensorDataset(cfg["train"], cfg["width"], cfg["height"], imgaug = True)
    val_dataset = utils.datasets.TensorDataset(cfg["val"], cfg["width"], cfg["height"], imgaug = False)

    batch_size = int(cfg["batch_size"] / cfg["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # Training set
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=utils.datasets.collate_fn,
                                                   num_workers=nw,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   persistent_workers=True
                                                   )
    # validation set
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 collate_fn=utils.datasets.collate_fn,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 persistent_workers=True
                                                 )

    # Specify the backend device CUDA&CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine whether to load a pretrained model
    load_param = False
    premodel_path = cfg["pre_weights"]
    if premodel_path != None and os.path.exists(premodel_path):
        load_param = True

    # Initialize the model structure
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], load_param).to(device)
    summary(model, input_size=(3, cfg["height"], cfg["width"]))

    # Load pretrained model parameters
    if load_param == True:
        model.load_state_dict(torch.load(premodel_path, map_location=device), strict = False)
        print("Load finefune model param: %s" % premodel_path)
    else:
        print("Initialize weights: model/backbone/backbone.pth")

    # Building an SGD optimizer
    optimizer = optim.SGD(params=model.parameters(), lr=cfg["learning_rate"], momentum=0.949, weight_decay=0.0005,)

    #optimizer = optim.Adamax(params=model.parameters(), lr=cfg["learning_rate"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # Learning rate decay strategy
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["steps"], gamma=0.2)

    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1, verbose=False)
    print('Starting training for %g epochs...' % cfg["epochs"])

    batch_num = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(train_dataloader)

        for imgs, targets in pbar:
            # data preprocessing
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)

            # Model reasoning
            preds = model(imgs)
            # loss calculation
            iou_loss, obj_loss, cls_loss, total_loss = utils.loss.compute_loss(preds, targets, cfg, device)

            # Backpropagation solves the gradient
            total_loss.backward()

            # Learning rate warm-up
            for g in optimizer.param_groups:
                warmup_num =  5 * len(train_dataloader)
                if batch_num <= warmup_num:
                    scale = math.pow(batch_num/warmup_num, 4)
                    g['lr'] = cfg["learning_rate"] * scale
                    
                lr = g["lr"]

            # Update model parameters
            if batch_num % cfg["subdivisions"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Printing related information
            info = "Epoch:%d LR:%f CIou:%f Obj:%f Cls:%f Total:%f" % (
                    epoch, lr, iou_loss, obj_loss, cls_loss, total_loss)
            pbar.set_description(info)

            batch_num += 1

        with open("./loss.txt", 'a') as fWrite:
            fWrite.writelines(str(total_loss))
            fWrite.writelines('\n')

        # model save
        if epoch % 10 == 0 and epoch > 0:
            model.eval()
            # Model evaluation
            print("Computing mAP...")
            _, _, AP, _ = utils.utils.evaluation(val_dataloader, cfg, model, device)
            print("Computing PR...")
            precision, recall, _, f1 = utils.utils.evaluation(val_dataloader, cfg, model, device, 0.3)
            print("Precision:%f Recall:%f AP:%f F1:%f"%(precision, recall, AP, f1))

            torch.save(model.state_dict(), "weights/%s-%d-epoch-%fap-model.pth" %
                      (cfg["model_name"], epoch, AP))

        # Learning rate adjustment
        scheduler.step()
