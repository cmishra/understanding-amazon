import argparse
import os
import json

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from satellitedata import SatelliteData
from utils import AverageMeter
import hashlib
import logging
import time


def retrieve_model(model_param):
    if args.model == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        model.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[1],
            model.classifier[2],
            model.classifier[3],
            model.classifier[4],
            model.classifier[5],
            nn.Linear(4096, 17),
        )
        return model
    else:
        exit("Model %s is not implemented" % args.model)

def retrieve_optimizer(args, params):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr)
    else:
        exit("Optimizer %s is not supported" % args.optimizer)
    return optimizer

def retrieve_criterion(args):
    if args.criterion == "mse":
        return nn.MSELoss()
    elif args.criterion == "l1":
        return nn.L1Loss()


def init_model(args):
    cache_name = hashlib.sha224(str(args)).hexdigest()
    foldername = os.path.join(args.cache_filepath, cache_name)

    if os.path.isdir(foldername):
        print("Cache folder %s exists -- deleting and recreating" % cache_name)
        for f in os.listdir(foldername):
            os.remove(os.path.join(foldername, f))
        os.rmdir(foldername)
    
    os.mkdir(foldername)
    with open(os.path.join(foldername, 'params.json'), 'w') as f:
        json.dump(vars(args), f)

    model = retrieve_model(args.model)
    optimizer = retrieve_optimizer(args, model.parameters())
    torch.save(
        model.state_dict(), 
        os.path.join(foldername, '000_model.tpy')
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(foldername, '000_optimizer.tpy'),
    )

    logging.log(20, "Model, optimizer  initialized and saved in cache folder %s" % foldername)
    
    
    
def evaluate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    end = time.time()
    for i, (x, y) in val_loader:
        input_var = autograd.Variable(x, volatile=True)
        output_var = autograd.Variable(y, volatile=True)

        pred = model(input_var)
        loss = criterion(pred, output_var)

        batch_time.update(time.time() - end)
        end = time.time()

    print(batch_time.avg)
    print(losses.avg)
    model.train()

def train_model(args):
    cache_name = hashlib.sha224(str(args)).hexdigest()
    foldername = os.path.join(args.cache_filepath, cache_name)

    model = retrieve_model(args.model)
    model.load_state_dict(
        torch.load(os.path.join(foldername, '000_model.tpy'))
    )
    optimizer = retrieve_optimizer(args, model.parameters())
    optimizer.load_state_dict(
        torch.load(os.path.join(foldername, '000_optimizer.tpy')),
    )
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    
    valdir = os.path.join(args.data, 'val')
    traindir = os.path.join(args.data, 'train')
    targetpath = os.path.join(args.data, 'train_v2.csv')
    train_loader = torch.utils.data.DataLoader(
        SatelliteData(
            traindir,
            targetpath,
            transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        SatelliteData(
            valdir,
            targetpath,
            transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    criterion = retrieve_criterion(args)


    evaluate(val_loader, model, criterion)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Trains a DL model")
    parser.add_argument("--data", 
                        help="filepath to splits")
    parser.add_argument("--cache_filepath",
                        help="filepath where to create and save model runs and statistics")
    parser.add_argument("--model",
                        help="NN architecture to use")
    parser.add_argument("--workers", 
                        type=int,
                        help="number of dataloading subprocesses")
    parser.add_argument("--epochs",
                        type=int,
                        help="number of epochs")
    parser.add_argument("--batch_size",
                        type=int,
                        help="batch size")
    parser.add_argument("--optimizer",
                        help="optimizer to be used")
    parser.add_argument("--momentum",
                        default=0.0,
                        type=float,
                        help="momentum term to be used. Not used for adam")
    parser.add_argument("--lr_schedule",
                        default="Constant",
                        help="learning rate schedule. Not user for adam")
    parser.add_argument("--lr",
                        type=float,
                        help="initial learning rate")
    parser.add_argument("--weight_decay",
                        help="weight decay")
    parser.add_argument("--criterion",
                        help="error criterion to use for training",
    )
    
    args = parser.parse_args()
    init_model(args)
    train_model(args)
    
