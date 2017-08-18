import argparse
import os
import json
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import torch.autograd as autograd
import torch.nn as nn
from satellitedata import SatelliteData
from utils import AverageMeter
import hashlib
import time
import logging
from sklearn.metrics import fbeta_score

logger = None

def encode_args(args):
    return hashlib.sha224(str(args).encode()).hexdigest()

def retrieve_model(model_param):
    if args.model == 'alexnet':
        model = torchvision.models.alexnet(pretrained=args.pretrained)
        model.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[1],
            model.classifier[2],
            model.classifier[3],
            model.classifier[4],
            model.classifier[5],
            nn.Linear(4096, 17),
        )
        model.train()
        return model
    elif args.model == "densenet201":
        model = torchvision.models.densenet201(pretrained=args.pretrained)
    else:
        logger.critical("Model %s is not implemented" % args.model)

def retrieve_optimizer(args, params):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr)
    else:
        logger.critical("Optimizer %s is not supported" % args.optimizer)
    return optimizer

def retrieve_criterion(args):
    if args.criterion == "mse":
        return nn.MSELoss()
    elif args.criterion == "l1":
        return nn.L1Loss()


def init_model(args):
    cache_name = encode_args(args)
    foldername = os.path.join(args.cache_filepath, cache_name)

    with open(os.path.join(foldername, 'params.json'), 'w') as f:
        json.dump(vars(args), f)

    model = retrieve_model(args.model)
    model.cuda()
    optimizer = retrieve_optimizer(args, model.parameters())
    torch.save(
        model.state_dict(), 
        os.path.join(foldername, '000_model.tpy')
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(foldername, '000_optimizer.tpy'),
    )

    logger.info("Model, optimizer  initialized and saved in cache folder %s" % foldername)
    
    
def f_score(y, y_pred, averaging, threshold=0.5):
    return fbeta_score(y, np.array(y_pred > threshold, dtype=float), 2, average=averaging)


def evaluate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    fbeta_score_samples = AverageMeter()
    fbeta_score_micro = AverageMeter()

    actuals = []
    preds = []
    model.eval()
    for i, (x, y) in enumerate(val_loader):
        input_var = autograd.Variable(x, volatile=True).cuda()
        output_var = autograd.Variable(y, volatile=True).cuda()
        pred = model(input_var)
        loss = criterion(pred, output_var)
        actuals.append(y)
        preds.append(pred.cpu().data)
        losses.update(loss.cpu().data.numpy()[0])
        
    model.train()

    actual = torch.cat(actuals, 0).numpy()
    pred = torch.cat(preds, 0).numpy()

    return losses.sum, f_score(actual, pred, 'samples'), f_score(actual, pred, 'micro')


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        output = model(input_var)
        loss = criterion(output, target_var)
        
        losses.update(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()

    logstring = "epoch=%d\tdata_time=%f\tbatch_time=%f\ttrain_loss=%f\t" % (epoch, data_time.sum, batch_time.sum, losses.sum)
    return logstring
    

def train_model(args):
    cache_name = encode_args(args)
    foldername = os.path.join(args.cache_filepath, cache_name)

    model = retrieve_model(args.model)
    model.load_state_dict(
        torch.load(os.path.join(foldername, '000_model.tpy'))
    )
    model.cuda()
    optimizer = retrieve_optimizer(args, model.parameters())
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.debug:
        logger.info("Debug mode enabled")

    valdir = os.path.join(args.data, 'val')
    traindir = os.path.join(args.data, 'train')
    targetpath = os.path.join(args.data, 'targets.json')
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
            logger,
            args.debug,
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
            logger,
            args.debug,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    criterion = retrieve_criterion(args).cuda()

    is_best = {
        "score": 0.0,
        "model_type": "",
        "model": 0.0,
    }

    if args.debug:
        args.epochs = 6
    for epoch in range(0, args.epochs):
        logstring = train(train_loader, model, criterion, optimizer, epoch)
        val_loss, fbeta_samples, fbeta_micro = evaluate(val_loader, model, criterion)
        if is_best["score"] < fbeta_samples:
            is_best["score"] = fbeta_samples
            is_best["model_type"] = args.model
            is_best["model"] = model.state_dict()
        logstring += "val_loss=%f\tfbeta_samples_cur=%f\tfbeta_micro_cur=%f\tfbeta_samples_best=%f\t" % (val_loss, fbeta_samples, fbeta_micro, is_best["score"])
        if epoch%2 == 0:
            logger.info(logstring)

    with open(os.path.join(foldername, 'params.json'), 'r') as f:
        args = json.load(f)
    args["best_val"] = is_best["score"]
    args["model"] = is_best["model"]
    with open(os.path.join(foldername, 'params.json'), 'w') as f:
        json.dump(args, f)


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Trains a DL model")
    parser.add_argument("--data", 
                        help="filepath to splits")
    parser.add_argument("--cache_filepath",
                        help="filepath where to create and save model runs and statistics")
    parser.add_argument("--model",
                        help="NN architecture to use")
    parser.add_argument("--pretrained",
                        default=True,
                        help="Should be pretrained weights be used")
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
                        help="learning rate schedule. Not used for adam")
    parser.add_argument("--lr",
                        type=float,
                        help="initial learning rate")
    parser.add_argument("--weight_decay",
                        help="weight decay")
    parser.add_argument("--criterion",
                        help="error criterion to use for training",
    )
    parser.add_argument("--debug",
                        action="store_true",
                        help="are we making sure the code works? Trains at 10% size")
    args = parser.parse_args()

    cache_name = encode_args(args)
    foldername = os.path.join(args.cache_filepath, cache_name)

    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    else:
        for f in os.listdir(foldername):
            os.remove(f)

    logger = logging.getLogger('')
    hdlr = logging.FileHandler(os.path.join(foldername, "logs.txt"), "a")
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    
    init_model(args)
    train_model(args)
    
