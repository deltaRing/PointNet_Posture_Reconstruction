"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

from new_models.ConvCoder import *
from new_models.self_coder import *
from new_models.self_attention_coder import *

import scipy.io as scio

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=4, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=2000, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=192, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, predictor, loader):
    classifier = model.eval()
    predictor    = predictor.eval()
    mse_loss     = get_mse_loss_function()
    losses = []

    for j, (points, target, poses) in tqdm(enumerate(loader), total=len(loader)):

        if j == 523:
            break

        if not args.use_cpu:
            points, target, poses = points.cuda(), target.cuda(), poses.cuda()

        points = points.transpose(2, 1)
        pred, tran = classifier(points)
        poses_re   = predictor(tran)

        loss = mse_loss(poses_re, poses)
        losses.append(loss.detach().cpu().numpy())

        scio.savemat(str(j) + 'gene.mat', {'gene_pose':poses_re.detach().cpu().numpy()})
        scio.savemat(str(j) + 'true.mat', {'pose': poses.detach().cpu().numpy()})

    print('Test Instance Loss: %f' % np.mean(np.array(losses)))


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/custom_motion/'
    pose_data = 'data/data_pos/'

    train_dataset = ModelNetDataLoader(root=data_path, pose_root=pose_data, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, pose_root=pose_data, args=args, split='test', process_data=args.process_data)
    # pose_train_dataset = ModelNetDataLoader(root=pose_data, pose_root=pose_data, args=args, split='train', process_data=args.process_data,
    #                                  is_pose=True)
    # pose_test_dataset = ModelNetDataLoader(root=pose_data, pose_root=pose_data, args=args, split='test', process_data=args.process_data,
    #                                  is_pose=True)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # poseTrainDataLoader = torch.utils.data.DataLoader(pose_train_dataset, batch_size=args.batch_size, shuffle=False,
                                                     # num_workers=10)
    # poseTestDataLoader = torch.utils.data.DataLoader(pose_test_dataset, batch_size=args.batch_size, shuffle=False,
                                                     # num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    predictor = Auto_Point_Encoder(batch_size=args.batch_size, input_size=1024, hidden_size=1024,
                                   output_size=-1, poses=11)
    pose_encoder = ConvEncoder(batch_size=args.batch_size)
    pose_decoder = ConvDecoder(batch_size=args.batch_size)
    mse_loss     = get_mse_loss_function()
    bce_loss     = get_bce_loss_function()
    kl_loss      = get_kl_loss_function()

    if not args.use_cpu:
        classifier = classifier.cuda()
        # criterion  = criterion.cuda()
        predictor    = predictor.cuda()
        pose_encoder = pose_encoder.cuda()
        pose_decoder = pose_decoder.cuda()
        mse_loss   = mse_loss.cuda()
        bce_loss   = bce_loss.cuda()
        kl_loss    = kl_loss.cuda()

    try:
        checkpoint = torch.load('./checkpoints/35.pth')
        # start_epoch = checkpoint['epoch']
        start_epoch = 0
        classifier.load_state_dict(checkpoint['state_dict'])
        predictor.load_state_dict(checkpoint['coder_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        optimizer2 = torch.optim.Adam(
            predictor.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        # with torch.no_grad():
        #     test(classifier, predictor, testDataLoader)

        mean_correct = []
        classifier = classifier.train()
        predictor = predictor.train()

        scheduler.step()
        iii = 0
        for batch_id, (points, target, poses) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            optimizer2.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target, poses = points.cuda(), target.cuda(), poses.cuda()

            pred, trans_feat = classifier(points)
            pose_pred        = predictor(trans_feat)

            #loss_mse_1            = mse_loss(x_pose, poses)
            #loss_mse_2            = mse_loss(r_pose, poses)
            #loss_bce_1            = bce_loss(x_conf, confi)
            #loss_bce_2            = bce_loss(r_conf, confi)

            pppp = poses.detach().cpu().numpy()
            ppp  = pose_pred.detach().cpu().numpy()

            if iii % 100 == 0:
                scio.savemat(str(iii) + 'data1.mat', {'data1':pppp})
                scio.savemat(str(iii) + 'data2.mat', {'data2': ppp})

            loss  = mse_loss(pose_pred, poses)

            log_string('Loss: %f' % loss.detach().cpu().numpy())

            loss.backward()
            # loss = criterion(pred, target.long(), trans_feat)
            # pred_choice = pred.data.max(1)[1]

            # correct = pred_choice.eq(target.long().data).cpu().sum()
            # mean_correct.append(correct.item() / float(points.size()[0]))
            # loss.backward()
            mean_correct.append(loss.detach().cpu().numpy())
            iii += 1
            optimizer.step()
            optimizer2.step()

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Loss: %f' % train_instance_acc)

        state = {
            'state_dict': classifier.state_dict(),
            'coder_state_dict': predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        log_string('Saving Dict...')
        savepath = str(checkpoints_dir) + '/' + str(global_step) + '.pth'
        torch.save(state, savepath)
        global_step += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
