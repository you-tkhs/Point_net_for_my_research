import os
import sys
import csv
import torch
import numpy as np

import datetime
import logging
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use('Agg')  # コンテナに$DISPLAYが無いため画面表示なしのバックエンドを使う
import matplotlib.pyplot as plt

from data_utils.RicePaddyDataLoader import RicePaddyDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--model', default='pointnet_regression', help='model name [default: pointnet_regression]')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed for train/val split (train/valで必ず同じ値を使うこと)')
    parser.add_argument('--data_root', type=str, default='data', help='root directory containing data/{year}/')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='train split ratio (学習データが372件と少ないため9:1をデフォルトにしている)')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def _compute_metrics(targets, preds):
    targets = np.concatenate(targets).reshape(-1)
    preds = np.concatenate(preds).reshape(-1)
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    return mae, rmse, r2


METRICS_FIELDS = ['epoch', 'train_mae', 'train_rmse', 'train_r2', 'val_mae', 'val_rmse', 'val_r2']


def _load_metrics_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [{k: (int(v) if k == 'epoch' else float(v)) for k, v in row.items()} for row in reader]


def _write_metrics_csv(history, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def _plot_learning_curve(history, save_path):
    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for ax, metric, ylabel in zip(axes, ['mae', 'rmse', 'r2'], ['MAE', 'RMSE', 'R2']):
        ax.plot(epochs, [row['train_' + metric] for row in history], label='train')
        ax.plot(epochs, [row['val_' + metric] for row in history], label='val')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('epoch')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def evaluate(model, loader, args):
    preds = []
    targets = []
    for points, target in tqdm(loader, total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = model(points)

        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())

    return _compute_metrics(targets, preds)


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
    exp_dir = exp_dir.joinpath('regression')
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

    # train/valは同じ root・split_ratio・seed で呼び出すこと（分割が食い違うとvalがtrainに漏れる）
    train_dataset = RicePaddyDataLoader(root=args.data_root, split='train', npoints=args.num_point,
                                         split_ratio=args.split_ratio, seed=args.seed)
    val_dataset = RicePaddyDataLoader(root=args.data_root, split='val', npoints=args.num_point,
                                       split_ratio=args.split_ratio, seed=args.seed)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    valDataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet_utils.py', str(exp_dir))
    shutil.copy('./train_regression.py', str(exp_dir))

    regressor = model.get_model(channel=3)
    criterion = model.get_loss()
    regressor.apply(inplace_relu)

    if not args.use_cpu:
        regressor = regressor.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        regressor.load_state_dict(checkpoint['model_state_dict'])
        best_val_mae = checkpoint.get('val_mae', float('inf'))
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        best_val_mae = float('inf')

    metrics_csv_path = str(log_dir / 'metrics.csv')
    learning_curve_path = str(log_dir / 'learning_curve.png')
    history = _load_metrics_csv(metrics_csv_path)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            regressor.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(regressor.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_epoch = 0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        train_preds = []
        train_targets = []
        regressor = regressor.train()

        scheduler.step()
        for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = regressor(points)
            loss = criterion(pred, target, trans_feat)
            loss.backward()
            optimizer.step()
            global_step += 1

            train_preds.append(pred.detach().cpu().numpy())
            train_targets.append(target.detach().cpu().numpy())

        train_mae, train_rmse, train_r2 = _compute_metrics(train_targets, train_preds)
        # augmentation適用後・trainモードでの参考値（楽観的/ノイズを含む。分類スクリプトのtrain accuracyと同じ位置づけ）
        log_string('Train MAE: %f, RMSE: %f, R2: %f' % (train_mae, train_rmse, train_r2))

        with torch.no_grad():
            val_mae, val_rmse, val_r2 = evaluate(regressor.eval(), valDataLoader, args)
            log_string('Val MAE: %f, RMSE: %f, R2: %f' % (val_mae, val_rmse, val_r2))

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch + 1
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'model_state_dict': regressor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            log_string('Best Val MAE: %f (epoch %d)' % (best_val_mae, best_epoch))

            history.append({
                'epoch': epoch + 1,
                'train_mae': train_mae, 'train_rmse': train_rmse, 'train_r2': train_r2,
                'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2,
            })
            _write_metrics_csv(history, metrics_csv_path)
            _plot_learning_curve(history, learning_curve_path)

            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
