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
    parser.add_argument('--no_standardize', action='store_true', default=False,
                         help='収量値の年度ごと標準化を無効にし、生のg/m²スケールのままlossを計算する（標準化前のbaselineとの比較用）')
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


def _write_year_stats_csv(year_stats, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['year', 'mean', 'std'])
        writer.writeheader()
        for year, (mean, std) in sorted(year_stats.items()):
            writer.writerow({'year': year, 'mean': mean, 'std': std})


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

        # target = [生の収量値, 年平均, 年標準偏差]。
        # --no_standardizeの場合は予測も生スケールなのでそのまま使う。
        # 標準化ありの場合は予測が標準化スケールなので、元のg/m²スケールに戻してから
        # MAE/RMSE/R2を計算する（標準化前のbaselineとそのまま比較できるようにするため）
        target_raw = target[:, 0:1]
        if args.no_standardize:
            pred_raw = pred.detach()
        else:
            year_mean = target[:, 1:2]
            year_std = target[:, 2:3]
            pred_raw = pred.detach() * year_std + year_mean

        preds.append(pred_raw.cpu().numpy())
        targets.append(target_raw.detach().cpu().numpy())

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
    # valのyear_statsは必ずtrain側のものを渡す（val自身の分布を統計量に混ぜるとリークになるため）
    val_dataset = RicePaddyDataLoader(root=args.data_root, split='val', npoints=args.num_point,
                                       split_ratio=args.split_ratio, seed=args.seed,
                                       year_stats=train_dataset.year_stats)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    valDataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    log_string('Year stats (train mean/std, standardize=%s): %s' % (not args.no_standardize, train_dataset.year_stats))
    _write_year_stats_csv(train_dataset.year_stats, str(log_dir / 'year_stats.csv'))

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

            # target = [生の収量値, 年平均, 年標準偏差]。
            # --no_standardizeの場合は生のg/m²スケールのままlossを計算する（標準化前のbaseline再現用）。
            # デフォルトは年ごとの水準差を取り除くため、標準化スケール((raw - mean) / std)で計算する
            target_raw = target[:, 0:1]
            pred, trans_feat = regressor(points)

            if args.no_standardize:
                loss = criterion(pred, target_raw, trans_feat)
                pred_raw = pred.detach()
            else:
                year_mean = target[:, 1:2]
                year_std = target[:, 2:3]
                target_std = (target_raw - year_mean) / year_std
                loss = criterion(pred, target_std, trans_feat)
                # 指標算出はg/m²の生スケールに戻して行う（標準化前と同じ単位で比較できるようにするため）
                pred_raw = pred.detach() * year_std + year_mean

            loss.backward()
            optimizer.step()
            global_step += 1

            train_preds.append(pred_raw.cpu().numpy())
            train_targets.append(target_raw.detach().cpu().numpy())

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
                    'standardized': not args.no_standardize,  # このチェックポイントが標準化ありで学習されたか
                    'year_stats': train_dataset.year_stats,  # 標準化に使った年ごとの平均・標準偏差（推論時の逆標準化に必要）
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
