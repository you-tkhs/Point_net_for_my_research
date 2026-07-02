'''
水田点群データから収量を予測する回帰タスク用の DataLoader。
data/{year}/ 以下の .ply（点群）と yield_value.txt（収量）を読み込む。
'''
import os
import re
import logging
import unicodedata

import numpy as np
from torch.utils.data import Dataset
from plyfile import PlyData

from data_utils.ModelNetDataLoader import pc_normalize

YEAR_DIR_PATTERN = re.compile(r'^\d{4}$')
PLY_FILENAME_PATTERN = re.compile(r'^(\d+)_(.+)\.ply$')


def _parse_yield_file(path):
    '''
    yield_value.txt をパースする。
    実データは「品種名, 収量値」の2列のみ（plot_id列は無い）。
    区切りはタブと連続スペースが混在しており、ヘッダー行（「品種」を含む）は
    先頭ではなくファイル末尾にあるため、位置ではなく内容で判定してスキップする。
    空行もファイル中間に存在するため位置に関わらずスキップする。
    収量値には桁区切りカンマが入っている行があるため float 変換前に除去する。
    ここで返す順序が、.ply ファイルとのランクベース対応付けの基準になる。
    '''
    records = []
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip('\n').rstrip('\r')
        if not line.strip():
            continue
        if '品種' in line:
            continue

        fields = re.split(r'\t| {2,}', line, maxsplit=1)
        variety = fields[0].strip()
        value_str = fields[1].strip().replace(',', '')
        yield_value = float(value_str)
        records.append((variety, yield_value))

    return records


def _list_ply_files(year_dir):
    '''
    year_dir 内の .ply ファイルを plot_id（ファイル名先頭の数字）昇順でソートして返す。
    品種名自体にアンダースコアを含むファイルがあるため、非貪欲にせず `(\\d+)_(.+)\\.ply` のまま使う。
    '''
    entries = []
    for filename in os.listdir(year_dir):
        if not filename.endswith('.ply'):
            continue
        match = PLY_FILENAME_PATTERN.match(filename)
        if match is None:
            raise ValueError(
                'ファイル名が想定パターン "<plot_id>_<variety>.ply" に一致しません: %s'
                % os.path.join(year_dir, filename)
            )
        plot_id = int(match.group(1))
        variety = match.group(2)
        entries.append((plot_id, variety, os.path.join(year_dir, filename)))

    # 数値としてソート（文字列ソートだと "10" < "2" になってしまうため注意）
    entries.sort(key=lambda e: e[0])
    return entries


def _build_year_index(year_dir, year_label):
    '''
    1年分の .ply ファイルと yield_value.txt をランクベースで対応付ける。
    plot_id 番号は年ごとに欠番があり直接キーにできないため、
    .ply を plot_id 昇順ソートしたものと、yield_value.txt のデータ行を出現順に
    並べたものを位置（順位）で1対1対応させる。
    '''
    ply_entries = _list_ply_files(year_dir)
    yield_records = _parse_yield_file(os.path.join(year_dir, 'yield_value.txt'))

    if len(ply_entries) != len(yield_records):
        raise ValueError(
            '%s: .ply ファイル数(%d)と yield_value.txt のデータ行数(%d)が一致しません。'
            '欠番・重複した plot_id が無いか確認してください。'
            % (year_dir, len(ply_entries), len(yield_records))
        )

    records = []
    for (plot_id, variety_filename, path), (variety_yieldfile, yield_value) in zip(ply_entries, yield_records):
        # 対応関係自体は変更しないが、表記ゆれ（全角/半角数字など）を診断用に警告しておく
        norm_filename = unicodedata.normalize('NFKC', variety_filename)
        norm_yieldfile = unicodedata.normalize('NFKC', variety_yieldfile)
        if norm_filename != norm_yieldfile:
            logging.warning(
                '%s: plot_id=%d の品種名表記が一致しません（対応関係は維持） file="%s" yield_value.txt="%s"',
                year_label, plot_id, variety_filename, variety_yieldfile
            )

        records.append({
            'path': path,
            'yield': yield_value,
            'year': year_label,
            'plot_id': plot_id,
            'variety_filename': variety_filename,
            'variety_yieldfile': variety_yieldfile,
        })

    return records


class RicePaddyDataLoader(Dataset):
    def __init__(self, root, split='train', years=None, npoints=4096, split_ratio=0.8, seed=42):
        assert split in ('train', 'val')
        self.npoints = npoints

        if years is None:
            years = sorted(
                name for name in os.listdir(root)
                if YEAR_DIR_PATTERN.match(name) and os.path.isdir(os.path.join(root, name))
            )
        else:
            years = sorted(years)

        rng = np.random.RandomState(seed)

        # 年ごとに層化分割してから結合する。2022年だけデータ数が突出して多い(160/372件)ため、
        # 単純な全体シャッフルだと年ごとの代表性が崩れる可能性を避ける。
        self.samples = []
        for year in years:
            year_records = _build_year_index(os.path.join(root, year), year)

            perm = rng.permutation(len(year_records))
            n_train = int(round(split_ratio * len(year_records)))
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]

            selected_idx = train_idx if split == 'train' else val_idx
            self.samples.extend(year_records[i] for i in selected_idx)

        print('The size of %s data is %d' % (split, len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]

        plydata = PlyData.read(record['path'])
        vertex = plydata['vertex']
        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)

        n = xyz.shape[0]
        if n >= self.npoints:
            choice = np.random.choice(n, self.npoints, replace=False)
        else:
            # 全372ファイルの最小点数は4,771点のため通常発生しない想定だが、
            # 万一未満のファイルが混入しても学習を止めないよう復元抽出でフォールバックする。
            logging.warning(
                '%s: 点数(%d)が npoints(%d)未満のため復元抽出でパディングします',
                record['path'], n, self.npoints
            )
            choice = np.random.choice(n, self.npoints, replace=True)
        point_set = xyz[choice, :]

        point_set = pc_normalize(point_set)

        # 回帰モデルの出力 [B, 1] と nn.MSELoss で次元を一致させるため形状を (1,) にする
        yield_value = np.array([record['yield']], dtype=np.float32)

        return point_set.astype(np.float32), yield_value


if __name__ == '__main__':
    import torch

    data = RicePaddyDataLoader('data', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True)
    for point, target in DataLoader:
        print(point.shape)
        print(target.shape)
        print(target)
        break
