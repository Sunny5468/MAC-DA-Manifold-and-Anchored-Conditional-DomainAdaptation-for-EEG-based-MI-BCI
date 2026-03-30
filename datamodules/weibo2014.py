from typing import Optional
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
import torch
import os

from .base import BaseDataModule, make_collate_fn
from .bcic4_2a import CombinedSourceTargetDataset, CDANCollate
from utils.load_weibo2014 import load_weibo2014


def _load_data_from_dataset(ds):
    """从 braindecode WindowsDataset 中提取 numpy 数组。"""
    X_list, y_list = [], []
    for run in ds.datasets:
        if hasattr(run, 'windows'):
            data = run.windows.load_data()._data
        elif hasattr(run, '_data') and run._data is not None:
            data = run._data
        else:
            data = np.array([run[i][0] for i in range(len(run))])
        X_list.append(data)
        y_list.append(run.y)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def _load_data_from_run(run):
    """从单个 run 中提取 numpy 数组。"""
    if hasattr(run, 'windows'):
        return run.windows.load_data()._data
    elif hasattr(run, '_data') and run._data is not None:
        return run._data
    else:
        return np.array([run[i][0] for i in range(len(run))])


class Weibo2014SubDep(BaseDataModule):
    """Subject-Dependent mode for Weibo2014 dataset.

    单 session 数据集，使用 stratified train/test split 替代 session split。
    筛选后每人 4 类 × 80 = 320 trials。
    """
    all_subject_ids = list(range(1, 11))  # 10 个被试
    class_names = ["left_hand", "right_hand", "hands", "feet"]
    channels = 22
    classes = 4

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_weibo2014(
            subject_ids=[self.subject_id],
            preprocessing_dict=self.preprocessing_dict,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        # Weibo2014 只有单 session，直接加载全部数据后做 stratified split
        X, y = _load_data_from_dataset(self.dataset)

        test_ratio = self.preprocessing_dict.get("test_ratio", 0.3)
        seed = self.preprocessing_dict.get("split_seed", 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=seed,
        )

        if self.preprocessing_dict.get("z_scale", False):
            X_train, X_test = BaseDataModule._z_scale(X_train, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)


class Weibo2014LOSO(Weibo2014SubDep):
    """Leave-One-Subject-Out mode for Weibo2014 dataset.

    源域: 其他 9 个被试的全部数据
    验证集: 从源域中按比例抽取
    目标域 test: 目标被试的 30% 数据
    """
    val_dataset = None

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        if self.dataset is not None:
            return
        self.dataset = load_weibo2014(
            subject_ids=self.all_subject_ids,
            preprocessing_dict=self.preprocessing_dict,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        splitted_ds = self.dataset.split("subject")
        train_subjects = [s for s in self.all_subject_ids if s != self.subject_id]

        test_ratio = self.preprocessing_dict.get("test_ratio", 0.3)
        seed = self.preprocessing_dict.get("split_seed", 42)

        # 源域: 其他被试的全部数据
        src_X_list, src_y_list = [], []
        for subj_id in train_subjects:
            subj_ds = splitted_ds[str(subj_id)]
            X_s, y_s = _load_data_from_dataset(subj_ds)
            src_X_list.append(X_s)
            src_y_list.append(y_s)

        X_src_all = np.concatenate(src_X_list, axis=0)
        y_src_all = np.concatenate(src_y_list, axis=0)

        # 从源域中抽取验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_src_all, y_src_all, test_size=test_ratio, stratify=y_src_all,
            random_state=seed,
        )

        # 目标被试: stratified split
        tgt_ds = splitted_ds[str(self.subject_id)]
        X_tgt, y_tgt = _load_data_from_dataset(tgt_ds)
        _, X_test, _, y_test = train_test_split(
            X_tgt, y_tgt, test_size=test_ratio, stratify=y_tgt,
            random_state=seed,
        )

        if self.preprocessing_dict.get("z_scale", False):
            X_train, X_val, X_test = BaseDataModule._z_scale_tvt(X_train, X_val, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=0,
            pin_memory=True,
        )


class Weibo2014LOSO_CDAN(Weibo2014LOSO):
    """CDAN/DANN/CORAL 专用的 LOSO DataModule for Weibo2014.

    源域: 其他 9 个被试的全部数据（有标签）
    目标域 train: 目标被试的 70% 数据（用于域适应，标签不使用）
    目标域 test: 目标被试的 30% 数据
    验证集: 从源域中按比例抽取
    """
    target_train_dataset = None

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        splitted_ds = self.dataset.split("subject")
        train_subjects = [s for s in self.all_subject_ids if s != self.subject_id]

        test_ratio = self.preprocessing_dict.get("test_ratio", 0.3)
        seed = self.preprocessing_dict.get("split_seed", 42)

        # 源域: 其他被试的全部数据
        src_X_list, src_y_list = [], []
        for subj_id in train_subjects:
            subj_ds = splitted_ds[str(subj_id)]
            X_s, y_s = _load_data_from_dataset(subj_ds)
            src_X_list.append(X_s)
            src_y_list.append(y_s)

        X_src_all = np.concatenate(src_X_list, axis=0)
        y_src_all = np.concatenate(src_y_list, axis=0)

        # 从源域中抽取验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_src_all, y_src_all, test_size=test_ratio, stratify=y_src_all,
            random_state=seed,
        )

        # 目标被试: stratified split → 70% 用于域适应, 30% 用于测试
        tgt_ds = splitted_ds[str(self.subject_id)]
        X_tgt, y_tgt = _load_data_from_dataset(tgt_ds)
        X_tgt_train, X_test, y_tgt_train, y_test = train_test_split(
            X_tgt, y_tgt, test_size=test_ratio, stratify=y_tgt,
            random_state=seed,
        )

        # 标准化: 用源域训练数据拟合 scaler
        if self.preprocessing_dict.get("z_scale", False):
            X_train, X_val, X_tgt_train, X_test = self._z_scale_cdan(
                X_train, X_val, X_tgt_train, X_test,
            )

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.target_train_dataset = BaseDataModule._make_tensor_dataset(X_tgt_train, y_tgt_train)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

        self.combined_train_dataset = CombinedSourceTargetDataset(
            self.train_dataset,
            self.target_train_dataset,
            use_interaug=self.preprocessing_dict.get("interaug", False),
        )

    @staticmethod
    def _z_scale_cdan(X_src, X_val, X_tgt_train, X_test):
        """用源域数据拟合 scaler，应用到所有数据。"""
        from sklearn.preprocessing import StandardScaler

        s, c, t = X_src.shape
        X_src_2d = X_src.transpose(1, 0, 2).reshape(c, -1).T
        X_val_2d = X_val.transpose(1, 0, 2).reshape(c, -1).T
        X_tgt_2d = X_tgt_train.transpose(1, 0, 2).reshape(c, -1).T
        X_test_2d = X_test.transpose(1, 0, 2).reshape(c, -1).T

        sc = StandardScaler().fit(X_src_2d)

        X_src = sc.transform(X_src_2d).T.reshape(c, s, t).transpose(1, 0, 2)
        X_val = sc.transform(X_val_2d).T.reshape(c, X_val.shape[0], t).transpose(1, 0, 2)
        X_tgt_train = sc.transform(X_tgt_2d).T.reshape(c, X_tgt_train.shape[0], t).transpose(1, 0, 2)
        X_test = sc.transform(X_test_2d).T.reshape(c, X_test.shape[0], t).transpose(1, 0, 2)

        return X_src, X_val, X_tgt_train, X_test

    def train_dataloader(self) -> DataLoader:
        num_workers = 0 if os.name == "nt" else 4
        return DataLoader(
            self.combined_train_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=CDANCollate(
                use_interaug=self.preprocessing_dict.get("interaug", False)
            ),
        )

    def val_dataloader(self) -> DataLoader:
        num_workers = 0 if os.name == "nt" else 4
        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    def test_dataloader(self) -> DataLoader:
        num_workers = 0 if os.name == "nt" else 4
        return DataLoader(
            self.test_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
