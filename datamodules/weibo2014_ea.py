"""
Weibo2014 DataModules with Euclidean Alignment (EA) and optional EA hard gating.

Mirrors the structure of bcic4_2a_ea.py but uses stratified train/test split
instead of session split (Weibo2014 has only a single session).
"""
from typing import Optional, Tuple
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import torch

from .base import BaseDataModule
from .bcic4_2a_ea import BCICIV2aEAUtils, CombinedSourceTargetDataset, CDANCollate
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


class Weibo2014SubDep(BaseDataModule, BCICIV2aEAUtils):
    """Subject-Dependent mode for Weibo2014 with optional EA."""

    all_subject_ids = list(range(1, 11))
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

        X, y = _load_data_from_dataset(self.dataset)

        test_ratio = self.preprocessing_dict.get("test_ratio", 0.3)
        seed = self.preprocessing_dict.get("split_seed", 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=seed,
        )

        if self.preprocessing_dict.get("ea", False):
            X_train, X_test = self._align_with_train_reference(X_train, X_train, X_test)

        if self.preprocessing_dict.get("z_scale", False):
            X_train, X_test = BaseDataModule._z_scale(X_train, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)


class Weibo2014LOSO(Weibo2014SubDep):
    """LOSO mode for Weibo2014 with optional EA.

    EA is applied per-subject: each subject's data is aligned using its own
    training split as reference, preventing cross-subject covariance leakage.
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

    def _subject_split_arrays(self, splitted_ds, subj_id):
        """Load and stratified-split a single subject's data into train/test."""
        subj_ds = splitted_ds[str(subj_id)]
        X, y = _load_data_from_dataset(subj_ds)

        test_ratio = self.preprocessing_dict.get("test_ratio", 0.3)
        seed = self.preprocessing_dict.get("split_seed", 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=seed,
        )
        return X_train, y_train, X_test, y_test

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        splitted_ds = self.dataset.split("subject")
        train_subjects = [s for s in self.all_subject_ids if s != self.subject_id]
        use_ea = self.preprocessing_dict.get("ea", False)

        src_train_xs, src_train_ys = [], []
        src_val_xs, src_val_ys = [], []

        for subj_id in train_subjects:
            X_tr, y_tr, X_te, y_te = self._subject_split_arrays(splitted_ds, subj_id)
            if use_ea:
                X_tr, X_te = self._align_with_train_reference(X_tr, X_tr, X_te)
            src_train_xs.append(X_tr)
            src_train_ys.append(y_tr)
            src_val_xs.append(X_te)
            src_val_ys.append(y_te)

        # Target subject
        X_tgt_tr, _, X_test, y_test = self._subject_split_arrays(splitted_ds, self.subject_id)
        if use_ea:
            X_test = self._align_with_train_reference(X_tgt_tr, X_test)[0]

        X_train = np.concatenate(src_train_xs, axis=0)
        y_train = np.concatenate(src_train_ys, axis=0)
        X_val = np.concatenate(src_val_xs, axis=0)
        y_val = np.concatenate(src_val_ys, axis=0)

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
    """CDAN/DANN/CORAL LOSO DataModule for Weibo2014 with EA + optional gate.

    Source: other 9 subjects (labeled)
    Target train: target subject 70% (unlabeled for adaptation)
    Target test: target subject 30%
    Validation: source subjects' test splits
    """
    target_train_dataset = None

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def _check_ea_gate(self, splitted_ds) -> bool:
        """Adaptive EA hard gating (1-sigma log-det rule) for Weibo2014."""
        from utils.ea_gate import compute_log_det, should_disable_ea

        all_log_dets = []
        for subj_id in self.all_subject_ids:
            X_tr, _, _, _ = self._subject_split_arrays(splitted_ds, subj_id)
            all_log_dets.append(compute_log_det(X_tr))

        target_log_det = all_log_dets[self.subject_id - 1]
        gated, threshold, mu, sigma = should_disable_ea(target_log_det, all_log_dets)

        print(f"[EA Gate] Subject {self.subject_id}: "
              f"log|det|={target_log_det:.2f}, "
              f"threshold(μ-σ)={threshold:.2f} "
              f"(μ={mu:.2f}, σ={sigma:.2f})")
        if gated:
            print(f"[EA Gate] ⚠ Subject {self.subject_id} GATED → EA disabled for entire fold")
        else:
            print(f"[EA Gate] ✓ Subject {self.subject_id} PASSED → EA enabled")

        return not gated

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        splitted_ds = self.dataset.split("subject")
        train_subjects = [s for s in self.all_subject_ids if s != self.subject_id]

        # Determine effective EA flag (may be overridden by gate)
        use_ea = self.preprocessing_dict.get("ea", False)
        if use_ea and self.preprocessing_dict.get("ea_gate", False):
            use_ea = self._check_ea_gate(splitted_ds)

        src_train_xs, src_train_ys = [], []
        src_val_xs, src_val_ys = [], []

        for subj_id in train_subjects:
            X_tr, y_tr, X_te, y_te = self._subject_split_arrays(splitted_ds, subj_id)
            if use_ea:
                X_tr, X_te = self._align_with_train_reference(X_tr, X_tr, X_te)
            src_train_xs.append(X_tr)
            src_train_ys.append(y_tr)
            src_val_xs.append(X_te)
            src_val_ys.append(y_te)

        # Target subject: stratified split → 70% adaptation, 30% test
        X_tgt_train, y_tgt_train, X_test, y_test = self._subject_split_arrays(
            splitted_ds, self.subject_id)
        if use_ea:
            X_tgt_train, X_test = self._align_with_train_reference(
                X_tgt_train, X_tgt_train, X_test)

        X_src = np.concatenate(src_train_xs, axis=0)
        y_src = np.concatenate(src_train_ys, axis=0)
        X_val = np.concatenate(src_val_xs, axis=0)
        y_val = np.concatenate(src_val_ys, axis=0)

        if self.preprocessing_dict.get("z_scale", False):
            X_src, X_val, X_tgt_train, X_test = self._z_scale_cdan(
                X_src, X_val, X_tgt_train, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_src, y_src)
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
                use_interaug=self.preprocessing_dict.get("interaug", False)),
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

    def target_train_dataloader(self) -> DataLoader:
        num_workers = 0 if os.name == "nt" else 4
        return DataLoader(
            self.target_train_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
