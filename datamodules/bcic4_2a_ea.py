from typing import Optional, Tuple
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import torch

from .base import BaseDataModule
from utils.load_bcic4 import load_bcic4


class BCICIV2aEAUtils:
    """Utility mixin for Euclidean Alignment (EA) on EEG trials."""

    @staticmethod
    def _safe_covariance(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        # x shape: [C, T]
        c = x.shape[0]
        cov = (x @ x.T) / max(x.shape[1], 1)
        cov = 0.5 * (cov + cov.T)
        cov += eps * np.eye(c, dtype=cov.dtype)
        return cov

    @staticmethod
    def _inv_sqrtm_spd(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        # Numerically stable inverse square-root for SPD matrix.
        evals, evecs = np.linalg.eigh(mat)
        evals = np.clip(evals, eps, None)
        inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        return inv_sqrt

    @classmethod
    def _fit_ea_transform(cls, x_ref: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Fit EA transform A = R^{-1/2} from reference trials.

        Args:
            x_ref: Reference trials, shape [N, C, T].
            eps: Numerical stability term.
        """
        covs = [cls._safe_covariance(trial, eps=eps) for trial in x_ref]
        ref_cov = np.mean(np.stack(covs, axis=0), axis=0)
        return cls._inv_sqrtm_spd(ref_cov, eps=eps)

    @staticmethod
    def _apply_ea(x: np.ndarray, transform: np.ndarray) -> np.ndarray:
        # Apply A @ X for each trial, preserving [N, C, T] layout.
        return np.einsum("ij,njt->nit", transform, x)

    @classmethod
    def _align_with_train_reference(
        cls,
        x_train_ref: np.ndarray,
        *arrays_to_align: np.ndarray,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, ...]:
        transform = cls._fit_ea_transform(x_train_ref, eps=eps)
        return tuple(cls._apply_ea(arr, transform) for arr in arrays_to_align)


class BCICIV2a(BaseDataModule, BCICIV2aEAUtils):
    """Subject-Dependent mode for BCI Competition IV 2a dataset."""

    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(
            subject_ids=[self.subject_id],
            dataset="2a",
            preprocessing_dict=self.preprocessing_dict,
        )

    @staticmethod
    def _load_data_from_dataset(ds):
        x_list, y_list = [], []
        for run in ds.datasets:
            if hasattr(run, "windows"):
                data = run.windows.load_data()._data
            elif hasattr(run, "_data") and run._data is not None:
                data = run._data
            else:
                data = np.array([run[i][0] for i in range(len(run))])
            x_list.append(data)
            y_list.append(run.y)
        return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)

    @staticmethod
    def _get_session_split(ds, session_type: str):
        session_split = ds.split("session")
        if session_type == "train":
            if "session_T" in session_split.keys():
                return session_split["session_T"]
            keys = list(session_split.keys())
            return session_split[keys[0]]
        if "session_E" in session_split.keys():
            return session_split["session_E"]
        keys = list(session_split.keys())
        return session_split[keys[1]]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        splitted_ds = self.dataset.split("session")
        if "session_T" in splitted_ds.keys():
            train_dataset, test_dataset = splitted_ds["session_T"], splitted_ds["session_E"]
        elif "0" in splitted_ds.keys():
            train_dataset, test_dataset = splitted_ds["0"], splitted_ds["1"]
        elif "0train" in splitted_ds.keys():
            train_dataset, test_dataset = splitted_ds["0train"], splitted_ds["1test"]
        else:
            keys = list(splitted_ds.keys())
            print(f"Available session keys: {keys}")
            train_dataset, test_dataset = splitted_ds[keys[0]], splitted_ds[keys[1]]

        x_train, y_train = self._load_data_from_dataset(train_dataset)
        x_test, y_test = self._load_data_from_dataset(test_dataset)

        if self.preprocessing_dict.get("ea", False):
            x_train, x_test = self._align_with_train_reference(x_train, x_train, x_test)

        if self.preprocessing_dict["z_scale"]:
            x_train, x_test = BaseDataModule._z_scale(x_train, x_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(x_train, y_train)
        self.test_dataset = BaseDataModule._make_tensor_dataset(x_test, y_test)


class BCICIV2aLOSO(BCICIV2a):
    """Leave-One-Subject-Out mode for BCI Competition IV 2a dataset."""

    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(
            subject_ids=self.all_subject_ids,
            dataset="2a",
            preprocessing_dict=self.preprocessing_dict,
        )

    @staticmethod
    def _load_data_from_run(run):
        if hasattr(run, "windows"):
            return run.windows.load_data()._data
        if hasattr(run, "_data") and run._data is not None:
            return run._data
        return np.array([run[i][0] for i in range(len(run))])

    def _subject_train_test_arrays(self, splitted_ds, subj_id: int):
        subj = splitted_ds[str(subj_id)]
        train_ds = self._get_session_split(subj, "train")
        test_ds = self._get_session_split(subj, "test")

        x_train = np.concatenate([self._load_data_from_run(run) for run in train_ds.datasets], axis=0)
        y_train = np.concatenate([run.y for run in train_ds.datasets], axis=0)
        x_test = np.concatenate([self._load_data_from_run(run) for run in test_ds.datasets], axis=0)
        y_test = np.concatenate([run.y for run in test_ds.datasets], axis=0)
        return x_train, y_train, x_test, y_test

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        splitted_ds = self.dataset.split("subject")
        train_subjects = [subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]

        src_train_xs, src_train_ys = [], []
        src_val_xs, src_val_ys = [], []

        for subj_id in train_subjects:
            x_tr, y_tr, x_te, y_te = self._subject_train_test_arrays(splitted_ds, subj_id)
            if self.preprocessing_dict.get("ea", False):
                x_tr, x_te = self._align_with_train_reference(x_tr, x_tr, x_te)
            src_train_xs.append(x_tr)
            src_train_ys.append(y_tr)
            src_val_xs.append(x_te)
            src_val_ys.append(y_te)

        # Target subject test set should use target-train reference for EA (no test leakage).
        x_tgt_tr, _y_tgt_tr, x_test, y_test = self._subject_train_test_arrays(splitted_ds, self.subject_id)
        if self.preprocessing_dict.get("ea", False):
            x_test = self._align_with_train_reference(x_tgt_tr, x_test)[0]

        x_train = np.concatenate(src_train_xs, axis=0)
        y_train = np.concatenate(src_train_ys, axis=0)
        x_val = np.concatenate(src_val_xs, axis=0)
        y_val = np.concatenate(src_val_ys, axis=0)

        if self.preprocessing_dict["z_scale"]:
            x_train, x_val, x_test = BaseDataModule._z_scale_tvt(x_train, x_val, x_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(x_train, y_train)
        self.val_dataset = BaseDataModule._make_tensor_dataset(x_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(x_test, y_test)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=0,
            pin_memory=True,
        )


class CombinedSourceTargetDataset(torch.utils.data.Dataset):
    """
    Combine source and target datasets for CDAN training.

    Each item is ((x_src, y_src), (x_tgt, y_tgt)).
    """

    def __init__(self, source_dataset: TensorDataset, target_dataset: TensorDataset, use_interaug: bool = False):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_len = len(source_dataset)
        self.target_len = len(target_dataset)
        self.use_interaug = use_interaug

    def __len__(self):
        return max(self.source_len, self.target_len)

    def __getitem__(self, idx):
        src_idx = idx % self.source_len
        x_src, y_src = self.source_dataset[src_idx]

        tgt_idx = idx % self.target_len
        x_tgt, y_tgt = self.target_dataset[tgt_idx]

        return (x_src, y_src), (x_tgt, y_tgt)


class CDANCollate:
    """Pickle-friendly collate class for Windows multiprocessing."""

    def __init__(self, use_interaug=False):
        self.use_interaug = use_interaug

    def __call__(self, batch):
        from utils.interaug import interaug

        src_xs, src_ys, tgt_xs, tgt_ys = [], [], [], []
        for (x_src, y_src), (x_tgt, y_tgt) in batch:
            src_xs.append(x_src)
            src_ys.append(y_src)
            tgt_xs.append(x_tgt)
            tgt_ys.append(y_tgt)

        x_src = torch.stack(src_xs)
        y_src = torch.tensor([y.item() if hasattr(y, "item") else y for y in src_ys], dtype=torch.long)
        x_tgt = torch.stack(tgt_xs)
        y_tgt = torch.tensor([y.item() if hasattr(y, "item") else y for y in tgt_ys], dtype=torch.long)

        if self.use_interaug:
            x_src, y_src = interaug([x_src, y_src])

        return (x_src, y_src), (x_tgt, y_tgt)


class BCICIV2aLOSO_CDAN(BCICIV2aLOSO):
    """
    CDAN LOSO DataModule with optional EA preprocessing.

    Source domain:
        other subjects' train data (labeled)
    Target domain:
        target subject train data (used unlabeled during adaptation)
    Test:
        target subject test data
    """

    target_train_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)

    def _check_ea_gate(self, splitted_ds) -> bool:
        """
        Adaptive EA hard gating based on log-determinant 1-sigma rule.

        If the target subject's log|det(R_bar)| < mu - sigma (computed across
        all subjects), EA is disabled for the entire fold to avoid noise
        amplification on collapsed covariance manifolds.

        Returns:
            True if EA should be used, False if gated (disabled).
        """
        from utils.ea_gate import compute_log_det, should_disable_ea

        all_log_dets = []
        for subj_id in self.all_subject_ids:
            x_tr, _, _, _ = self._subject_train_test_arrays(splitted_ds, subj_id)
            all_log_dets.append(compute_log_det(x_tr))

        target_log_det = all_log_dets[self.subject_id - 1]  # subject_id is 1-based
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
        train_subjects = [subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]

        # Determine effective EA flag (may be overridden by gate)
        use_ea = self.preprocessing_dict.get("ea", False)
        if use_ea and self.preprocessing_dict.get("ea_gate", False):
            use_ea = self._check_ea_gate(splitted_ds)

        src_train_xs, src_train_ys = [], []
        src_val_xs, src_val_ys = [], []

        for subj_id in train_subjects:
            x_tr, y_tr, x_te, y_te = self._subject_train_test_arrays(splitted_ds, subj_id)
            if use_ea:
                x_tr, x_te = self._align_with_train_reference(x_tr, x_tr, x_te)
            src_train_xs.append(x_tr)
            src_train_ys.append(y_tr)
            src_val_xs.append(x_te)
            src_val_ys.append(y_te)

        x_tgt_train, y_tgt_train, x_test, y_test = self._subject_train_test_arrays(splitted_ds, self.subject_id)
        if use_ea:
            x_tgt_train, x_test = self._align_with_train_reference(x_tgt_train, x_tgt_train, x_test)

        x_src = np.concatenate(src_train_xs, axis=0)
        y_src = np.concatenate(src_train_ys, axis=0)
        x_val = np.concatenate(src_val_xs, axis=0)
        y_val = np.concatenate(src_val_ys, axis=0)

        if self.preprocessing_dict["z_scale"]:
            x_src, x_val, x_tgt_train, x_test = self._z_scale_cdan(x_src, x_val, x_tgt_train, x_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(x_src, y_src)
        self.val_dataset = BaseDataModule._make_tensor_dataset(x_val, y_val)
        self.target_train_dataset = BaseDataModule._make_tensor_dataset(x_tgt_train, y_tgt_train)
        self.test_dataset = BaseDataModule._make_tensor_dataset(x_test, y_test)

        self.combined_train_dataset = CombinedSourceTargetDataset(
            self.train_dataset,
            self.target_train_dataset,
            use_interaug=self.preprocessing_dict.get("interaug", False),
        )

    @staticmethod
    def _z_scale_cdan(x_src, x_val, x_tgt_train, x_test):
        s, c, t = x_src.shape
        x_src_2d = x_src.transpose(1, 0, 2).reshape(c, -1).T
        x_val_2d = x_val.transpose(1, 0, 2).reshape(c, -1).T
        x_tgt_train_2d = x_tgt_train.transpose(1, 0, 2).reshape(c, -1).T
        x_test_2d = x_test.transpose(1, 0, 2).reshape(c, -1).T

        sc = StandardScaler().fit(x_src_2d)

        x_src = sc.transform(x_src_2d).T.reshape(c, s, t).transpose(1, 0, 2)
        x_val = sc.transform(x_val_2d).T.reshape(c, x_val.shape[0], t).transpose(1, 0, 2)
        x_tgt_train = sc.transform(x_tgt_train_2d).T.reshape(c, x_tgt_train.shape[0], t).transpose(1, 0, 2)
        x_test = sc.transform(x_test_2d).T.reshape(c, x_test.shape[0], t).transpose(1, 0, 2)

        return x_src, x_val, x_tgt_train, x_test

    def train_dataloader(self) -> DataLoader:
        num_workers = 0 if os.name == "nt" else 4
        return DataLoader(
            self.combined_train_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=CDANCollate(use_interaug=self.preprocessing_dict.get("interaug", False)),
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
