from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import torch

from .base import BaseDataModule, make_collate_fn
from utils.load_bcic4 import load_bcic4
from sklearn.model_selection import train_test_split
import os


class BCICIV2a(BaseDataModule):
    """Subject-Dependent (Sub-Dep) mode for BCI Competition IV 2a dataset."""
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4 
    
    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("session")
        # Handle different session naming conventions in different braindecode versions
        if "session_T" in splitted_ds.keys():
            train_dataset, test_dataset = splitted_ds["session_T"], splitted_ds["session_E"]
        elif "0" in splitted_ds.keys():
            train_dataset, test_dataset = splitted_ds["0"], splitted_ds["1"]
        elif "0train" in splitted_ds.keys():
            train_dataset, test_dataset = splitted_ds["0train"], splitted_ds["1test"]
        else:
            # Try to find the keys
            keys = list(splitted_ds.keys())
            print(f"Available session keys: {keys}")
            train_dataset, test_dataset = splitted_ds[keys[0]], splitted_ds[keys[1]]

        # load the data - handle different braindecode versions
        def load_data_from_dataset(ds):
            X_list, y_list = [], []
            for run in ds.datasets:
                # Try different ways to access data depending on braindecode version
                if hasattr(run, 'windows'):
                    data = run.windows.load_data()._data
                elif hasattr(run, '_data') and run._data is not None:
                    data = run._data
                else:
                    # For newer braindecode versions, access data directly
                    data = np.array([run[i][0] for i in range(len(run))])
                X_list.append(data)
                y_list.append(run.y)
            return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
        
        X, y = load_data_from_dataset(train_dataset)
        X_test, y_test = load_data_from_dataset(test_dataset)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)                                                                


class BCICIV2aLOSO(BCICIV2a):
    """Leave-One-Subject-Out (LOSO) mode for BCI Competition IV 2a dataset."""
    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=self.all_subject_ids, dataset="2a",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]
        
        # Helper function to get session split with version compatibility
        def get_session_split(ds, session_type):
            session_split = ds.split("session")
            if session_type == "train":
                if "session_T" in session_split.keys():
                    return session_split["session_T"]
                keys = list(session_split.keys())
                return session_split[keys[0]]
            else:  # test/eval
                if "session_E" in session_split.keys():
                    return session_split["session_E"]
                keys = list(session_split.keys())
                return session_split[keys[1]]
        
        train_datasets = [get_session_split(splitted_ds[str(subj_id)], "train")
                            for subj_id in train_subjects]
        val_datasets = [get_session_split(splitted_ds[str(subj_id)], "test")
                        for subj_id in train_subjects]
        test_dataset = get_session_split(splitted_ds[str(self.subject_id)], "test")

        # Helper function to load data from dataset with version compatibility
        def load_data_from_run(run):
            if hasattr(run, 'windows'):
                return run.windows.load_data()._data
            elif hasattr(run, '_data') and run._data is not None:
                return run._data
            else:
                return np.array([run[i][0] for i in range(len(run))])

        # load the data
        X = np.concatenate([load_data_from_run(run) for train_dataset in
                            train_datasets for run in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for train_dataset in train_datasets for run in
                            train_dataset.datasets], axis=0)
        X_val = np.concatenate([load_data_from_run(run) for val_dataset in
                            val_datasets for run in val_dataset.datasets], axis=0)
        y_val = np.concatenate([run.y for val_dataset in val_datasets for run in
                            val_dataset.datasets], axis=0)
        X_test = np.concatenate([load_data_from_run(run) for run in test_dataset.datasets],
                                axis=0)
        y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=0,  # Set to 0 for Windows compatibility
                          pin_memory=True,
                        )


# ============================================================================
# CDAN DataModule: 支持源域和目标域同时加载
# ============================================================================

class CombinedSourceTargetDataset(torch.utils.data.Dataset):
    """
    组合源域和目标域数据的 Dataset
    
    每次迭代返回 ((x_src, y_src), (x_tgt, y_tgt))
    其中 y_tgt 在训练时不使用（无监督域适应）
    
    interaug在这里执行，由worker进程并行处理，提高GPU利用率
    """
    def __init__(self, source_dataset: TensorDataset, target_dataset: TensorDataset, use_interaug: bool = False):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_len = len(source_dataset)
        self.target_len = len(target_dataset)
        self.use_interaug = use_interaug
    
    def __len__(self):
        # 返回较大数据集的长度，确保所有数据都被使用
        return max(self.source_len, self.target_len)
    
    def __getitem__(self, idx):
        # 源域：循环使用
        src_idx = idx % self.source_len
        x_src, y_src = self.source_dataset[src_idx]
        
        # 目标域：循环使用
        tgt_idx = idx % self.target_len
        x_tgt, y_tgt = self.target_dataset[tgt_idx]
        
        # 注意：interaug设计用于batch级别，不适合在单样本级别使用
        # 因此保留在collate_fn中执行
        
        return (x_src, y_src), (x_tgt, y_tgt)


class CDANCollate:
    """
    CDAN 专用的 collate 类（可序列化）
    
    处理 ((x_src, y_src), (x_tgt, y_tgt)) 格式的 batch
    使用类而不是闭包函数，以便在Windows上使用multiprocessing时可以被pickle序列化
    
    interaug在主进程中执行（batch级别操作，无法在worker中单样本执行）
    """
    def __init__(self, use_interaug=False):
        self.use_interaug = use_interaug
    
    def __call__(self, batch):
        from utils.interaug import interaug
        
        # batch 是 [((x_src, y_src), (x_tgt, y_tgt)), ...] 的列表
        src_xs, src_ys, tgt_xs, tgt_ys = [], [], [], []
        
        for (x_src, y_src), (x_tgt, y_tgt) in batch:
            src_xs.append(x_src)
            src_ys.append(y_src)
            tgt_xs.append(x_tgt)
            tgt_ys.append(y_tgt)
        
        x_src = torch.stack(src_xs)
        y_src = torch.tensor([y.item() if hasattr(y, 'item') else y for y in src_ys], 
                            dtype=torch.long)
        x_tgt = torch.stack(tgt_xs)
        y_tgt = torch.tensor([y.item() if hasattr(y, 'item') else y for y in tgt_ys], 
                            dtype=torch.long)
        
        # 对源域数据应用 interaug（如果启用）
        if self.use_interaug:
            x_src, y_src = interaug([x_src, y_src])
        
        return (x_src, y_src), (x_tgt, y_tgt)


class BCICIV2aLOSO_CDAN(BCICIV2aLOSO):
    """
    CDAN 专用的 LOSO DataModule
    
    在 LOSO 场景下：
    - 源域：其他被试的训练数据（有标签）
    - 目标域：测试被试的训练数据（无标签，用于域适应）
    - 测试：测试被试的测试数据
    
    训练时同时加载源域和目标域数据，用于域对抗训练
    """
    target_train_dataset = None  # 目标域训练数据（无标签）
    
    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        
        # split the data
        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]
        
        # Helper function to get session split with version compatibility
        def get_session_split(ds, session_type):
            session_split = ds.split("session")
            if session_type == "train":
                if "session_T" in session_split.keys():
                    return session_split["session_T"]
                keys = list(session_split.keys())
                return session_split[keys[0]]
            else:  # test/eval
                if "session_E" in session_split.keys():
                    return session_split["session_E"]
                keys = list(session_split.keys())
                return session_split[keys[1]]
        
        # 源域数据：其他被试的训练和验证数据
        train_datasets = [get_session_split(splitted_ds[str(subj_id)], "train")
                            for subj_id in train_subjects]
        val_datasets = [get_session_split(splitted_ds[str(subj_id)], "test")
                        for subj_id in train_subjects]
        
        # 目标域数据：测试被试的训练数据（用于域适应）
        target_train_dataset = get_session_split(splitted_ds[str(self.subject_id)], "train")
        
        # 测试数据：测试被试的测试数据
        test_dataset = get_session_split(splitted_ds[str(self.subject_id)], "test")

        # Helper function to load data from dataset with version compatibility
        def load_data_from_run(run):
            if hasattr(run, 'windows'):
                return run.windows.load_data()._data
            elif hasattr(run, '_data') and run._data is not None:
                return run._data
            else:
                return np.array([run[i][0] for i in range(len(run))])

        # 加载源域数据
        X_src = np.concatenate([load_data_from_run(run) for train_dataset in
                            train_datasets for run in train_dataset.datasets], axis=0)
        y_src = np.concatenate([run.y for train_dataset in train_datasets for run in
                            train_dataset.datasets], axis=0)
        
        # 加载源域验证数据
        X_val = np.concatenate([load_data_from_run(run) for val_dataset in
                            val_datasets for run in val_dataset.datasets], axis=0)
        y_val = np.concatenate([run.y for val_dataset in val_datasets for run in
                            val_dataset.datasets], axis=0)
        
        # 加载目标域训练数据（用于域适应）
        X_tgt_train = np.concatenate([load_data_from_run(run) for run in 
                                      target_train_dataset.datasets], axis=0)
        y_tgt_train = np.concatenate([run.y for run in target_train_dataset.datasets], axis=0)
        
        # 加载测试数据
        X_test = np.concatenate([load_data_from_run(run) for run in test_dataset.datasets],
                                axis=0)
        y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)

        # 标准化数据
        # 注意：使用源域数据拟合 scaler，然后应用到所有数据
        if self.preprocessing_dict["z_scale"]:
            X_src, X_val, X_tgt_train, X_test = self._z_scale_cdan(
                X_src, X_val, X_tgt_train, X_test)

        # 创建数据集
        self.train_dataset = BaseDataModule._make_tensor_dataset(X_src, y_src)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.target_train_dataset = BaseDataModule._make_tensor_dataset(X_tgt_train, y_tgt_train)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
        
        # 创建组合数据集（用于 CDAN 训练）
        # interaug在Dataset中由worker进程并行执行
        self.combined_train_dataset = CombinedSourceTargetDataset(
            self.train_dataset, 
            self.target_train_dataset,
            use_interaug=self.preprocessing_dict.get("interaug", False)
        )

    @staticmethod
    def _z_scale_cdan(X_src, X_val, X_tgt_train, X_test):
        """
        CDAN 专用的标准化方法
        
        使用源域数据拟合 scaler，然后应用到所有数据
        """
        s, c, t = X_src.shape
        X_src_2d = X_src.transpose(1, 0, 2).reshape(c, -1).T
        X_val_2d = X_val.transpose(1, 0, 2).reshape(c, -1).T
        X_tgt_train_2d = X_tgt_train.transpose(1, 0, 2).reshape(c, -1).T
        X_test_2d = X_test.transpose(1, 0, 2).reshape(c, -1).T

        # 使用源域数据拟合 scaler
        sc = StandardScaler().fit(X_src_2d)
        
        # 应用到所有数据
        X_src = sc.transform(X_src_2d).T.reshape(c, s, t).transpose(1, 0, 2)
        X_val = sc.transform(X_val_2d).T.reshape(c, X_val.shape[0], t).transpose(1, 0, 2)
        X_tgt_train = sc.transform(X_tgt_train_2d).T.reshape(c, X_tgt_train.shape[0], t).transpose(1, 0, 2)
        X_test = sc.transform(X_test_2d).T.reshape(c, X_test.shape[0], t).transpose(1, 0, 2)
        
        return X_src, X_val, X_tgt_train, X_test

    def train_dataloader(self) -> DataLoader:
        """返回 CDAN 训练用的 DataLoader"""
        num_workers = 0 if os.name == "nt" else 4
        return DataLoader(
            self.combined_train_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            shuffle=True,
            num_workers=num_workers,  # Windows 下禁用多进程，避免共享内存映射错误
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=CDANCollate(use_interaug=self.preprocessing_dict.get("interaug", False))
        )

    def val_dataloader(self) -> DataLoader:
        """返回验证用的 DataLoader（使用源域验证数据）"""
        num_workers = 0 if os.name == "nt" else 4
        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    def test_dataloader(self) -> DataLoader:
        """返回测试用的 DataLoader"""
        num_workers = 0 if os.name == "nt" else 4
        return DataLoader(
            self.test_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    
    def target_train_dataloader(self) -> DataLoader:
        """返回目标域训练数据的 DataLoader（可选，用于其他用途）"""
        num_workers = 0 if os.name == "nt" else 4
        return DataLoader(
            self.target_train_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
