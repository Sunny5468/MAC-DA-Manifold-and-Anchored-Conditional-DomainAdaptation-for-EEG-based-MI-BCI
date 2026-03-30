from typing import Dict, List

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)


def scale(data, factor):
    """Scale data by a factor."""
    return data * factor


# BCIC2a 的 22 通道名（基于扩展 10-20 系统）
BCIC2A_22_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

# Weibo2014 选取的 4 类 event_id（过滤掉复合动作和休息态）
WEIBO2014_EVENT_ID = {
    "left_hand": 0,
    "right_hand": 1,
    "hands": 2,
    "feet": 3,
}


def load_weibo2014(subject_ids: List[int], preprocessing_dict: Dict,
                   verbose: str = "WARNING"):
    """
    加载 Weibo2014 数据集，执行预处理并返回 windows_dataset。

    流程:
    1. 通过 MOABB 加载原始数据
    2. 剔除 EOG 通道，只保留 EEG
    3. 选取与 BCIC2a 对应的 22 通道
    4. Resample 到 250Hz
    5. 创建 windows，只保留 4 类 MI 事件
    """
    dataset = MOABBDataset("Weibo2014", subject_ids=subject_ids)

    # 先获取实际通道名，做大小写不敏感匹配
    raw_ch_names = dataset.datasets[0].raw.ch_names
    ch_name_map = {ch.lower(): ch for ch in raw_ch_names}

    pick_channels = []
    for ch in BCIC2A_22_CHANNELS:
        matched = ch_name_map.get(ch.lower())
        if matched:
            pick_channels.append(matched)
        else:
            print(f"[WARNING] Channel '{ch}' not found in Weibo2014. Skipping.")

    preprocessors = [
        # 1. 剔除 EOG，只保留 EEG
        Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=verbose),
        # 2. 选取 22 通道
        Preprocessor("pick_channels", ch_names=pick_channels, ordered=True, verbose=verbose),
        # 3. 微伏缩放
        Preprocessor(scale, factor=1e6, apply_on_array=True),
        # 4. Resample 到 250Hz
        Preprocessor("resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose),
    ]

    l_freq = preprocessing_dict.get("low_cut")
    h_freq = preprocessing_dict.get("high_cut")
    if l_freq is not None or h_freq is not None:
        preprocessors.append(
            Preprocessor("filter", l_freq=l_freq, h_freq=h_freq, verbose=verbose)
        )

    preprocess(dataset, preprocessors)

    # 创建 windows，只保留 4 类
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(preprocessing_dict.get("start", 0) * sfreq)
    trial_stop_offset_samples = int(preprocessing_dict.get("stop", 0) * sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        preload=False,
        mapping=WEIBO2014_EVENT_ID,
    )

    return windows_dataset
