import mne
import numpy as np
import warnings

# 1. 屏蔽所有无关的硬件滤波器警告和旧函数警告
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_edf_channels(file_path):
    """
    仅读取 EDF 头部信息，快速获取所有通道名称
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        return raw.ch_names
    except Exception as e:
        print(f"解析通道失败: {e}")
        return []


def process_uploaded_edf(file_path, target_channel="EEG Fpz-Cz"):
    """
    读取 EDF 文件，并严格按照训练时的预处理逻辑进行处理
    """
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # 2. 通道选择
    if target_channel in raw.ch_names:
        raw.pick(picks=[target_channel])
        actual_channel = target_channel
    else:
        actual_channel = raw.ch_names[0]
        raw.pick(picks=[actual_channel])

    # 3. Zero-phase FIR 带通滤波 (0.3-35.0 Hz)
    raw.filter(l_freq=0.3, h_freq=35.0, method='fir', phase='zero', verbose=False)

    # 4. 动态重采样至 100 Hz
    if raw.info['sfreq'] != 100.0:
        raw.resample(100.0)

    # 5. 提取连续的 1D 脑电数据
    signal_data = raw.get_data()[0]

    # 6. 按 30 秒 (3000 个数据点) 进行分段对齐
    window_size = 3000
    total_points = len(signal_data)
    num_batches = total_points // window_size

    signal_trimmed = signal_data[:num_batches * window_size]

    # 7. Subject-wise Z-score 标准化
    mean_val = np.mean(signal_trimmed)
    std_val = np.std(signal_trimmed)
    if std_val != 0:
        signal_normalized = (signal_trimmed - mean_val) / std_val
    else:
        signal_normalized = signal_trimmed

    # 8. 重塑数组维度 [batch, 1, 3000]
    eeg_batched = signal_normalized.reshape(num_batches, 1, window_size)

    # 9. 转换为 NumPy float32 数组 (彻底摆脱 PyTorch)
    numpy_input = eeg_batched.astype(np.float32)

    return numpy_input, signal_trimmed, actual_channel, num_batches


# ==========================================
# 真实标签解析与首尾截断功能
# ==========================================
STAGE_MAPPING = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}


def extract_true_labels(label_file_path, num_batches):
    """
    解析真实的 Hypnogram 注释文件，并映射为 30s 一个 epoch 的对应标签
    """
    try:
        annotations = mne.read_annotations(label_file_path)
        labels = np.full(num_batches, 5, dtype=np.int32)

        for ann in annotations:
            onset_sec = ann['onset']
            duration_sec = ann['duration']
            description = ann['description']

            if description in STAGE_MAPPING:
                label = STAGE_MAPPING[description]
                start_epoch = int(onset_sec // 30)
                end_epoch = int((onset_sec + duration_sec) // 30)

                end_epoch = min(end_epoch, num_batches)
                if start_epoch < num_batches:
                    labels[start_epoch:end_epoch] = label

        return labels
    except Exception as e:
        print(f"标签解析失败: {e}")
        return None


# def truncate_wake_periods(signal_trimmed, predictions, true_labels=None, keep_wake_mins=30, epoch_sec=30, sfreq=100):
#     """
#     根据标签截断首尾过长的 Wake 阶段，仅保留实际睡眠前后各 keep_wake_mins 分钟的数据。
#     """
#     ref_labels = true_labels if true_labels is not None else predictions
#     is_sleep = ((ref_labels != 0) & (ref_labels != 5)).astype(int)
#
#     # 直接定位实际睡眠的起止点
#     non_wake_indices = np.where(is_sleep)[0]
#     if len(non_wake_indices) == 0:
#         # 彻底没睡，不执行截断
#         return signal_trimmed, predictions, true_labels, len(predictions)
#
#     first_sleep_idx = non_wake_indices[0]
#     last_sleep_idx = non_wake_indices[-1]
#
#     epochs_per_min = 60 // epoch_sec
#     padding = keep_wake_mins * epochs_per_min
#
#     # 计算保留的起止点索引，防止越界
#     start_idx = max(0, first_sleep_idx - padding)
#     end_idx = min(len(ref_labels) - 1, last_sleep_idx + padding)
#
#     valid_range = np.arange(start_idx, end_idx + 1)
#
#     # 1. 截断标签数组
#     trunc_predictions = predictions[valid_range]
#     trunc_true_labels = true_labels[valid_range] if true_labels is not None else None
#
#     # 2. 截断一维原始 EEG 信号
#     pts_per_epoch = int(sfreq * epoch_sec)
#     signal_start_pt = start_idx * pts_per_epoch
#     signal_end_pt = (end_idx + 1) * pts_per_epoch
#     trunc_signal = signal_trimmed[signal_start_pt:signal_end_pt]
#
#     num_batches = len(valid_range)
#
#     return trunc_signal, trunc_predictions, trunc_true_labels, num_batches



def truncate_wake_periods(signal_trimmed, predictions, true_labels=None, keep_wake_mins=30, epoch_sec=30, sfreq=100,
                          min_consecutive_sleep=5):
    """
    根据标签截断首尾过长的 Wake 阶段，仅保留实际睡眠前后各 keep_wake_mins 分钟的数据。
    【防抖升级】：要求连续 min_consecutive_sleep 个 epoch 为非 Wake 状态，才确认真实的入睡/苏醒边界。
    """
    # 确定用于寻找边界的参考标签 (0: Wake, 5: Unknown)
    ref_labels = true_labels if true_labels is not None else predictions

    # 生成 0 和 1 的布尔数组：1 代表睡眠 (非 Wake 且非 Unknown)，0 代表未睡
    is_sleep = ((ref_labels != 0) & (ref_labels != 5)).astype(int)

    # 使用一维卷积 (滑动窗口) 寻找连续的睡眠阶段
    # np.ones(5) 会在序列上滑动求和。如果某连续 5 个点都是 1，和就是 5。
    window = np.ones(min_consecutive_sleep, dtype=int)
    sleep_blocks = np.convolve(is_sleep, window, mode='valid')

    # 找到所有连续睡眠块的起始索引
    valid_starts = np.where(sleep_blocks == min_consecutive_sleep)[0]

    if len(valid_starts) > 0:
        # 找到了真正的连续睡眠期
        first_sleep_idx = valid_starts[0]
        # 最后一个连续睡眠块的结束位置
        last_sleep_idx = valid_starts[-1] + min_consecutive_sleep - 1
    else:
        # 容错降级：如果没有连续 5 个 epoch 的睡眠（极其罕见或完全失眠），退化为寻找单点
        non_wake_indices = np.where(is_sleep)[0]
        if len(non_wake_indices) == 0:
            # 彻底没睡，不执行截断
            return signal_trimmed, predictions, true_labels, len(predictions)
        first_sleep_idx = non_wake_indices[0]
        last_sleep_idx = non_wake_indices[-1]

    epochs_per_min = 60 // epoch_sec
    padding = keep_wake_mins * epochs_per_min

    # 计算保留的起止点索引，防止越界
    start_idx = max(0, first_sleep_idx - padding)
    end_idx = min(len(ref_labels) - 1, last_sleep_idx + padding)

    valid_range = np.arange(start_idx, end_idx + 1)

    # 1. 截断标签数组
    trunc_predictions = predictions[valid_range]
    trunc_true_labels = true_labels[valid_range] if true_labels is not None else None

    # 2. 截断一维原始 EEG 信号
    pts_per_epoch = int(sfreq * epoch_sec)
    signal_start_pt = start_idx * pts_per_epoch
    signal_end_pt = (end_idx + 1) * pts_per_epoch
    trunc_signal = signal_trimmed[signal_start_pt:signal_end_pt]

    num_batches = len(valid_range)

    return trunc_signal, trunc_predictions, trunc_true_labels, num_batches