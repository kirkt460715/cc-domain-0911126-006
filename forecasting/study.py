"""研究驱动：从点时数据集推导向前折与保留段时间界线。"""

from forecasting.dataset import build_dataset
from forecasting.training import run_study


def derive_boundaries(samples, fold_weeks=4, holdout_weeks=4, test_weeks=2):
    """按客户轮转块对齐切出向前折、校准段与最终测试段。

    布局（时间从早到晚）：
      向前折对（训练块 -> 验证块）依次排开 -> 校准段 -> 最终测试段

    合成研究的训练客户每 fold_weeks 在两个互斥子集间轮转，因此每个向前
    折的训练块与紧邻验证块客户天然不相交；折与折之间按时间向前推进，后折
    不使用任何晚于其训练块的信息。同一客户若仍跨越某条界线，由 splits 的
    剔除逻辑删去验证侧记录并记录代价。校准段为 4 周、测试段为 2 周，与
    客户队列进入时间对齐。
    """
    timestamps = sorted(sample['request_ts'] for sample in samples)
    week = 7 * 86400
    block = fold_weeks * week
    # 以最早请求时间为相位，而不是向前取整到自然周边界，保证折块与数据
    # 生成器的客户轮转块同相位；真实数据上退化为从首样本起算的时间块。
    origin = timestamps[0]
    last_week_start = (timestamps[-1] // week) * week
    test_start = last_week_start - (test_weeks - 1) * week
    calibration_start = test_start - holdout_weeks * week

    folds = []
    cursor = origin
    fold_index = 0
    # 非重叠相邻折：每折消耗两个块（训练 + 验证）。
    while cursor + 2 * block <= calibration_start:
        folds.append({
            'name': f'fold_{fold_index:02d}',
            'train_start': cursor,
            'train_end': cursor + block,
            'valid_start': cursor + block,
            'valid_end': cursor + 2 * block,
        })
        cursor += 2 * block
        fold_index += 1
    holdout_splits = {
        'train_end': calibration_start,
        'calibration_start': calibration_start,
        'calibration_end': test_start,
        'test_start': test_start,
        'test_end': last_week_start + test_weeks * week,
    }
    return folds, holdout_splits


def run(dataset, label_effective_at, label_knowledge_at, coverage=0.9,
        fold_weeks=4, holdout_weeks=4, test_weeks=2, period_days=14):
    point_in_time = build_dataset(dataset, label_effective_at, label_knowledge_at)
    folds, split_kwargs = derive_boundaries(
        point_in_time['samples'], fold_weeks, holdout_weeks, test_weeks)
    holdout = {
        'label_effective_at': label_effective_at,
        'label_knowledge_at': label_knowledge_at,
        'splits': split_kwargs,
    }
    return run_study(dataset, folds, holdout, coverage=coverage,
                     period_days=period_days)
