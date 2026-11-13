"""时间向前（forward-chaining）拆分与客户分组约束。

规则：

* 训练折按请求时间向前滚动；超参数只能在“训练 -> 验证”的时间折内选择；
* 同一客户不得同时出现在同一折的训练与验证两侧；
* 当客户的请求跨越训练/验证时间界线时，显式删去验证侧记录，并在折报告中
  记录删去的 visit、客户数与样本代价；
* 校准集与最终测试集在时间上更晚且互不复用，同样按客户去重。
"""

from dispatch_log.schema import canonical, digest
import json


def _clients(samples):
    return {sample['client'] for sample in samples}


def assign_blocks(samples, boundaries):
    """按 request_ts 与时间界线把样本分到命名时间块。

    boundaries: [(block_name, start_ts, end_ts), ...]，区间左闭右开。
    """
    blocks = {name: [] for name, _, _ in boundaries}
    for sample in sorted(samples, key=lambda row: (row['request_ts'], row['visit'])):
        for name, start, end in boundaries:
            if start <= sample['request_ts'] < end:
                blocks[name].append(sample)
                break
    return blocks


def _purge_client_overlap(train, validation):
    """删除验证侧与训练侧客户重叠的记录，返回 (purged_validation, report)。"""
    train_clients = _clients(train)
    kept, dropped = [], []
    for sample in validation:
        if sample['client'] in train_clients:
            dropped.append(sample)
        else:
            kept.append(sample)
    report = {
        'dropped_visits': [sample['visit'] for sample in dropped],
        'dropped_clients': sorted({sample['client'] for sample in dropped}),
        'dropped_count': len(dropped),
        'cost_note': 'validation records whose client also appears in the training side '
                     'of the same fold are removed to prevent same-client leakage',
    }
    return kept, report


def make_folds(samples, fold_boundaries):
    """构造向前滚动折。

    fold_boundaries: [{'name', 'train_start', 'train_end', 'valid_start',
    'valid_end'}, ...]，每折训练侧严格早于验证侧。
    """
    folds = []
    for boundary in fold_boundaries:
        train = [row for row in samples
                 if boundary['train_start'] <= row['request_ts'] < boundary['train_end']]
        validation = [row for row in samples
                      if boundary['valid_start'] <= row['request_ts'] < boundary['valid_end']]
        purged, report = _purge_client_overlap(train, validation)
        folds.append({
            'name': boundary['name'],
            'train_visits': [row['visit'] for row in train],
            'validation_visits': [row['visit'] for row in purged],
            'train_clients': sorted(_clients(train)),
            'validation_clients': sorted(_clients(purged)),
            'purge': report,
            'train_start': boundary['train_start'],
            'train_end': boundary['train_end'],
            'valid_start': boundary['valid_start'],
            'valid_end': boundary['valid_end'],
        })
    return folds


def make_holdout_splits(samples, train_end, calibration_start, calibration_end,
                        test_start, test_end):
    """训练 / 校准 / 最终测试三段，时间递增，客户互不复用。

    若校准或测试客户在训练侧出现，记录从对应侧删去的样本及代价；同时
    校准与测试两侧也互相去重（测试侧优先保留更晚数据）。
    """
    train = [row for row in samples if row['request_ts'] < train_end]
    calibration = [row for row in samples
                   if calibration_start <= row['request_ts'] < calibration_end]
    test = [row for row in samples
            if test_start <= row['request_ts'] < test_end]
    train_clients = _clients(train)

    cal_kept, cal_report = _purge_client_overlap(train, calibration)
    test_kept, test_report = _purge_client_overlap(train, test)

    # 校准与测试互不复用：测试侧若出现校准客户，从测试侧删除（更晚的最终
    # 测试保留独立客户更重要；删除代价显式记录）。
    cal_clients = _clients(cal_kept)
    test_final, cross_report = _purge_client_overlap(
        [{'client': c} for c in cal_clients], test_kept)
    splits = {
        'train': [row['visit'] for row in train],
        'calibration': [row['visit'] for row in cal_kept],
        'test': [row['visit'] for row in test_final],
        'train_clients': sorted(train_clients),
        'calibration_clients': sorted(_clients(cal_kept)),
        'test_clients': sorted(_clients(test_final)),
        'purge_calibration': cal_report,
        'purge_test_vs_train': test_report,
        'purge_test_vs_calibration': cross_report,
        'time_order': ['train', 'calibration', 'test'],
    }
    body = json.loads(canonical(splits))
    return {**body, 'split_digest': digest(body)}


def new_client_mask(samples, split_client_sets):
    """标记样本客户在训练侧从未出现（即新客户）。"""
    train_clients = split_client_sets
    return {sample['visit']: sample['client'] not in train_clients
            for sample in samples}
