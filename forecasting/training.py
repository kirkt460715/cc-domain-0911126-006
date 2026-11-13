"""训练编排：向前折超参选择、校准、最终测试与冻结模型包。

流程严格按时间顺序：

1. 点时数据集（特征截点 = 每次请求的 requested_at，标签来自双截止重放）；
2. 向前滚动折内做超参选择，同客户不跨训练/验证；
3. 互不复用的校准集拟合单侧时长上界；
4. 更晚的最终测试集只评估一次；
5. 冻结模型包包含特征定义、拆分归属、变换参数与原始事件摘要；
   重新加载后可预测新请求，后补事件只能产生新版本。
"""

import hashlib
import json
from pathlib import Path

from dispatch_log.schema import canonical, digest
from forecasting.dataset import build_dataset, feature_spec
from forecasting.models import MedianBaseline, RidgeDurationModel
from forecasting.splits import make_folds, make_holdout_splits
from forecasting.evaluation import (OneSidedUpperBound, PredictionPolicy,
                                    evaluate_by_dimensions, mae)

ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)


def _samples_by_id(samples):
    return {sample['visit']: sample for sample in samples}


def select_hyperparameters(samples, folds, alpha_grid=ALPHA_GRID):
    """只在向前折内选择岭模型 alpha；中位数基线无超参，作为并列对照。"""
    by_id = _samples_by_id(samples)
    fold_reports = []
    aggregate = {alpha: [] for alpha in alpha_grid}
    baseline_errors = []
    for fold in folds:
        train = [by_id[visit] for visit in fold['train_visits']]
        validation = [by_id[visit] for visit in fold['validation_visits']]
        if len(train) < 10 or len(validation) < 3:
            fold_reports.append({'name': fold['name'], 'skipped': 'insufficient_samples',
                                 'purge': fold['purge']})
            continue
        baseline = MedianBaseline().fit(train)
        baseline_pred = {sample['visit']: baseline.predict_seconds(sample['features'])[0]
                         for sample in validation}
        baseline_mae = mae([s['label']['active_seconds'] for s in validation],
                           [baseline_pred[s['visit']] for s in validation])
        baseline_errors.append(baseline_mae)
        alpha_results = {}
        for alpha in alpha_grid:
            model = RidgeDurationModel(alpha=alpha).fit(train)
            predictions = {}
            for sample in validation:
                estimate, status, _ = model.predict_seconds(sample['features'])
                predictions[sample['visit']] = estimate
            score = mae([s['label']['active_seconds'] for s in validation],
                        [predictions[s['visit']] for s in validation])
            alpha_results[alpha] = score
            aggregate[alpha].append(score)
        fold_reports.append({
            'name': fold['name'],
            'skipped': False,
            'train_size': len(train),
            'validation_size': len(validation),
            'baseline_mae_seconds': baseline_mae,
            'ridge_mae_by_alpha': alpha_results,
            'purge': fold['purge'],
        })
    mean_by_alpha = {alpha: sum(values) / len(values)
                     for alpha, values in aggregate.items() if values}
    selected_alpha = min(mean_by_alpha, key=mean_by_alpha.get) if mean_by_alpha else 1.0
    return {
        'folds': fold_reports,
        'mean_validation_mae_by_alpha': mean_by_alpha,
        'selected_alpha': selected_alpha,
        'baseline_mean_validation_mae': (sum(baseline_errors) / len(baseline_errors)
                                         if baseline_errors else None),
        'selection_rule': 'minimum mean forward-fold validation MAE in seconds',
    }


def fit_final_models(samples, split, selected_alpha, coverage=0.9):
    """在训练段拟合两个最终模型，在校准段拟合单侧上界。"""
    by_id = _samples_by_id(samples)
    train = [by_id[visit] for visit in split['train']]
    calibration = [by_id[visit] for visit in split['calibration']]

    ridge = RidgeDurationModel(alpha=selected_alpha).fit(train)
    baseline = MedianBaseline().fit(train)

    calibrated = {}
    for name, model in (('ridge_log1p', ridge), ('median_baseline', baseline)):
        predictions = [model.predict_seconds(sample['features'])[0] for sample in calibration]
        actual = [sample['label']['active_seconds'] for sample in calibration]
        bound = OneSidedUpperBound(coverage).fit(actual, predictions)
        calibrated[name] = bound
    return ridge, baseline, calibrated


def _predict_all(model, samples):
    policy = PredictionPolicy(model)
    predictions, decisions = {}, {}
    for sample in samples:
        decision = policy.decide(sample['features'])
        decisions[sample['visit']] = decision
        predictions[sample['visit']] = decision['prediction_seconds']
    return predictions, decisions


def _upper_all(bound, predictions):
    upper = {}
    for visit, estimate in predictions.items():
        if estimate is None:
            upper[visit] = None
        else:
            upper[visit] = bound.upper_seconds(estimate)
    return upper


def run_study(dataset, fold_boundaries, holdout, coverage=0.9, alpha_grid=ALPHA_GRID,
              period_days=14):
    """端到端研究，返回可冻结的完整结果。"""
    point_in_time = build_dataset(dataset, holdout['label_effective_at'],
                                  holdout['label_knowledge_at'])
    samples = point_in_time['samples']
    split = make_holdout_splits(samples, **holdout['splits'])
    folds = make_folds(samples, fold_boundaries)
    selection = select_hyperparameters(samples, folds, alpha_grid)

    ridge, baseline, calibrated = fit_final_models(
        samples, split, selection['selected_alpha'], coverage)

    by_id = _samples_by_id(samples)
    calibration_samples = [by_id[v] for v in split['calibration']]
    test_samples = [by_id[v] for v in split['test']]
    train_clients = set(split['train_clients'])
    new_client_visits = {sample['visit'] for sample in test_samples
                         if sample['client'] not in train_clients}

    models = {'ridge_log1p': ridge, 'median_baseline': baseline}
    evaluation = {}
    for name, model in models.items():
        cal_pred, cal_decisions = _predict_all(model, calibration_samples)
        test_pred, test_decisions = _predict_all(model, test_samples)
        bound = calibrated[name]
        cal_upper = _upper_all(bound, cal_pred)
        test_upper = _upper_all(bound, test_pred)
        evaluation[name] = {
            'calibration': evaluate_by_dimensions(
                calibration_samples, cal_pred, cal_upper, coverage, period_days),
            'test': evaluate_by_dimensions(
                test_samples, test_pred, test_upper, coverage, period_days,
                new_client_visits=new_client_visits),
            'calibration_decisions': cal_decisions,
            'test_decisions': test_decisions,
            'upper_bound': bound.to_dict(),
        }

    frozen = freeze_package(dataset, point_in_time, split, folds, selection,
                            ridge, baseline, calibrated, evaluation, coverage)
    return {
        'point_in_time': point_in_time,
        'split': split,
        'folds': folds,
        'selection': selection,
        'evaluation': evaluation,
        'package': frozen,
    }


def freeze_package(dataset, point_in_time, split, folds, selection,
                   ridge, baseline, calibrated, evaluation, coverage):
    """构造自描述的冻结模型包（不落盘）。"""
    event_summary = _event_summary(dataset)
    package = {
        'package_version': 1,
        'purpose': 'scheduling_reference_only_not_dispatch_or_worktime_authority',
        'created_from_input_digest': dataset.fingerprint,
        'feature_spec': feature_spec(),
        'label': {
            'target': 'active_seconds_between_started_and_completed_excluding_pauses',
            'effective_at': point_in_time['label_effective_at'],
            'knowledge_at': point_in_time['label_knowledge_at'],
            'exclusion_reason_counts': point_in_time['reason_counts'],
        },
        'split_assignment': {
            'holdout': {key: value for key, value in split.items()
                        if key not in ('split_digest',)},
            'forward_folds': folds,
        },
        'transforms': {
            'target_transform': ridge.to_dict()['transform'],
            'inverse_transform': 'expm1 applied before all scoring; metrics reported in seconds',
            'numeric_standardization': {
                'mean': ridge.numeric_mean,
                'scale': ridge.numeric_scale,
                'support': ridge.numeric_support,
            },
            'category_vocabularies': ridge.categories,
        },
        'models': {
            'ridge_log1p': ridge.to_dict(),
            'median_baseline': baseline.to_dict(),
        },
        'calibration': {name: bound.to_dict() for name, bound in calibrated.items()},
        'hyperparameter_selection': selection,
        'evaluation': evaluation,
        'raw_event_summary': event_summary,
        'coverage_target': coverage,
        'reload_contract': 'load package -> reconstruct models via from_dict -> '
                           'predict features built strictly at request time',
        'versioning_rule': 'events arriving after freeze can only produce a new package version',
    }
    body = json.loads(canonical(package))
    package['package_digest'] = digest(body)
    return package


def _event_summary(dataset):
    """原始事件摘要：计数、修订、void、观测延迟，供审计而不暴露全量行。"""
    from collections import defaultdict
    revisions = defaultdict(list)
    kind_counts = defaultdict(int)
    void_count = 0
    observed_lag = []
    for event in dataset.events:
        revisions[event['id']].append(event['revision'])
        kind_counts[event['kind']] += 1
        if event['void']:
            void_count += 1
        from dispatch_log.schema import instant
        observed_lag.append(instant(event['observed_at'], 'o') -
                            instant(event['effective_at'], 'e'))
    multi_revision = {ident: revs for ident, revs in revisions.items() if len(revs) > 1}
    return {
        'event_count': len(dataset.events),
        'visit_count': len(dataset.visits),
        'kind_counts': dict(sorted(kind_counts.items())),
        'multi_revision_events': multi_revision,
        'void_event_count': void_count,
        'max_observed_lag_seconds': max(observed_lag) if observed_lag else 0,
        'input_fingerprint': dataset.fingerprint,
    }


def save_package(package, directory):
    """原子写盘并附清单校验。"""
    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    import os
    import shutil
    import tempfile
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.forecast-', dir=target.parent))
    try:
        files = {
            'model-package.json': canonical(package),
        }
        manifest = {
            'version': 1,
            'files': {name: hashlib.sha256(content.encode()).hexdigest()
                      for name, content in sorted(files.items())},
            'package_digest': package['package_digest'],
        }
        files['manifest.json'] = canonical(manifest)
        for name, content in files.items():
            (staging / name).write_text(content, encoding='utf-8')
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return verify_package(target)


def verify_package(directory):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('package checksum mismatch')
    package = json.loads((root / 'model-package.json').read_text(encoding='utf-8'))
    if package['package_digest'] != manifest['package_digest']:
        raise ValueError('package digest mismatch')
    return {'verified': True, 'package_digest': package['package_digest'],
            'files': len(manifest['files'])}


def load_package(directory):
    verify_package(directory)
    return json.loads((Path(directory) / 'model-package.json').read_text(encoding='utf-8'))


def predict_new_request(package, features, model_name='ridge_log1p'):
    """用冻结包预测新请求；特征必须按请求时刻构造。"""
    if model_name == 'ridge_log1p':
        model = RidgeDurationModel.from_dict(package['models']['ridge_log1p'])
    else:
        model = MedianBaseline.from_dict(package['models']['median_baseline'])
    bound = OneSidedUpperBound.from_dict(package['calibration'][model_name])
    decision = PredictionPolicy(model).decide(features)
    if decision['prediction_seconds'] is not None:
        decision['upper_bound_seconds'] = bound.upper_seconds(
            decision['prediction_seconds'])
        decision['upper_bound_supported'] = bound.supported
    else:
        decision['upper_bound_seconds'] = None
        decision['upper_bound_supported'] = bound.supported
    decision['model'] = model_name
    decision['package_digest'] = package['package_digest']
    return decision
