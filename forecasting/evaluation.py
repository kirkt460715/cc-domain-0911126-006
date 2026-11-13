"""评分、校准与拒判策略。

指标全部回到秒数口径：

* MAE：平均绝对误差；
* pinball（分位数）损失：对应单侧上界覆盖率；
* 上界覆盖率与超时幅度（实际超过上界时超出多少秒）；
* 按时间段与服务类型分组，并单列新客户。

单侧时长上界用有限样本分位数规则（保守次序统计量），校准集不足以支撑
所选覆盖率时显式标注，不伪造覆盖率。
"""

import math


def mae(actual, predicted):
    if not actual:
        return None
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def pinball_loss(actual, upper, quantile):
    """单侧上界 q 的 pinball 损失，q 接近 1 表示高覆盖上界。"""
    if not actual:
        return None
    total = 0.0
    for value, bound in zip(actual, upper):
        error = value - bound
        total += max(quantile * error, (quantile - 1) * error)
    return total / len(actual)


def coverage_and_overshoot(actual, upper):
    if not actual:
        return {'coverage': None, 'mean_overshoot_seconds': None,
                'max_overshoot_seconds': None, 'breaches': 0, 'count': 0}
    breaches = [max(0, value - bound) for value, bound in zip(actual, upper)]
    covered = sum(1 for gap in breaches if gap == 0)
    nonzero = [gap for gap in breaches if gap > 0]
    return {
        'coverage': covered / len(actual),
        'mean_overshoot_seconds': sum(nonzero) / len(nonzero) if nonzero else 0.0,
        'max_overshoot_seconds': max(nonzero) if nonzero else 0.0,
        'breaches': len(nonzero),
        'count': len(actual),
    }


def finite_sample_quantile_index(sample_size, coverage):
    """保守有限样本分位数次序统计量下标（0 基）。

    采用次序统计量规则：取第 ceil(coverage * (n+1)) 个次序统计量（1 基）。
    当该秩超过样本量 n 时，说明 n 个独立观测不足以支撑所选覆盖率，
    返回 None 由上层显式标注，而不是退化成样本最大值冒充高覆盖上界。
    """
    if not 0.5 < coverage < 1.0:
        raise ValueError('one-sided upper bound coverage must lie in (0.5, 1)')
    one_based = math.ceil(coverage * (sample_size + 1))
    if one_based > sample_size:
        return None
    return one_based - 1


class OneSidedUpperBound:
    """基于校准残差的单侧上界。

    点预测 + 残差上分位数。残差 = 实际秒数 - 点预测秒数（秒口径）。
    """

    def __init__(self, coverage):
        self.coverage = coverage
        self.residual_quantile = None
        self.calibration_size = None
        self.supported = False
        self.note = None

    def fit(self, actual_seconds, predicted_seconds):
        residuals = sorted(a - p for a, p in zip(actual_seconds, predicted_seconds))
        self.calibration_size = len(residuals)
        index = finite_sample_quantile_index(len(residuals), self.coverage)
        if index is None:
            self.supported = False
            minimum_n = math.ceil(self.coverage / (1 - self.coverage))
            self.note = (f'calibration size {len(residuals)} cannot support coverage '
                         f'{self.coverage}: the ceil(coverage*(n+1)) order statistic '
                         f'exceeds n; at least {minimum_n} independent observations are '
                         'required before an upper bound is emitted')
            return self
        self.supported = True
        self.residual_quantile = residuals[index]
        return self

    def upper_seconds(self, predicted_seconds):
        if not self.supported:
            return None
        return max(0.0, predicted_seconds + self.residual_quantile)

    def to_dict(self):
        return {
            'coverage': self.coverage,
            'residual_quantile_seconds': self.residual_quantile,
            'calibration_size': self.calibration_size,
            'supported': self.supported,
            'note': self.note,
            'rule': 'order-statistic ceil(coverage*(n+1)) on calibration residuals (seconds)',
        }

    @classmethod
    def from_dict(cls, body):
        bound = cls(body['coverage'])
        bound.residual_quantile = body['residual_quantile_seconds']
        bound.calibration_size = body['calibration_size']
        bound.supported = body['supported']
        bound.note = body['note']
        return bound


class PredictionPolicy:
    """拒判与保守基线策略。

    情形：
    * 未见服务类型 -> 拒判（abstain），不返回精确值；
    * 数值特征超出训练支持域 -> 拒判；
    * 新客户（无历史）但服务类型见过 -> 保守基线（全局中位数 + 显式标记）；
    * 校准不支持所选覆盖率 -> 点预测照给，上界标记不可用。
    """

    def __init__(self, model, abstain_on_oos=True):
        self.model = model
        self.abstain_on_oos = abstain_on_oos

    def decide(self, features):
        known_services = self.train_services()
        if features['service'] not in known_services:
            return {'status': 'abstain', 'reason': 'unseen_service_category',
                    'prediction_seconds': None}
        if hasattr(self.model, 'out_of_support'):
            outside = self.model.out_of_support(features)
            if self.abstain_on_oos and outside:
                return {'status': 'abstain',
                        'reason': 'numeric_feature_outside_training_support',
                        'out_of_support_features': outside,
                        'prediction_seconds': None}
        prediction, status, source = self.model.predict_seconds(features)
        is_new_client = features['client_prior_completed_count'] == 0
        return {
            'status': status,
            'prediction_seconds': prediction,
            'source': source,
            'new_client': is_new_client,
            'conservative_baseline': status != 'ok',
        }

    def train_services(self):
        if hasattr(self.model, 'train_services'):
            return set(self.model.train_services)
        return set(self.model.support)


def score_group(rows, predictions, upper_bounds=None, coverage=None):
    """rows: 带 label 的样本；predictions: {visit: seconds or None}。"""
    actual, pred, upper, kept, abstained = [], [], [], [], []
    for sample in rows:
        value = sample['label']['active_seconds']
        estimate = predictions.get(sample['visit'])
        if estimate is None:
            abstained.append(sample['visit'])
            continue
        actual.append(value)
        pred.append(estimate)
        kept.append(sample)
        if upper_bounds is not None:
            bound = upper_bounds.get(sample['visit'])
            if bound is not None:
                upper.append(bound)
    result = {
        'count': len(kept),
        'abstained': abstained,
        'abstained_count': len(abstained),
        'mae_seconds': mae(actual, pred),
    }
    if upper and len(upper) == len(actual):
        result['pinball_loss'] = pinball_loss(actual, upper, coverage)
        result['upper_bound'] = coverage_and_overshoot(actual, upper)
    return result


def evaluate_by_dimensions(samples, predictions, upper_bounds=None, coverage=None,
                           period_days=14, new_client_visits=None):
    """按时间段与服务类型汇总，并单列新客户表现。"""
    if not samples:
        return {}
    origin = min(sample['request_ts'] for sample in samples)
    periods, services, new_clients = {}, {}, []
    for sample in samples:
        period_index = (sample['request_ts'] - origin) // (period_days * 86400)
        periods.setdefault(period_index, []).append(sample)
        services.setdefault(sample['features']['service'], []).append(sample)
        if new_client_visits and sample['visit'] in new_client_visits:
            new_clients.append(sample)
    report = {
        'period_days': period_days,
        'by_period': {
            f'period_{index:02d}': score_group(rows, predictions, upper_bounds, coverage)
            for index, rows in sorted(periods.items())
        },
        'by_service': {
            service: score_group(rows, predictions, upper_bounds, coverage)
            for service, rows in sorted(services.items())
        },
        'new_clients': score_group(new_clients, predictions, upper_bounds, coverage),
        'overall': score_group(samples, predictions, upper_bounds, coverage),
    }
    return report
