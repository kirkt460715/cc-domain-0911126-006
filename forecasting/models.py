"""时长模型：分组中位数基线与可正则化的线性模型。

* 中位数基线按服务类型（必要时回退到全局）取训练集分位数中位数；
* 线性模型在 log1p(秒) 目标上做岭回归（L2 可正则化），类别 one-hot、
  数值量按训练折均值/标准差标准化；预测与评分一律反变换回秒；
* 未见类别、数值特征超出训练支持范围时由上层策略给出拒判或保守基线，
  模型本身如实暴露支持域。
"""

import math

from dispatch_log.schema import canonical, digest
import json

CATEGORICAL = ('service', 'region', 'weekday', 'hour_of_day')
NUMERIC = ('client_prior_completed_count', 'client_prior_active_mean_seconds',
           'client_prior_active_median_seconds', 'client_prior_by_service_mean_seconds')
LOG_TRANSFORM = 'log1p'


def log1p_seconds(value):
    return math.log1p(max(0.0, float(value)))


def expm1_seconds(value):
    return max(0.0, math.expm1(value))


def _quantile(sorted_values, q):
    """线性插值分位数，q in (0,1)。"""
    if not sorted_values:
        raise ValueError('empty quantile input')
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def median(values):
    return _quantile(sorted(values), 0.5)


class MedianBaseline:
    """按服务类型分组的中位数（秒）基线。"""

    def __init__(self, min_group_size=5):
        self.min_group_size = min_group_size
        self.by_service = {}
        self.global_median = None
        self.support = {}
        self.fitted = False

    def fit(self, samples):
        grouped = {}
        all_values = []
        for sample in samples:
            value = sample['label']['active_seconds']
            grouped.setdefault(sample['features']['service'], []).append(value)
            all_values.append(value)
        self.by_service = {
            service: median(values) for service, values in grouped.items()
            if len(values) >= self.min_group_size
        }
        self.global_median = median(all_values)
        self.support = {service: {'count': len(values),
                                  'min': min(values), 'max': max(values)}
                        for service, values in grouped.items()}
        self.fitted = True
        return self

    def predict_seconds(self, features):
        service = features['service']
        if service in self.by_service:
            return self.by_service[service], 'ok', 'service_median'
        # 未见服务类型：明示保守基线（全局中位数），不伪造精确值。
        return self.global_median, 'unseen_category_fallback', 'global_median_conservative'

    def to_dict(self):
        return {
            'kind': 'median_baseline',
            'min_group_size': self.min_group_size,
            'by_service': self.by_service,
            'global_median': self.global_median,
            'support': self.support,
            'transform': 'identity_seconds',
        }

    @classmethod
    def from_dict(cls, body):
        model = cls(min_group_size=body['min_group_size'])
        model.by_service = body['by_service']
        model.global_median = body['global_median']
        model.support = body['support']
        model.fitted = True
        return model


class RidgeDurationModel:
    """log1p 目标上的岭回归，纯 Python 小矩阵正规方程求解。"""

    def __init__(self, alpha=1.0, min_group_size=5):
        self.alpha = alpha
        self.min_group_size = min_group_size
        self.categories = {}      # feature -> {value: column index}
        self.numeric_mean = {}
        self.numeric_scale = {}
        self.numeric_support = {}
        self.coefficients = None  # 含截距
        self.global_median = None
        self.train_services = set()
        self.fitted = False

    def _numeric_vector(self, features):
        vector = []
        for name in NUMERIC:
            value = features[name]
            if value is None:
                value = 0.0
            mean = self.numeric_mean[name]
            scale = self.numeric_scale[name] or 1.0
            vector.append((float(value) - mean) / scale)
        return vector

    def _design_row(self, features):
        row = [1.0]  # 截距
        for name in CATEGORICAL:
            vocab = self.categories[name]
            encoded = [0.0] * len(vocab)
            if features[name] in vocab:
                encoded[vocab[features[name]]] = 1.0
            row.extend(encoded)
        row.extend(self._numeric_vector(features))
        return row

    def fit(self, samples):
        # 类别词表只来自训练数据；低频类别不入词表，落到全零（未知）列。
        counts = {name: {} for name in CATEGORICAL}
        numeric_values = {name: [] for name in NUMERIC}
        targets = []
        all_seconds = []
        for sample in samples:
            for name in CATEGORICAL:
                value = str(sample['features'][name])
                counts[name][value] = counts[name].get(value, 0) + 1
            for name in NUMERIC:
                if sample['features'][name] is not None:
                    numeric_values[name].append(float(sample['features'][name]))
            targets.append(log1p_seconds(sample['label']['active_seconds']))
            all_seconds.append(sample['label']['active_seconds'])
        self.categories = {}
        for name in CATEGORICAL:
            vocab = sorted(value for value, count in counts[name].items()
                           if count >= self.min_group_size)
            self.categories[name] = {value: index for index, value in enumerate(vocab)}
        for name in NUMERIC:
            values = numeric_values[name]
            mean = sum(values) / len(values) if values else 0.0
            variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
            self.numeric_mean[name] = mean
            self.numeric_scale[name] = math.sqrt(variance)
            if values:
                self.numeric_support[name] = {'min': min(values), 'max': max(values)}
            else:
                self.numeric_support[name] = {'min': None, 'max': None}
        design = [self._design_row(sample['features']) for sample in samples]
        width = len(design[0])
        matrix = [[0.0] * width for _ in range(width)]
        rhs = [0.0] * width
        for row, target in zip(design, targets):
            for i, xi in enumerate(row):
                rhs[i] += xi * target
                for j in range(i + 1):
                    matrix[i][j] += xi * row[j]
        for i in range(width):
            for j in range(i + 1):
                matrix[j][i] = matrix[i][j]
        # L2 惩罚不正则化截距。
        for i in range(1, width):
            matrix[i][i] += self.alpha
        self.coefficients = _solve(matrix, rhs)
        self.global_median = median(all_seconds)
        self.train_services = set(counts['service'])
        self.fitted = True
        return self

    def out_of_support(self, features):
        """返回超支持域的特征名列表。"""
        outside = []
        for name in NUMERIC:
            value = features[name]
            if value is None:
                continue
            support = self.numeric_support[name]
            if support['min'] is not None and (value < support['min'] or value > support['max']):
                outside.append(name)
        return outside

    def predict_seconds(self, features):
        if features['service'] not in self.train_services:
            return self.global_median, 'unseen_category_fallback', 'global_median_conservative'
        row = self._design_row(features)
        score = sum(coef * value for coef, value in zip(self.coefficients, row))
        return expm1_seconds(score), 'ok', 'ridge_log1p'

    def to_dict(self):
        return {
            'kind': 'ridge_log1p',
            'alpha': self.alpha,
            'min_group_size': self.min_group_size,
            'categories': self.categories,
            'numeric_mean': self.numeric_mean,
            'numeric_scale': self.numeric_scale,
            'numeric_support': self.numeric_support,
            'coefficients': self.coefficients,
            'global_median': self.global_median,
            'train_services': sorted(self.train_services),
            'transform': LOG_TRANSFORM,
        }

    @classmethod
    def from_dict(cls, body):
        model = cls(alpha=body['alpha'], min_group_size=body['min_group_size'])
        model.categories = body['categories']
        model.numeric_mean = body['numeric_mean']
        model.numeric_scale = body['numeric_scale']
        model.numeric_support = body['numeric_support']
        model.coefficients = body['coefficients']
        model.global_median = body['global_median']
        model.train_services = set(body['train_services'])
        model.fitted = True
        return model


def _solve(matrix, rhs):
    """高斯-若尔当消元解线性方程组。"""
    n = len(rhs)
    work = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-12:
            raise ValueError('singular design matrix; increase regularization')
        work[column], work[pivot] = work[pivot], work[column]
        pivot_value = work[column][column]
        work[column] = [value / pivot_value for value in work[column]]
        for row in range(n):
            if row == column:
                continue
            factor = work[row][column]
            work[row] = [a - factor * b for a, b in zip(work[row], work[column])]
    return [work[i][n] for i in range(n)]


def model_digest(model):
    return digest(json.loads(canonical(model.to_dict())))
