"""点时（point-in-time）训练数据集构建。

每条样本对应一次服务请求，信息截点固定为该请求的 ``requested_at``：

* 特征只允许使用截点当时已知的服务类型、地区、星期，以及同一客户在截点
  之前已完成、且完成事件在截点当时已经登记（observed_at <= requested_at）
  的历史服务摘要；
* 标签通过业务截止与知识截止双截止重放获得，完成状态、未来修订、当次暂停、
  事后认领员工均不进入特征；
* 未完成、取消、非法时间线是互不相同的不可训练原因，不能拿当前已耗时充当
  真实总时长。

模块只依赖标准库，并复用 dispatch_log 的事件选择与重放逻辑，保证与运行
管线同一套口径。
"""

import json
from datetime import datetime, timezone

from dispatch_log.schema import instant, canonical, digest
from dispatch_log.selection import select
from dispatch_log.replay import replay_visit
from collections import defaultdict

# 与 replay.TRANSITIONS 的两个吸收态保持一致。
TERMINAL_STATES = ('completed', 'cancelled')

# 不可训练原因清单（显式、互斥、可统计）。
REASON_COMPLETED = 'trainable_completed'
REASON_INCOMPLETE = 'incomplete_at_label_cutoff'
REASON_CANCELLED = 'cancelled'
REASON_INVALID_TIMELINE = 'invalid_timeline'
REASON_NON_POSITIVE = 'non_positive_active_seconds'
TRAINABLE_REASONS = frozenset({REASON_COMPLETED})


def _weekday(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).weekday()  # 0=Monday


def _hour_of_day(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).hour


def feature_spec():
    """冻结的特征定义。只声明截点当时可知的量。"""
    return {
        'version': 1,
        'as_of': 'requested_at',
        'known_at_cutoff': [
            {'name': 'service', 'kind': 'categorical', 'source': 'visit.service'},
            {'name': 'region', 'kind': 'categorical', 'source': 'visit.region'},
            {'name': 'weekday', 'kind': 'categorical',
             'source': 'weekday(visit.requested_at, UTC)', 'values': list(range(7))},
            {'name': 'hour_of_day', 'kind': 'categorical',
             'source': 'hour(visit.requested_at, UTC)', 'values': list(range(24))},
            {'name': 'client_prior_completed_count', 'kind': 'numeric',
             'source': 'same-client visits completed with observed completed event at cutoff'},
            {'name': 'client_prior_active_mean_seconds', 'kind': 'numeric',
             'source': 'mean active_seconds of same-client completed history at cutoff'},
            {'name': 'client_prior_active_median_seconds', 'kind': 'numeric',
             'source': 'median active_seconds of same-client completed history at cutoff'},
            {'name': 'client_prior_by_service_mean_seconds', 'kind': 'numeric',
             'source': 'mean active_seconds of same-client, same-service completed history at cutoff'},
        ],
        'forbidden': [
            'completion state of this visit', 'future event revisions',
            'pauses of this visit', 'worker claimed after the request',
            'elapsed time of the current visit', 'any field observed after requested_at',
        ],
    }


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _client_history(dataset, knowledge_events_by_visit, cutoff, client,
                    client_visits):
    """同一客户在截点之前已完成且当时已登记的服务摘要。

    完成事件必须同时满足 effective_at < cutoff 与 observed_at <= cutoff，
    因此事后补录或修订不会泄漏到特征里。knowledge_events_by_visit 已按
    cutoff 对应的知识截止选好事件版本，避免每个请求重复全量选择。
    """
    history = []
    for ident in client_visits.get(client, ()):
        visit = dataset.visits[ident]
        if instant(visit['requested_at'], 'requested_at') >= cutoff:
            continue  # 截点之后才请求，绝不能进历史
        rows = knowledge_events_by_visit.get(ident)
        if not rows:
            continue
        replayed = replay_visit(dataset, visit, rows, _iso(cutoff))
        if replayed['state'] != 'completed' or not replayed['valid']:
            continue
        completed_at = replayed['milestones'].get('completed', [cutoff + 1])[-1]
        if completed_at >= cutoff:
            continue
        active = replayed['active_seconds']
        if active <= 0:
            continue
        history.append({
            'visit': ident,
            'service': visit['service'],
            'active_seconds': active,
            'completed_at': completed_at,
        })
    history.sort(key=lambda row: (row['completed_at'], row['visit']))
    return history


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace('+00:00', 'Z')


def build_features(dataset, visit, cutoff):
    """只使用 cutoff 时刻已知信息构造一条特征记录（单请求便捷入口）。"""
    timestamp = instant(visit['requested_at'], 'requested_at')
    if timestamp != cutoff:
        raise ValueError('feature cutoff must equal the visit request instant')
    selection = select(dataset.events, _iso(cutoff))
    by_visit = defaultdict(list)
    for event in selection['events']:
        by_visit[event['visit']].append(event)
    client_visits = defaultdict(list)
    for ident, row in dataset.visits.items():
        client_visits[row['client']].append(ident)
    history = _client_history(dataset, by_visit, cutoff, visit['client'], client_visits)
    return _features_from_history(visit, timestamp, history), history


def _features_from_history(visit, timestamp, history):
    active_values = [row['active_seconds'] for row in history]
    by_service = [row['active_seconds'] for row in history
                  if row['service'] == visit['service']]
    return {
        'service': visit['service'],
        'region': visit['region'],
        'weekday': _weekday(timestamp),
        'hour_of_day': _hour_of_day(timestamp),
        'client_prior_completed_count': len(history),
        'client_prior_active_mean_seconds': sum(active_values) / len(active_values) if active_values else None,
        'client_prior_active_median_seconds': _median(active_values) if active_values else None,
        'client_prior_by_service_mean_seconds': (
            sum(by_service) / len(by_service) if by_service else None),
    }


def build_label(dataset, visit, effective_at, knowledge_at, selection=None):
    """按双截止重放得到标签。

    未完成、取消、非法时间线、非正有效时长各自给出显式原因；只有时间线
    合法且已完成的样本带有效秒数标签。
    """
    if selection is None:
        selection = select(dataset.events, knowledge_at)
    event_visits = defaultdict(set)
    for event in dataset.events:
        event_visits[event['id']].add(event['visit'])
    rows = [event for event in selection['events'] if event['visit'] == visit['id']]
    issues = [issue for issue in selection['issues']
              if visit['id'] in event_visits.get(issue['event'], ())]
    replayed = replay_visit(dataset, visit, rows, effective_at, issues)
    if not replayed['valid']:
        reason = REASON_INVALID_TIMELINE
    elif replayed['state'] == 'not_requested':
        reason = 'not_requested_at_label_cutoff'
    elif replayed['state'] == 'completed':
        if replayed['active_seconds'] <= 0:
            reason = REASON_NON_POSITIVE
        else:
            reason = REASON_COMPLETED
    elif replayed['state'] == 'cancelled':
        reason = REASON_CANCELLED
    else:
        reason = REASON_INCOMPLETE
    return {
        'visit': visit['id'],
        'state': replayed['state'],
        'reason': reason,
        'trainable': reason == REASON_COMPLETED,
        'active_seconds': replayed['active_seconds'] if reason == REASON_COMPLETED else None,
        'pause_seconds': replayed.get('pause_seconds', 0),
        'right_censored': replayed.get('right_censored', True),
        'issues': replayed['issues'],
        'label_effective_at': effective_at,
        'label_knowledge_at': knowledge_at,
    }


def build_dataset(dataset, label_effective_at, label_knowledge_at):
    """构建全量点时样本集。

    标签在业务/知识双截止下重放；特征严格按各请求自己的 requested_at 截点
    构造。输出包含可训练样本与所有排除记录（带原因），排除记录不进任何
    训练折，但保留在包内供审计。
    """
def build_dataset(dataset, label_effective_at, label_knowledge_at):
    """构建全量点时样本集。

    标签在业务/知识双截止下重放；特征严格按各请求自己的 requested_at 截点
    构造。输出包含可训练样本与所有排除记录（带原因），排除记录不进任何
    训练折，但保留在包内供审计。

    性能口径：事件可见性按截点增量推进（避免每个请求全量重放事件选择），
    每个 visit 的终态重放只在标签截止下预计算一次；存在多修订或 void 的
    visit 走按截点重放的通用路径，保证修订语义不被优化破坏。
    """
    effective = instant(label_effective_at, 'label_effective_at')
    knowledge = instant(label_knowledge_at, 'label_knowledge_at')
    if knowledge < effective:
        raise ValueError('knowledge cutoff must not precede business cutoff')
    samples = []
    exclusions = []
    reason_counts = defaultdict(int)
    label_selection = select(dataset.events, label_knowledge_at)

    # 预索引：客户 -> 其全部 visit（按请求时间排序）。
    client_visits = defaultdict(list)
    for ident, row in dataset.visits.items():
        client_visits[row['client']].append(ident)
    for ident in client_visits:
        client_visits[ident].sort(
            key=lambda vid: instant(dataset.visits[vid]['requested_at'], 'r'))

    # 每个事件 id 的全部修订，按 observed_at 排序，用于增量可见性。
    revisions_by_event = defaultdict(list)
    for event in dataset.events:
        revisions_by_event[event['id']].append(event)
    for rows in revisions_by_event.values():
        rows.sort(key=lambda row: instant(row['observed_at'], 'o'))
    change_log = []  # (observed_at, event_id)
    for ident, rows in revisions_by_event.items():
        for row in rows:
            change_log.append((instant(row['observed_at'], 'o'), ident))
    change_log.sort()
    special_visits = set()  # 含多修订或 void 事件的 visit，走通用重放路径。
    for ident, rows in revisions_by_event.items():
        if len(rows) > 1 or any(row['void'] for row in rows):
            special_visits.add(rows[0]['visit'])

    # 终态预计算：标签知识截止下每个 visit 的重放结果与最大观测时间。
    label_events_by_visit = defaultdict(list)
    for event in label_selection['events']:
        label_events_by_visit[event['visit']].append(event)
    terminal = {}
    latest_observed = {}
    for ident, visit in dataset.visits.items():
        rows = label_events_by_visit.get(ident, [])
        terminal[ident] = replay_visit(dataset, visit, rows, label_effective_at)
        event_ids = {event['id'] for event in dataset.events if event['visit'] == ident}
        latest_observed[ident] = max(
            (instant(row['observed_at'], 'o')
             for event_id in event_ids
             for row in revisions_by_event[event_id]),
            default=-1)

    def visible_completed_history(cutoff, client):
        """截点 cutoff 下该客户已完成且已登记的历史（增量可见性）。"""
        while change_pointer[0] < len(change_log) and change_log[change_pointer[0]][0] <= cutoff:
            _, event_id = change_log[change_pointer[0]]
            chosen = next((row for row in reversed(revisions_by_event[event_id])
                           if instant(row['observed_at'], 'o') <= cutoff and not row['void']),
                          None)
            if chosen is None:
                visible_events.pop(event_id, None)
            else:
                visible_events[event_id] = chosen
            change_pointer[0] += 1
        history = []
        for ident in client_visits.get(client, ()):
            visit = dataset.visits[ident]
            if instant(visit['requested_at'], 'requested_at') >= cutoff:
                break  # 已按请求时间排序，后面的都在截点之后
            if ident in special_visits:
                rows = [event for event in visible_events.values()
                        if event['visit'] == ident]
                replayed = replay_visit(dataset, visit, rows, _iso(cutoff))
                if replayed['state'] != 'completed' or not replayed['valid']:
                    continue
                completed_at = replayed['milestones'].get('completed', [cutoff + 1])[-1]
            else:
                replayed = terminal[ident]
                if replayed['state'] != 'completed' or not replayed['valid']:
                    continue
                completed_at = replayed['milestones'].get('completed', [cutoff + 1])[-1]
                # 该单事件必须在截点前全部登记，否则当时尚不可知。
                if latest_observed[ident] > cutoff:
                    continue
            if completed_at >= cutoff or replayed['active_seconds'] <= 0:
                continue
            history.append({
                'visit': ident,
                'service': visit['service'],
                'active_seconds': replayed['active_seconds'],
                'completed_at': completed_at,
            })
        return history

    visible_events = {}
    change_pointer = [0]
    ordered_visits = sorted(dataset.visits.values(),
                            key=lambda row: (instant(row['requested_at'], 'r'), row['id']))
    for visit in ordered_visits:
        ident = visit['id']
        request_ts = instant(visit['requested_at'], 'requested_at')
        if request_ts > effective:
            # 业务截止之后才请求的单子在该重放中尚不存在。
            reason_counts['not_requested_at_label_cutoff'] += 1
            continue
        history = visible_completed_history(request_ts, visit['client'])
        features = _features_from_history(visit, request_ts, history)
        label = build_label(dataset, visit, label_effective_at, label_knowledge_at,
                            selection=label_selection)
        sample = {
            'visit': ident,
            'client': visit['client'],
            'requested_at': visit['requested_at'],
            'request_ts': request_ts,
            'features': features,
            'label': label,
            'history_visits': [row['visit'] for row in history],
        }
        if label['trainable']:
            samples.append(sample)
        else:
            exclusions.append(sample)
        reason_counts[label['reason']] += 1
    body = {
        'feature_spec': feature_spec(),
        'label_effective_at': label_effective_at,
        'label_knowledge_at': label_knowledge_at,
        'samples': samples,
        'exclusions': exclusions,
        'reason_counts': dict(sorted(reason_counts.items())),
        'input_digest': dataset.fingerprint,
    }
    body = json.loads(canonical(body))
    return {**body, 'dataset_digest': digest(body)}
