"""预测研究用的确定性合成事件数据生成器。

生成的 Dataset 文档与生产管线同一 schema，刻意覆盖：

* 多个服务类型与地区、不同客户的时长倾向（新客户/老客户）；
* 暂停（从开始到完成之间扣除暂停后的有效服务秒数才是标签）；
* 取消单、业务截止时仍未完成的截尾单、非法时间线单；
* 事件的未来修订（revision 提升、void、延迟 observed_at），用于验证
  点时特征不会被未来信息污染；
* 一个只在最终测试期才出现的全新服务类型，用于验证未见类别拒判。

所有随机数来自带种子的 random.Random，同种子逐字节复现。
"""

import random
from datetime import datetime, timezone

from dispatch_log.schema import canonical

SERVICES = [
    # id, 响应秒, 有效时长上下限, 基础均值（秒）, 基础波动
    ('care', 1800, 1800, 7200, 3600, 700),
    ('repair', 1800, 2400, 10800, 5400, 1100),
    ('install', 1800, 3000, 14400, 7200, 1400),
]
REGIONS = ('east', 'west', 'north', 'south')
PAUSE_REASONS = ('client_break', 'worker_break', 'medical', 'equipment')
CANCEL_REASONS = ('client', 'provider', 'weather', 'duplicate', 'medical')
DAY = 86400


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace('+00:00', 'Z')


def _day_start(date_index, base_ts):
    return base_ts + date_index * DAY


class EventBuilder:
    def __init__(self, rng):
        self.rng = rng
        self.events = []
        self._counter = 0

    def add(self, visit, sequence, kind, effective_at, observed_at,
            worker='w1', reason=None, revision=1, void=False):
        self._counter += 1
        self.events.append({
            'id': f'{visit}-e{sequence}',
            'revision': revision,
            'visit': visit,
            'sequence': sequence,
            'effective_at': _iso(effective_at),
            'observed_at': _iso(observed_at),
            'kind': kind,
            'worker': worker,
            'reason': reason,
            'void': void,
        })

    def revise(self, visit, sequence, kind, effective_at, observed_at,
               revision, worker='w1', reason=None, void=False):
        """追加一个更高 revision 的事件版本（历史 revision 保留）。"""
        self.add(visit, sequence, kind, effective_at, observed_at, worker,
                 reason, revision=revision, void=void)


def _client_taste(rng):
    # 每个客户有稳定的时长乘子，制造客户间异质性。
    return {client: rng.uniform(0.75, 1.35) for client in CLIENTS}


CLIENTS = [f'client-{i:03d}' for i in range(36)]
# 三个互不复用的客户队列：训练队列在早期进入，校准队列与测试队列分别在
# 对应保留段开始时才首次出现，保证同一客户不会跨进训练与验证/校准/测试。
TRAIN_COHORT = CLIENTS[:20]
CALIBRATION_COHORT = CLIENTS[20:28]
TEST_COHORT = CLIENTS[28:36]
WORKERS = [
    {'id': 'w1', 'region': 'east', 'daily_regular_seconds': 28800},
    {'id': 'w2', 'region': 'west', 'daily_regular_seconds': 28800},
    {'id': 'w3', 'region': 'north', 'daily_regular_seconds': 28800},
    {'id': 'w4', 'region': 'south', 'daily_regular_seconds': 28800},
]


def generate(seed=20261113, days=26 * 7, label_effective_day=25 * 7,
             label_knowledge_offset=2 * DAY):
    """返回 (document, meta)。

    days: 生成的请求日总数；前 label_effective_day 天在业务截止之前。
    客户按三个时间队列进入：训练队列（第 0 天起）、校准队列（最后 4 周
    起）、测试队列（最后 2 周起），队列间客户互不复用。
    """
    rng = random.Random(seed)
    base_ts = int(datetime(2026, 1, 5, tzinfo=timezone.utc).timestamp())  # Monday
    tastes = _client_taste(rng)
    builder = EventBuilder(rng)
    visits = []
    service_index = {row[0]: row for row in SERVICES}
    # 训练客户按 fold_weeks 块轮转：同一块内同一客户多次请求（保留客户
    # 历史特征），相邻块使用不同客户子集，保证向前折训练/验证窗客户不重叠。
    fold_weeks = 4
    train_groups = [TRAIN_COHORT[:10], TRAIN_COHORT[10:20]]
    new_client_threshold_day = 7 * 7
    # 校准段 4 周、测试段 2 周，且与 28 天客户轮转块对齐：
    # 前 140 天 = 5 个完整轮转块，随后校准、测试，界线无跨块客户。
    calibration_first_day = days - 42
    test_first_day = days - 14
    invalid_visit_ids = set()
    cancelled_visit_ids = set()
    open_visit_ids = set()
    revised_visit_ids = set()
    voided_visit_ids = set()

    visit_counter = 0
    for day in range(days):
        day_ts = _day_start(day, base_ts)
        # 每天安排 6 个请求：老客户轮转 + 部分新客户 + 周末略少
        weekday = day % 7
        slots = 5 if weekday >= 5 else 6
        for slot in range(slots):
            visit_counter += 1
            vid = f'gen-{visit_counter:04d}'
            hour = 8 + slot * 2
            request_ts = day_ts + hour * 3600
            # 客户队列按时间进入，队列间无共享客户；训练队列按 4 周块轮转。
            if day >= test_first_day:
                cohort = TEST_COHORT
            elif day >= calibration_first_day:
                cohort = CALIBRATION_COHORT
            else:
                cohort = train_groups[(day // (fold_weeks * 7)) % len(train_groups)]
            client = cohort[(day * 7 + slot) % len(cohort)]
            is_new_client = client not in TRAIN_COHORT
            # 服务类型：install 只在最后两周（最终测试期）出现
            if day >= days - 14:
                service = SERVICES[rng.randrange(3)][0]
            else:
                service = SERVICES[rng.randrange(2)][0]
            region = REGIONS[(int(client.split('-')[1]) + slot) % len(REGIONS)]
            worker = f'w{(REGIONS.index(region) % 4) + 1}'
            promised_ts = request_ts + 1800
            visits.append({
                'id': vid, 'client': client, 'service': service, 'region': region,
                'requested_at': _iso(request_ts), 'promised_start': _iso(promised_ts),
                'parent': None,
            })

            # 业务截止之后的单子不生成完成事件（天然截尾/未来）。
            after_cutoff = day >= label_effective_day
            pre_cutoff_open_window = label_effective_day - 2 <= day < label_effective_day
            roll = rng.random()
            if not after_cutoff and roll < 0.05:
                cancelled_visit_ids.add(vid)
                _emit_cancelled(builder, rng, vid, request_ts, worker)
                continue
            if not after_cutoff and roll < 0.08:
                # 非法时间线：paused 后直接 completed（缺少 resumed）。
                invalid_visit_ids.add(vid)
                _emit_invalid(builder, rng, vid, request_ts, worker, service_index[service],
                              tastes[client])
                continue
            if after_cutoff:
                continue
            if pre_cutoff_open_window and roll < 0.5:
                # 业务截止前 2 天内请求、但截止时仍未完成：右截尾单。
                open_visit_ids.add(vid)
                _emit_open(builder, rng, vid, request_ts, worker)
                continue

            _emit_completed(builder, rng, vid, request_ts, worker,
                            service_index[service], tastes[client],
                            new_client=is_new_client)

    # 针对一个已完成单注入“未来修订”：其完成事件在很久之后才被修订，
    # 旧 revision 的 observed_at 较早、新 revision 把实际完成时间改长，
    # 用来检验点时特征与早期折只看得到旧版本。
    completed_ids = [v['id'] for v in visits
                     if v['id'] not in cancelled_visit_ids
                     and v['id'] not in invalid_visit_ids
                     and v['id'] not in open_visit_ids]
    leak_target = completed_ids[10]
    target_visit = next(v for v in visits if v['id'] == leak_target)
    request_ts = int(datetime.fromisoformat(target_visit['requested_at'].replace('Z', '+00:00')).timestamp())
    worker = f'w{(REGIONS.index(target_visit["region"]) % 4) + 1}'
    # 找到原完成事件，追加 revision=2：完成时刻延后 900 秒，且在知识截止后才登记。
    original = next(e for e in builder.events if e['visit'] == leak_target and e['kind'] == 'completed')
    orig_eff = int(datetime.fromisoformat(original['effective_at'].replace('Z', '+00:00')).timestamp())
    new_eff = orig_eff + 900
    observed = _day_start(label_effective_day, base_ts) + label_knowledge_offset + DAY
    builder.revise(leak_target, original['sequence'], 'completed', new_eff, observed,
                   revision=2, worker=worker)
    revised_visit_ids.add(leak_target)

    # 再注入一个“事后作废”事件：某完成单的 accepted 事件在知识截止后被
    # 登记为 void 的 revision=2（revision=1 仍有效且早登记），用于检验
    # 截点特征与早期标签只看得到未作废版本，且该 visit 走通用重放路径。
    void_target = completed_ids[20]
    void_visit = next(v for v in visits if v['id'] == void_target)
    void_worker = f'w{(REGIONS.index(void_visit["region"]) % 4) + 1}'
    void_original = next(e for e in builder.events
                         if e['visit'] == void_target and e['kind'] == 'accepted')
    void_eff = int(datetime.fromisoformat(
        void_original['effective_at'].replace('Z', '+00:00')).timestamp())
    void_observed = _day_start(label_effective_day, base_ts) + label_knowledge_offset + DAY
    builder.revise(void_target, void_original['sequence'], 'accepted', void_eff,
                   void_observed, revision=2, worker=void_worker, void=True)
    voided_visit_ids = {void_target}

    document = {
        'version': 1,
        'services': [
            {'id': row[0], 'response_seconds': row[1],
             'minimum_active_seconds': row[2], 'maximum_active_seconds': row[3]}
            for row in SERVICES
        ],
        'workers': WORKERS,
        'visits': visits,
        'shifts': _shifts(base_ts, days),
        'events': builder.events,
    }
    meta = {
        'seed': seed,
        'days': days,
        'base_ts': base_ts,
        'label_effective_at': _iso(_day_start(label_effective_day, base_ts)),
        'label_knowledge_at': _iso(_day_start(label_effective_day, base_ts) + label_knowledge_offset),
        'invalid_visit_ids': sorted(invalid_visit_ids),
        'cancelled_visit_ids': sorted(cancelled_visit_ids),
        'open_visit_ids': sorted(open_visit_ids),
        'revised_visit_ids': sorted(revised_visit_ids),
        'voided_visit_ids': sorted(voided_visit_ids),
        'new_service_first_day': days - 14,
        'new_client_first_day': new_client_threshold_day,
        'calibration_first_day': calibration_first_day,
        'test_first_day': test_first_day,
        'cohorts': {
            'train': TRAIN_COHORT,
            'calibration': CALIBRATION_COHORT,
            'test': TEST_COHORT,
        },
    }
    return document, meta


def _shifts(base_ts, days):
    shifts = []
    for worker in WORKERS:
        for day in range(days + 3):
            start = base_ts + day * DAY + 7 * 3600
            shifts.append({
                'id': f'{worker["id"]}-s{day:03d}',
                'worker': worker['id'],
                'start': _iso(start),
                'end': _iso(start + 12 * 3600),
                'breaks': [
                    {'start': _iso(start + 5 * 3600), 'end': _iso(start + 5 * 3600 + 1800)}
                ],
            })
    return shifts


def _emit_common_prefix(builder, rng, vid, request_ts, worker):
    accept = request_ts + rng.randint(300, 1200)
    depart = accept + rng.randint(300, 900)
    arrive = depart + rng.randint(900, 2400)
    start = arrive + rng.randint(300, 900)
    builder.add(vid, 1, 'accepted', accept, accept, worker)
    builder.add(vid, 2, 'departed', depart, depart, worker)
    builder.add(vid, 3, 'arrived', arrive, arrive, worker)
    builder.add(vid, 4, 'started', start, start, worker)
    return start


def _emit_completed(builder, rng, vid, request_ts, worker, service_row, taste, new_client=False):
    _, _, minimum, maximum, base_mean, base_spread = service_row
    start = _emit_common_prefix(builder, rng, vid, request_ts, worker)
    # 新客户平均时长更高（排班员低估的来源之一）。
    shift = 900 if new_client else 0
    gross = int(max(minimum, min(maximum,
                 rng.gauss(base_mean * taste + shift, base_spread))))
    # 一到两次暂停，暂停不计入有效时长。
    cursor = start
    pause_total = 0
    sequence = 5
    pause_count = rng.randrange(3)
    remaining = gross
    for _ in range(pause_count):
        active_chunk = max(300, remaining // (pause_count + 1))
        cursor += active_chunk
        remaining -= active_chunk
        pause_len = rng.randint(300, 1500)
        reason = PAUSE_REASONS[rng.randrange(len(PAUSE_REASONS))]
        builder.add(vid, sequence, 'paused', cursor, cursor, worker, reason)
        sequence += 1
        builder.add(vid, sequence, 'resumed', cursor + pause_len, cursor + pause_len, worker)
        sequence += 1
        cursor += pause_len
        pause_total += pause_len
    completion_ts = cursor + max(300, remaining)
    builder.add(vid, sequence, 'completed', completion_ts, completion_ts, worker)


def _emit_cancelled(builder, rng, vid, request_ts, worker):
    accept = request_ts + rng.randint(300, 1200)
    builder.add(vid, 1, 'accepted', accept, accept, worker)
    reason = CANCEL_REASONS[rng.randrange(len(CANCEL_REASONS))]
    cancel_ts = accept + rng.randint(600, 3600)
    builder.add(vid, 2, 'cancelled', cancel_ts, cancel_ts, worker, reason)


def _emit_invalid(builder, rng, vid, request_ts, worker, service_row, taste):
    # started -> paused -> completed（缺 resumed），重放必报 invalid_transition。
    start = _emit_common_prefix(builder, rng, vid, request_ts, worker)
    pause_ts = start + rng.randint(600, 1800)
    reason = PAUSE_REASONS[rng.randrange(len(PAUSE_REASONS))]
    builder.add(vid, 5, 'paused', pause_ts, pause_ts, worker, reason)
    complete_ts = pause_ts + rng.randint(900, 2400)
    builder.add(vid, 6, 'completed', complete_ts, complete_ts, worker)


def _emit_open(builder, rng, vid, request_ts, worker):
    # 业务截止时处于 active 或 arrived 的右截尾单。
    accept = request_ts + rng.randint(300, 1200)
    depart = accept + rng.randint(300, 900)
    arrive = depart + rng.randint(900, 2400)
    builder.add(vid, 1, 'accepted', accept, accept, worker)
    builder.add(vid, 2, 'departed', depart, depart, worker)
    builder.add(vid, 3, 'arrived', arrive, arrive, worker)
    if rng.random() < 0.5:
        builder.add(vid, 4, 'started', arrive + 600, arrive + 600, worker)


def write_fixture(path, seed=20261113, days=26 * 7):
    document, meta = generate(seed=seed, days=days)
    Path_write(path, canonical(document))
    return document, meta


def Path_write(path, text):
    from pathlib import Path
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
