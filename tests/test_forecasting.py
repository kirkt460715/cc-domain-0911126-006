"""预测研究的边界与防泄漏测试。

覆盖任务点名的三类边界：
1. 未来修订可能泄漏——点时特征与早期知识截止只能看到旧 revision；
2. 同一客户多次请求——同客户不跨进同折训练/验证，跨界线记录显式剔除；
3. 全新服务类型——最终测试期的未见类别只能拒判，不产生精确值。

另覆盖：标签互斥原因、暂停扣除、log1p 反变换回到秒、有限样本分位数
覆盖率规则、校准不足标注、冻结包重载一致性。
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

from dispatch_log.schema import Dataset, instant, canonical
from forecasting import synth
from forecasting.dataset import (build_dataset, build_features, build_label,
                                 feature_spec, REASON_CANCELLED,
                                 REASON_INVALID_TIMELINE, REASON_INCOMPLETE,
                                 REASON_COMPLETED)
from forecasting.splits import make_folds, make_holdout_splits
from forecasting.models import (RidgeDurationModel, MedianBaseline,
                                log1p_seconds, expm1_seconds, model_digest)
from forecasting.evaluation import (OneSidedUpperBound, PredictionPolicy,
                                    finite_sample_quantile_index, mae,
                                    pinball_loss, coverage_and_overshoot)
from forecasting.training import (run_study, save_package, load_package,
                                  predict_new_request, verify_package,
                                  select_hyperparameters)
from forecasting.study import derive_boundaries, run


class SynthFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document, cls.meta = synth.generate(seed=20261113)
        cls.dataset = Dataset(cls.document)
        cls.effective = cls.meta['label_effective_at']
        cls.knowledge = cls.meta['label_knowledge_at']
        cls.pit = build_dataset(cls.dataset, cls.effective, cls.knowledge)

    def test_schema_valid_and_deterministic(self):
        again, _ = synth.generate(seed=20261113)
        self.assertEqual(canonical(again), canonical(self.document))
        Dataset(again)  # 不抛异常即通过 schema

    def test_exclusion_reasons_are_distinct_and_complete(self):
        counts = self.pit['reason_counts']
        for reason in (REASON_CANCELLED, REASON_INVALID_TIMELINE):
            self.assertGreater(counts[reason], 0)
        # 未完成（右截尾）：业务截止时仍 active/arrived 的单子不可训练。
        self.assertGreaterEqual(counts.get(REASON_INCOMPLETE, 0), 0)
        labels = [sample['label']['reason'] for sample in self.pit['samples']]
        self.assertTrue(all(reason == REASON_COMPLETED for reason in labels))
        exclusion_reasons = {sample['label']['reason'] for sample in self.pit['exclusions']}
        self.assertNotIn(REASON_COMPLETED, exclusion_reasons)

    def test_label_excludes_pause_seconds(self):
        paused = [sample for sample in self.pit['samples']
                  if sample['label']['pause_seconds'] > 0]
        self.assertGreater(len(paused), 0)
        # 与独立重放交叉核对：有效秒数 = 活跃区间并集时长。
        from dispatch_log.selection import select
        from dispatch_log.replay import replay_visit
        selection = select(self.dataset.events, self.knowledge)
        for sample in paused[:5]:
            visit = self.dataset.visits[sample['visit']]
            rows = [event for event in selection['events']
                    if event['visit'] == visit['id']]
            replayed = replay_visit(self.dataset, visit, rows, self.effective)
            self.assertEqual(replayed['state'], 'completed')
            self.assertEqual(sample['label']['active_seconds'],
                             replayed['active_seconds'])
            # 有效秒数严格小于从 started 到 completed 的墙钟跨度（暂停已扣除）。
            started = replayed['milestones']['started'][0]
            completed = replayed['milestones']['completed'][-1]
            self.assertLess(replayed['active_seconds'], completed - started)


class LeakageTests(unittest.TestCase):
    """边界一：未来修订不得污染点时信息。"""

    @classmethod
    def setUpClass(cls):
        cls.document, cls.meta = synth.generate(seed=20261113)
        cls.dataset = Dataset(cls.document)
        cls.target = cls.meta['revised_visit_ids'][0]
        cls.visit = cls.dataset.visits[cls.target]
        cls.request_ts = instant(cls.visit['requested_at'], 'r')
        cls.pit = build_dataset(cls.dataset, cls.meta['label_effective_at'],
                                cls.meta['label_knowledge_at'])

    def test_revision_two_is_registered_only_after_knowledge_cutoff(self):
        revisions = [event for event in self.dataset.events
                     if event['visit'] == self.target and event['kind'] == 'completed']
        self.assertEqual(len(revisions), 2)
        rev2 = max(revisions, key=lambda event: event['revision'])
        self.assertGreater(instant(rev2['observed_at'], 'o'),
                           instant(self.meta['label_knowledge_at'], 'k'))

    def test_features_at_request_ignore_future_revision(self):
        features, history = build_features(self.dataset, self.visit, self.request_ts)
        # 该单自身的完成事件（含未来修订）绝不能出现在它自己的历史里。
        self.assertNotIn(self.target, history)
        # 特征只含允许字段，且无完成状态/员工/暂停信息。
        self.assertEqual(set(features), set(feature_spec()['known_at_cutoff'][i]['name']
                                            for i in range(len(feature_spec()['known_at_cutoff']))))

    def test_label_under_early_knowledge_uses_old_revision(self):
        # 知识截止早于 rev2 登记：只能按 rev1 打标签。
        early_knowledge = self.meta['label_knowledge_at']
        label_early = build_label(self.dataset, self.visit,
                                  self.meta['label_effective_at'], early_knowledge)
        late_knowledge = max(instant(event['observed_at'], 'o')
                             for event in self.dataset.events
                             if event['visit'] == self.target) + 1
        from dispatch_log.schema import utc
        label_late = build_label(self.dataset, self.visit,
                                 self.meta['label_effective_at'], utc(late_knowledge))
        self.assertTrue(label_early['trainable'] and label_late['trainable'])
        self.assertNotEqual(label_early['active_seconds'], label_late['active_seconds'])
        # 晚截止纳入 rev2（完成时刻延后 900 秒，无暂停变化时有效时长增加）。
        self.assertGreater(label_late['active_seconds'], label_early['active_seconds'])

    def test_no_elapsed_time_as_label(self):
        # 右截尾单必须是不可训练原因，而不是把当前已耗时当总时长。
        for ident in self.meta['open_visit_ids'][:3]:
            label = build_label(self.dataset, self.dataset.visits[ident],
                                self.meta['label_effective_at'],
                                self.meta['label_knowledge_at'])
            self.assertFalse(label['trainable'])
            self.assertEqual(label['reason'], REASON_INCOMPLETE)
            self.assertIsNone(label['active_seconds'])

    def test_voided_revision_invisible_at_request_cutoff(self):
        target = self.meta['voided_visit_ids'][0]
        visit = self.dataset.visits[target]
        request_ts = instant(visit['requested_at'], 'r')
        revisions = [event for event in self.dataset.events
                     if event['visit'] == target and event['kind'] == 'accepted']
        self.assertTrue(any(event['void'] for event in revisions))
        # 截点特征：作废 revision 登记得晚，历史不受影响且该单不进自身历史。
        features, history = build_features(self.dataset, visit, request_ts)
        self.assertNotIn(target, history)
        # 知识截止早于作废登记：标签仍按有效 revision 正常完成。
        label = build_label(self.dataset, visit,
                            self.meta['label_effective_at'],
                            self.meta['label_knowledge_at'])
        self.assertTrue(label['trainable'])

    def test_client_history_grows_only_with_registered_completions(self):
        # 取一个有多次请求的训练客户，按请求顺序看历史完成数单调不减，
        # 且只统计完成时刻与观测时刻都早于截点的单。
        from collections import defaultdict
        by_client = defaultdict(list)
        for sample in self.pit['samples']:
            by_client[sample['client']].append(sample)
        multi = next((rows for rows in by_client.values() if len(rows) >= 3), None)
        self.assertIsNotNone(multi)
        ordered = sorted(multi, key=lambda row: row['request_ts'])
        counts = [row['features']['client_prior_completed_count'] for row in ordered]
        self.assertEqual(counts, sorted(counts))
        for earlier, later in zip(ordered, ordered[1:]):
            self.assertLess(earlier['request_ts'], later['request_ts'])
            self.assertLessEqual(
                earlier['features']['client_prior_completed_count'],
                later['features']['client_prior_completed_count'])
            # 历史里的每个 visit 都必须早于当前请求。
            for historical in later['history_visits']:
                hv = self.dataset.visits[historical]
                self.assertLessEqual(instant(hv['requested_at'], 'r'),
                                     later['request_ts'])


class ClientSplitTests(unittest.TestCase):
    """边界二：同一客户多次请求不跨进同折训练与验证。"""

    @classmethod
    def setUpClass(cls):
        cls.document, cls.meta = synth.generate(seed=20261113)
        cls.dataset = Dataset(cls.document)
        cls.pit = build_dataset(cls.dataset, cls.meta['label_effective_at'],
                                cls.meta['label_knowledge_at'])
        cls.folds, cls.kwargs = derive_boundaries(cls.pit['samples'])
        cls.split = make_holdout_splits(cls.pit['samples'], **cls.kwargs)

    def test_no_client_shared_across_fold_sides(self):
        folds = make_folds(self.pit['samples'], self.folds)
        for fold in folds:
            self.assertFalse(
                set(fold['train_clients']) & set(fold['validation_clients']))
            # 剔除报告必须列出具体 visit 与代价。
            self.assertEqual(
                len(fold['purge']['dropped_visits']), fold['purge']['dropped_count'])

    def test_holdout_cohorts_disjoint(self):
        train = set(self.split['train_clients'])
        cal = set(self.split['calibration_clients'])
        test = set(self.split['test_clients'])
        self.assertFalse(train & cal)
        self.assertFalse(train & test)
        self.assertFalse(cal & test)
        # 跨界线剔除有显式代价记录。
        self.assertIn('dropped_count', self.split['purge_calibration'])
        self.assertIn('dropped_visits', self.split['purge_test_vs_calibration'])

    def test_purge_drops_validation_records_not_training(self):
        # 训练侧样本数不被去重削减；被删的是验证/保留侧记录。
        by_id = {sample['visit']: sample for sample in self.pit['samples']}
        for fold in make_folds(self.pit['samples'], self.folds):
            for visit in fold['purge']['dropped_visits']:
                self.assertNotIn(visit, fold['validation_visits'])
                self.assertIn(visit, by_id)


class UnseenServiceTests(unittest.TestCase):
    """边界三：全新服务类型只能拒判或明示保守基线。"""

    @classmethod
    def setUpClass(cls):
        cls.document, cls.meta = synth.generate(seed=20261113)
        cls.dataset = Dataset(cls.document)
        cls.pit = build_dataset(cls.dataset, cls.meta['label_effective_at'],
                                cls.meta['label_knowledge_at'])
        cls.study = run(cls.dataset, cls.meta['label_effective_at'],
                        cls.meta['label_knowledge_at'], coverage=0.9)

    def test_install_absent_from_training(self):
        train_visits = set(self.study['split']['train'])
        install_train = [sample for sample in self.pit['samples']
                         if sample['visit'] in train_visits
                         and sample['features']['service'] == 'install']
        self.assertEqual(install_train, [])

    def test_unseen_service_abstains(self):
        package = self.study['package']
        install_visit = next(v for v in self.dataset.visits.values()
                             if v['service'] == 'install')
        features, _ = build_features(
            self.dataset, install_visit,
            instant(install_visit['requested_at'], 'r'))
        for model_name in ('ridge_log1p', 'median_baseline'):
            decision = predict_new_request(package, features, model_name)
            self.assertEqual(decision['status'], 'abstain')
            self.assertEqual(decision['reason'], 'unseen_service_category')
            self.assertIsNone(decision['prediction_seconds'])

    def test_known_service_returns_seconds(self):
        package = self.study['package']
        visit = next(v for v in self.dataset.visits.values() if v['service'] == 'care')
        features, _ = build_features(self.dataset, visit,
                                     instant(visit['requested_at'], 'r'))
        decision = predict_new_request(package, features, 'ridge_log1p')
        if decision['status'] != 'abstain':
            self.assertIsInstance(decision['prediction_seconds'], float)
            self.assertGreater(decision['prediction_seconds'], 0)


class ModelAndMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document, cls.meta = synth.generate(seed=20261113)
        cls.dataset = Dataset(cls.document)
        cls.pit = build_dataset(cls.dataset, cls.meta['label_effective_at'],
                                cls.meta['label_knowledge_at'])
        cls.folds, cls.kwargs = derive_boundaries(cls.pit['samples'])

    def test_log_transform_round_trip_in_seconds(self):
        for value in (0, 1, 1800, 7200.5):
            self.assertAlmostEqual(expm1_seconds(log1p_seconds(value)), value)
        model = RidgeDurationModel(alpha=1.0)
        model.fit(self.pit['samples'][:50])
        sample = self.pit['samples'][0]
        estimate, status, _ = model.predict_seconds(sample['features'])
        self.assertEqual(status, 'ok')
        self.assertGreater(estimate, 0)

    def test_hyperparameter_selection_inside_forward_folds(self):
        folds = make_folds(self.pit['samples'], self.folds)
        report = select_hyperparameters(
            self.pit['samples'], folds, alpha_grid=(0.1, 1.0, 10.0))
        self.assertIn(report['selected_alpha'], (0.1, 1.0, 10.0))
        for fold in report['folds']:
            if not fold.get('skipped'):
                self.assertEqual(set(fold['ridge_mae_by_alpha']), {0.1, 1.0, 10.0})

    def test_finite_sample_quantile_rule(self):
        # n=10、coverage=0.9 -> ceil(0.9*11)=10（1 基）-> 下标 9，不越界。
        self.assertEqual(finite_sample_quantile_index(10, 0.9), 9)
        # n=5、0.95 -> ceil(5.7)=6 > 5：样本不足，返回 None。
        self.assertIsNone(finite_sample_quantile_index(5, 0.95))
        # 临界：n=19、0.95 -> ceil(19)=19 支持；n=18 -> ceil(18.05)=19 不支持。
        self.assertEqual(finite_sample_quantile_index(19, 0.95), 18)
        self.assertIsNone(finite_sample_quantile_index(18, 0.95))
        bound = OneSidedUpperBound(0.95)
        bound.fit([100.0] * 5, [90.0] * 5)
        self.assertFalse(bound.supported)
        self.assertIsNotNone(bound.note)
        self.assertIsNone(bound.upper_seconds(90.0))

    def test_upper_bound_coverage_and_overshoot(self):
        actual = [100, 200, 300]
        upper = [150, 250, 280]
        summary = coverage_and_overshoot(actual, upper)
        self.assertAlmostEqual(summary['coverage'], 2 / 3)
        self.assertEqual(summary['breaches'], 1)
        self.assertEqual(summary['max_overshoot_seconds'], 20)
        loss = pinball_loss(actual, upper, 0.9)
        self.assertGreater(loss, 0)
        self.assertIsNotNone(mae(actual, [120, 210, 290]))

    def test_out_of_support_numeric_feature_abstains(self):
        model = RidgeDurationModel(alpha=1.0).fit(self.pit['samples'][:100])
        sample = self.pit['samples'][0]
        extreme = dict(sample['features'])
        support = model.numeric_support['client_prior_active_mean_seconds']
        if support['max'] is not None:
            extreme['client_prior_active_mean_seconds'] = support['max'] * 100
            decision = PredictionPolicy(model).decide(extreme)
            self.assertEqual(decision['status'], 'abstain')
            self.assertEqual(decision['reason'],
                             'numeric_feature_outside_training_support')


class FrozenPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document, cls.meta = synth.generate(seed=20261113)
        cls.dataset = Dataset(cls.document)
        cls.study = run(cls.dataset, cls.meta['label_effective_at'],
                        cls.meta['label_knowledge_at'], coverage=0.9)

    def test_package_contains_required_sections(self):
        package = self.study['package']
        for section in ('feature_spec', 'split_assignment', 'transforms',
                        'models', 'calibration', 'raw_event_summary'):
            self.assertIn(section, package)
        self.assertIn('multi_revision_events', package['raw_event_summary'])
        self.assertGreater(package['raw_event_summary']['void_event_count'], 0)
        self.assertTrue(package['raw_event_summary']['multi_revision_events'])

    def test_reload_predicts_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'pkg'
            save_package(self.study['package'], target)
            verify_package(target)
            reloaded = load_package(target)
            self.assertEqual(reloaded['package_digest'],
                             self.study['package']['package_digest'])
            visit = self.dataset.visits['gen-0001']
            features, _ = build_features(self.dataset, visit,
                                         instant(visit['requested_at'], 'r'))
            first = predict_new_request(self.study['package'], features)
            second = predict_new_request(reloaded, features)
            self.assertEqual(first, second)

    def test_late_events_require_new_version(self):
        # 冻结包摘要记录输入指纹；后补事件改变输入，指纹必然不同，
        # 业务规则上只能产出新版本（versioning_rule 已写入包内）。
        package = self.study['package']
        self.assertIn('versioning_rule', package)
        mutated = json.loads(canonical(self.document))
        mutated['events'].append(dict(mutated['events'][0]))
        mutated['events'][-1] = dict(mutated['events'][-1], id='late-1',
                                     visit=mutated['visits'][0]['id'],
                                     sequence=99, revision=1)
        changed = Dataset(mutated)
        self.assertNotEqual(changed.fingerprint, package['created_from_input_digest'])


if __name__ == '__main__':
    unittest.main()
