# 上门服务履约与工时

本项目使用 Python 3.11 及以上版本，仅依赖标准库，无需安装第三方包。数据文件均为本地合成记录，不含真实个人或机构资料。

dispatch_log 保存双时点事件并重放服务过程；worktime 处理暂停、路程、重叠及班次归属；quality 计算响应及履约口径；operations 保存汇总、查询、发布与恢复；forecasting 是仅供排班参考的服务时长预测研究，不自动改派单或员工工时。

请求日、完结日和实际工作日分别统计。未完成记录为截尾，取消不算完成。员工重叠工时按时间并集校验，按服务分摊后仍需守恒。

## 排班参考时长预测（forecasting，研究用途）

目标量是从 started 到 completed 之间扣除暂停后的有效服务秒数（active_seconds）。每条样本以该请求的 `requested_at` 为信息截点：特征只含当时已知的服务类型、地区、星期、小时，以及同一客户在截点前已完成且完成事件当时已登记的历史摘要；完成状态、未来修订、当次暂停、事后认领员工、当前已耗时都不得入特征。标签按业务截止与知识截止双截止重放得到，未完成、取消、非法时间线、非正时长是互斥的不可训练原因。

模型对照按服务分组的中位数基线与 log1p 目标上的岭回归（可 L2 正则化）；预测与评分一律反变换回秒。超参只在时间向前折内按验证 MAE 选择，同一客户不跨进同折训练与验证，跨界线记录从验证侧剔除并在包内记录代价。校准集与最终测试集时间更晚且客户互不复用；单侧时长上界用有限样本次序统计量规则 `ceil(coverage*(n+1))`，校准集不足以支撑所选覆盖率时显式标注而不输出上界。未见服务类型或数值特征超出训练支持域时返回拒判（abstain），不伪造精确值。

冻结模型包包含特征定义、拆分归属、变换参数、类别词表与原始事件摘要，带 sha256 清单与包摘要；重新加载后可对按请求时刻构造的新特征预测，后补事件只能产出新版本包。研究合成数据由带种子生成器确定性产出，覆盖暂停、取消、右截尾、非法时间线、未来事件修订、跨界线客户以及仅在最终测试期出现的全新服务类型。

```sh
# 生成确定性合成研究数据
python3 visit_metrics.py forecast-gen --output fixtures/generated/forecast_history.json
# 跑完整研究并冻结模型包
python3 visit_metrics.py forecast-run --input fixtures/generated/forecast_history.json \
  --effective-at 2026-06-29T00:00:00Z --knowledge-at 2026-07-01T00:00:00Z \
  --coverage 0.9 --package-directory ./forecast-package
# 用冻结包预测新请求（特征需按请求时刻构造）
python3 visit_metrics.py forecast-predict --package-directory ./forecast-package --features features.json
python3 visit_metrics.py forecast-verify --package-directory ./forecast-package
```

边界与防泄漏测试见 `tests/test_forecasting.py`，覆盖未来修订泄漏、同客户多次请求的跨折去重、全新服务类型拒判三类点名场景。

在项目根目录运行完整测试：

```sh
python3 -m unittest discover -s tests -v
```

编译检查：

```sh
python3 -m compileall -q .
```

实际演示：

```sh
python3 visit_metrics.py demo --fixture fixtures/visits.json
```

演示在临时目录内建立本地状态，结束后自动清理。测试、编译与演示在依赖已准备的环境内均不下载数据或软件。输入错误以非零退出码及错误对象报告。更多子命令可通过对应入口的 --help 查看。

容器环境可执行 `docker build -t local-analysis .`，再运行 `docker run --rm --network none local-analysis`。镜像固定基础内容摘要，构建后运行只访问本地文件。完整测试与演示也可直接使用已有 Python 环境。
