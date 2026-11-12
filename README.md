# 上门服务履约与工时

本项目使用 Python 3.11 及以上版本，仅依赖标准库，无需安装第三方包。数据文件均为本地合成记录，不含真实个人或机构资料。

dispatch_log 保存双时点事件并重放服务过程；worktime 处理暂停、路程、重叠及班次归属；quality 计算响应及履约口径；operations 保存汇总、查询、发布与恢复。

请求日、完结日和实际工作日分别统计。未完成记录为截尾，取消不算完成。员工重叠工时按时间并集校验，按服务分摊后仍需守恒。

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
