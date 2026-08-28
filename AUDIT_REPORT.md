# SceneFlowTracker 审计报告

## 1. 新框架目录结构

```text
SceneFlowTracker/
├── configs/config.yaml
├── scripts/run_batch_scene_tracking.py
├── scene_flow_tracker/
│   ├── config.py
│   ├── manifest.py
│   ├── jobs.py
│   ├── segment_planner.py
│   ├── queues.py
│   ├── workers/
│   │   ├── decode_worker.py
│   │   └── model_worker.py
│   ├── pipeline/
│   │   ├── model_loader.py
│   │   ├── segment_processor.py
│   │   └── episode_aggregator.py
│   └── storage/writers.py
└── tests/
```

## 2. 直接复用的旧模块

框架通过 `legacy_modules.module_root` 指向已有实验代码，并直接复用：

- `robot_sampling.RobotPointSampler`
- `robot_sampling.EnvironmentPointSampler`
- `tracking.filter_robot_tracks`
- 本地 CoTracker 包：`models.cotracker.source_root`

这些模块包含已有的机械臂采样、环境采样、CoTracker 轨迹处理和 structured-motion-aware 轨迹过滤逻辑。

## 3. 新增 wrapper / adapter

新增代码只负责批处理组织：

- `pipeline/model_loader.py`：每个 model worker 启动时加载一次 YOLO、CoTracker、采样器和过滤函数。
- `pipeline/segment_processor.py`：处理单个 segment，串联 YOLO、采样、query 合并、CoTracker 和轨迹过滤。
- `query_groups.py`：记录 `group_id` 和 `local_point_id`，并将 CoTracker 输出拆回 LEFT / RIGHT / ENV。

## 4. 输入 JSON 如何解析

`manifest.py` 读取 `input.manifest_path` 指定的 JSON 数组。每个 item 会生成一个 `EpisodeJob`。

## 5. 为什么以 `video_segments` 为准

ABC-130K 中多个 episode 可能共享同一个物理 MP4，因此不能认为一个 `video_path` 就是一个 episode。框架只使用：

```python
episode["video_segments"][view_key]
```

中的 `video_path`、`start_frame`、`end_frame` 和 `fps` 来确定真实视频范围。

## 6. 左闭右开如何实现

全框架统一使用 `[start_frame, end_frame)`。有效帧数恒等于：

```python
end_frame - start_frame
```

输出元数据中显式保存：

```json
{
  "frame_range_semantics": "half_open"
}
```

## 7. Segment 如何生成

`segment_planner.plan_segments` 使用：

```yaml
video:
  segment_frames: 15
```

作为唯一切片长度来源，不依赖浮点秒数。

## 8. 如何保证相邻 segment 无重叠

segment 生成方式是：

```text
[0,15)
[15,30)
[30,45)
```

后一段的 `start_frame` 等于前一段的 `end_frame`，因此不会重复处理任何物理帧。

## 9. Episode frame 和 physical video frame 如何转换

如果 episode 在物理视频中的起点是 `source_start_frame=3329`，那么 episode 内部 segment `[15,30)` 会映射为物理视频 `[3344,3359)`。

## 10. Decode Worker 架构

`workers/decode_worker.py` 只负责：

- 视频 seek；
- 解码 `[source_start_frame, source_end_frame)`；
- 严格检查帧数；
- 输出 `DecodedSegment`。

它不加载 YOLO、CoTracker，也不做采样或轨迹过滤。

## 11. Model Worker 架构

`workers/model_worker.py` 启动时调用 `load_model_bundle`，之后循环处理 decoded segment。模型生命周期绑定 worker，而不是 segment。

## 12. 两类 worker 数量如何配置

配置项相互独立：

```yaml
workers:
  decode_workers: 4
  model_workers: 1
  model_devices:
    - cuda:0
```

`model_workers > len(model_devices)` 默认报错，除非显式启用 `allow_device_sharing`。

## 13. Bounded queue 如何实现

`queues.py` 使用 multiprocessing bounded queue：

- `segment_job_queue_size`
- `decoded_segment_queue_size`
- `result_queue_size`

这样 decode worker 解码过快时会自然阻塞，避免 decoded RGB segment 无限堆积。

## 14. YOLO 路径如何配置

YOLO 模型路径来自：

```yaml
models:
  yolo:
    model_path: /path/to/best.pt
    confidence_threshold: 0.15
```

启动时会检查路径存在。

## 15. CoTracker 路径如何配置

CoTracker 配置来自：

```yaml
models:
  cotracker:
    source_root: /path/to/co-tracker
    model_path: /path/to/scaled_offline.pth
```

启动时会检查 checkpoint 存在。

## 16. Model Worker 如何只加载模型一次

`load_model_bundle` 只在 model worker 进程启动时执行一次。之后同一个 worker 会复用同一套 YOLO、CoTracker、sampler 和 filter。

## 17. YOLO 是否每 segment 只运行一次

是。`segment_processor.process_segment` 只在 segment-relative frame 0 上执行一次 YOLO，得到左右机械臂 bbox。

## 18. Edge 是否每 segment 只检测一次

LEFT / RIGHT 机械臂采样通过 `RobotPointSampler.sample` 共享同一次 full-frame edge bundle。

当前 ENV 仍通过已有 `EnvironmentPointSampler` adapter 调用，所以 ENV 的 edge 预处理还会单独执行一次。这是当前已知限制，后续可以把 edge bundle 显式暴露出来进一步共享。

## 19. LEFT / RIGHT / ENV 是否共用 shared preprocessing

LEFT / RIGHT 已共用。ENV 复用旧模块接口，但物理上还不是同一个 edge bundle。

## 20. CoTracker 是否每 segment 只运行一次

是。LEFT、RIGHT、ENV 三组 query 会先 concat，然后一次性送入 CoTracker。

## 21. TrajectoryAnalyzer 是否统一执行

是。三组轨迹都会调用同一个 `filter_robot_tracks`，继续使用 structured-motion-aware jitter filtering。

## 22. LEFT / RIGHT / ENV 如何拆分

`query_groups.py` 中定义：

```text
0 = LEFT
1 = RIGHT
2 = ENVIRONMENT
```

同时保存 `local_point_id`。CoTracker 输出按 layout slice 拆回三组。

## 23. Episode Aggregator 如何处理异步乱序

`EpisodeAggregator` 以 `episode_id` 归集结果，并按 `segment_id` 排序写出。它不依赖 worker 返回顺序，因此可以防止跨 episode 串数据。

## 24. NPZ 输出

每个 episode / view 输出一个 NPZ：

```text
output_root/dataset/episode_id/{view_key}.npz
```

NPZ 中保存：

- episode metadata；
- segment frame metadata；
- query group id；
- local point id；
- raw tracks；
- smooth tracks；
- visibility；
- track state；
- motion state；
- usable mask。

## 25. summary 输出

每个 episode 目录下保存 `summary.json`，记录 segment 状态、采样统计、轨迹状态统计和耗时。

## 26. resume

当：

```yaml
batch:
  resume: true
```

如果某个 episode 的 NPZ 非空且 `summary.json` 存在，则自动跳过。

## 27. atomic write

NPZ 会先写到 `.tmp` 文件，然后用 `os.replace` 原子替换目标文件，避免中断留下损坏文件却被误认为完成。

## 28. segment error recovery

decode 或 model 阶段的异常会被包装成 `SegmentResult(status="FAILED")`。如果：

```yaml
batch:
  continue_on_segment_error: true
```

则批处理继续处理后续 segment。

## 29. logging

使用标准 `logging`。每个 segment 的结果中包含：

- decode time；
- YOLO time；
- sampling time；
- CoTracker time；
- trajectory filter time；
- segment total time。

## 30. 性能统计

批处理结束后输出 `performance_summary.json`，包含：

- episodes processed；
- episodes failed；
- segments processed；
- segments failed；
- average decode time；
- average YOLO time；
- average CoTracker time；
- average segment processing time；
- total wall time；
- segments / second；
- video seconds / wall second。

## 31. 单元测试结果

远端轻量测试通过：`11/11`。

覆盖内容包括：

- segment keep/drop；
- 左闭右开无重复无遗漏；
- source offset 映射；
- query group merge/split；
- YAML `view_key` 指定视角；
- resume 判断；
- decode exact count；
- decode worker sentinel 行为；
- 异步聚合不串 episode。

远端系统 Python 和 `co-tracker` 环境没有安装 pytest，因此使用轻量 runner 直接执行所有 test 函数。

## 32. 当前已知限制

- `output.debug_visualization` 目前只是配置项，新批处理 writer 尚未完整输出调试可视化。
- ENV 采样虽然复用旧模块，但还没有和 LEFT / RIGHT 共享同一个 edge bundle。
- 目前只做二维稀疏轨迹，不做深度。
- 大规模运行前建议先用单 episode / 单 segment 做 GPU smoke test。

## 33. 后续支持多 GPU / depth 时推荐如何扩展

多 GPU 时，建议一个 model worker 对应一张 GPU：

```yaml
workers:
  model_workers: 2
  model_devices:
    - cuda:0
    - cuda:1
```

后续如果要引入 depth，应新增独立 depth sampler adapter，并继续保持：

```yaml
processing:
  depth_enabled: false
```

作为默认行为，不把深度模型混入当前 orchestration 核心。

## 运行示例

批量运行：

```bash
python scripts/run_batch_scene_tracking.py \
  --config configs/config.yaml
```

调试单个 segment：

```bash
python scripts/run_batch_scene_tracking.py \
  --config configs/config.yaml \
  --episode-index 0 \
  --segment-id 0 \
  --debug
```
