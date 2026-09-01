# SceneFlowTracker 数据处理流程说明

本文档说明当前重构后的 SceneFlowTracker 如何把 ABC / LeRobot manifest 中的指定视角视频转换为 episode 级稀疏轨迹 NPZ。

当前版本的核心变化是：正式 pipeline 不再使用旧的统一 `model_worker`，也不再导入外部业务工程里的 `RobotPointSampler`、`EnvironmentPointSampler`、`filter_robot_tracks`。YOLO、采样、完整视频解码、CoTracker、轨迹过滤、segment cache 和 episode 合并已经拆成独立阶段。

## 1. 输入

入口配置在：

```text
configs/config.yaml
```

核心输入字段：

```yaml
input:
  manifest_path: /mnt/data/chachaxu/save/abc_130k_v3/abc_130k_v3_train_all_views.json
  view_key: observation.images.top
```

框架只处理 `input.view_key` 指定的一个视角。实际视频路径与帧范围来自：

```python
episode["video_segments"][view_key]
```

支持的视角名称取决于 manifest，例如：

```text
observation.images.top
observation.images.left_wrist
observation.images.right_wrist
```

所有帧范围都使用左闭右开 `[start_frame, end_frame)`。

## 2. 配置结构

主配置只保留运行时常改参数：

- 输入 manifest 和 view_key
- fps 模式与 segment 长度
- CPU worker 数
- YOLO / CoTracker 路径、设备和 batch size
- 固定 query 点数
- segment cache 策略
- 输出路径和 NPZ 保存选项

算法细节放在独立文件：

```yaml
algorithm_configs:
  sampling: sampling.yaml
  trajectory_filter: trajectory_filter.yaml
```

配置加载顺序是：

```text
DEFAULT_CONFIG -> algorithm_configs -> config.yaml 主配置覆盖
```

因此主配置中的字段优先级最高。

## 3. Episode 与 Segment

`scene_flow_tracker/manifest.py` 负责把 manifest 记录转成 `EpisodeJob`。

`scene_flow_tracker/segment_planner.py` 负责把一个 episode 切成连续、不重叠的 `SegmentJob`：

```text
segment_0: [0, segment_frames)
segment_1: [segment_frames, 2 * segment_frames)
...
tail: keep 或 drop
```

每个 `SegmentJob` 同时保存：

- episode 内帧范围：`episode_start_frame / episode_end_frame`
- 物理视频帧范围：`source_start_frame / source_end_frame`

合并输出时会校验所有 segment 在 episode 和 source 两个坐标系里都连续。

## 4. 新 Pipeline 拓扑

正式入口：

```text
scene_flow_tracker/runner.py
scene_flow_tracker/orchestration/pipeline_runner.py
```

数据流为：

```text
EpisodeJob
  -> SegmentJob
  -> first-frame decode pool
  -> YOLO batch stage
  -> sampling pool
  -> full-segment decode pool
  -> CoTracker segment batch stage
  -> trajectory filter pool
  -> segment cache
  -> episode NPZ + summary.json
```

旧文件 `workers/model_worker.py`、`pipeline/model_loader.py`、`pipeline/segment_processor.py` 保留在仓库中作为历史兼容代码，但新入口不再调用它们。

## 5. 视频解码

解码工具在：

```text
scene_flow_tracker/video_decode.py
```

当前优先使用系统 `ffmpeg/ffprobe` 精确读取帧，OpenCV 只作为兜底。这样可以处理 OpenCV 对 AV1 等编码支持不完整导致的解码失败。

首帧阶段只读取每个 segment 的第一帧：

```text
decode_first_frame_rgb(video_path, source_start_frame)
```

完整 segment 阶段在采样完成后才读取整段：

```text
decode_frames_rgb(video_path, source_start_frame, frame_count)
```

这个顺序避免了 YOLO 或采样失败时提前把整段视频放进内存。

## 6. YOLO Batch

YOLO wrapper 在：

```text
scene_flow_tracker/inference/yolo_model.py
```

YOLO 只处理每个 segment 的首帧。多个首帧按 `models.yolo.batch_size` 组成 batch，一次调用 `model.predict(...)`。

YOLO 可以返回：

- 0 个机械臂 bbox
- 1 个机械臂 bbox
- 2 个机械臂 bbox
- 多于 2 个候选 bbox

缺 bbox 不会让 segment 失败。slot 分配逻辑是：

1. 如果 YOLO class name 含 left/right，优先按类别名分配。
2. 否则按 bbox 中心点 x 坐标分配。
3. 只有一个 bbox 时，bbox 中心在图像左半边则作为 left，否则作为 right。
4. 没有 bbox 时 left/right 都标记为无效。

最终保存到 segment/episode 结果中的 YOLO slot 固定为：

```text
left
right
```

## 7. 固定 Query 点数

query 分配逻辑在：

```text
scene_flow_tracker/algorithms/query_allocator.py
scene_flow_tracker/workers/sampling_worker.py
```

主配置：

```yaml
sampling:
  query_allocation:
    total_query_points: 300
    points_per_detected_arm: 100
```

每个 segment 最终 query 数必须严格等于 `total_query_points`。

分配规则：

```text
左臂 bbox 有效 -> 左臂目标点数 = points_per_detected_arm
右臂 bbox 有效 -> 右臂目标点数 = points_per_detected_arm
缺失 bbox -> 对应机械臂目标点数 = 0
环境点目标数 = total_query_points - 实际左臂点数 - 实际右臂点数
```

如果某个机械臂 bbox 有效但候选点不足，少掉的点也会转给环境点。采样后会做去重和越界校验；如果无法凑够固定点数，该 segment 才会失败。

## 8. 机械臂采样

机械臂采样在：

```text
scene_flow_tracker/algorithms/robot_sampling.py
```

它只使用当前仓库内部实现，不导入外部业务采样器。

候选区域来自 YOLO bbox，主要证据包括：

- bbox 内像素 mask
- Canny 边缘
- 小连通域过滤
- 机械臂黑/白颜色先验
- Shi-Tomasi / cornerMinEigenVal trackability
- 空间均衡采样

采样输出：

```text
points_xy [N,2]
features
stats
```

坐标顺序为 `(x, y)`，坐标空间为原始图像像素。

## 9. 环境采样

环境采样在：

```text
scene_flow_tracker/algorithms/environment_sampling.py
```

环境采样会排除有效机械臂 bbox，并排除已经被机械臂采样占用的点。候选点优先来自边缘和可追踪性高的区域，不足时再退化到空间网格和有效图像区域随机补点。

环境采样的职责有两个：

- 为背景/非机械臂区域提供参考轨迹。
- 吸收 YOLO 漏检或机械臂候选不足导致的 query 缺额，保证每段 query 总数固定。

## 10. CoTracker Segment Batch

CoTracker wrapper 在：

```text
scene_flow_tracker/inference/cotracker_model.py
```

完整 segment 解码后，视频帧会通过 `multiprocessing.shared_memory` 写入共享内存引用：

```text
SharedArrayRef(name, shape, dtype, nbytes, owner, debug_id)
```

CoTracker 阶段按 key 聚合：

```text
(T, H, W, N)
```

正常 segment 满足相同 `T/H/W/N` 时，会按 `models.cotracker.segment_batch_size` 组成真正的 segment batch：

```text
video:   [B, T, C, H, W]
queries: [B, N, 3]
```

其中 query 的三个值为：

```text
[t, x, y]
```

当前所有点都在 segment 起始帧采样，所以 `t=0`。

尾段如果长度小于 `video.segment_frames`，不会和正常段硬拼 batch，而是单独推理。

`models.cotracker.point_chunk_size` 表示单次 CoTracker 调用中 query 点维度的 chunk size。它和 segment batch size 是两个概念：

- `segment_batch_size` 控制一次推理几个 segment。
- `point_chunk_size` 控制每个 segment 的点太多时是否拆点。

## 11. 轨迹过滤

轨迹过滤在：

```text
scene_flow_tracker/algorithms/trajectory_filter.py
scene_flow_tracker/workers/filter_worker.py
```

输入：

```text
tracks_xy [N,T,2]
visibility [N,T]
query_group [N]
```

输出：

```text
tracks_xy_raw
tracks_xy_smooth
track_state
motion_state
usable_for_robot_scene_flow
filter_features
```

当前默认策略：

- 不因为 CoTracker visibility 低直接判 failed。
- 保留 static / moving / uncertain。
- 过滤 jitter。
- partial track 是否保留由配置控制。

过滤特征包括：

- visibility_ratio
- net_displacement
- path_length
- path_efficiency
- jitter_rms
- jitter_residual_ratio
- turn_consistency
- turn_angle_mad
- normalized_jerk
- direction_reversal_ratio

## 12. Segment Cache

segment cache 在：

```text
scene_flow_tracker/storage/segment_cache.py
```

每个 segment 完成后先写临时缓存：

```text
output_root/dataset/episode_id/.segment_cache/<safe_view_name>/segment_000000.npz
```

cache 中保存：

- cache schema version
- 配置 fingerprint
- SegmentJob
- status / error_code / error_message
- detections
- sampling
- groups
- timings

断点续跑时，runner 会先检查每个 segment cache 是否存在、非空、fingerprint 一致、segment frame range 一致。

如果：

```yaml
cache:
  retry_cached_failed_segments: false
```

则失败 segment 的 cache 也会被复用，便于完整输出 episode 的失败状态。

如果 episode 级 NPZ 和 summary 成功写入并通过内部 payload 校验，且：

```yaml
cache:
  delete_after_successful_merge: true
```

则该 episode 的 `.segment_cache` 会被删除。

## 13. Episode 合并与输出

最终 writer 仍复用：

```text
scene_flow_tracker/storage/writers.py
```

每个 episode 输出：

```text
output_root/
└── dataset/
    └── episode_id/
        ├── observation.images.top_scene_tracks.npz
        └── observation.images.top_summary.json
```

批处理级输出：

```text
processing_manifest.jsonl
performance_summary.json
```

resume 的完成判断基于 `*_scene_tracks.npz + *_summary.json`，不依赖日志。

## 14. NPZ Schema

当前 schema 版本：

```text
1.2
```

定义在：

```text
scene_flow_tracker/storage/schema.py
```

关键字段：

```text
segment_ids [S]
episode_start_frames [S]
episode_end_frames [S]
source_start_frames [S]
source_end_frames [S]
segment_lengths [S]
segment_status [S]
episode_frame_indices [S,Tmax]
source_frame_indices [S,Tmax]
```

YOLO 字段：

```text
yolo_bboxes [S,1,2,4]
yolo_bbox_confidence [S,1,2]
yolo_bbox_valid [S,1,2]
```

三组轨迹：

```text
left_*
right_*
env_*
```

以 left 为例：

```text
left_num_points [S]
left_point_valid [S,Nmax]
left_query_xy [S,Nmax,2]
left_tracks_raw [S,Tmax,Nmax,2]
left_tracks_smooth [S,Tmax,Nmax,2]
left_visibility [S,Tmax,Nmax]
left_track_state [S,Nmax]
left_motion_state [S,Nmax]
left_usable [S,Nmax]
left_group_status [S]
```

变长点数使用 padding 表示：

- `*_num_points` 表示每段真实点数。
- `*_point_valid` 标记真实点。
- query / tracks padding 使用 NaN。
- visibility/state padding 使用默认枚举值。
- 禁止 `dtype=object`。

## 15. 状态枚举

`TrackState`：

```text
0 INVALID_OR_PADDING
1 VALID
2 PARTIAL
3 FAILED
```

`MotionState`：

```text
0 UNKNOWN_OR_PADDING
1 STATIC
2 MOVING
3 JITTER
4 UNCERTAIN
```

`SegmentStatus`：

```text
0 UNKNOWN_OR_PADDING
1 OK
2 PARTIAL
3 FAILED
```

`GroupStatus`：

```text
0 UNKNOWN
1 OK
2 NO_DETECTION
3 NO_CANDIDATES
4 TRACK_FAILED
5 PROCESSING_FAILED
```

## 16. 运行命令

常规运行：

```bash
cd /mnt/workspace/SceneFlowTracker
bash example/run.sh
```

处理一个 episode：

```bash
bash example/run.sh --episode-index 0 --no-resume
```

只处理一个 segment：

```bash
bash example/run.sh --episode-index 0 --segment-id 0 --debug --no-resume
```

新入口等价命令：

```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

性能测试：

```bash
python scripts/benchmark_pipeline.py --config configs/config.yaml --max-episodes 1 --no-resume
```

停止 `example/run.sh` 启动的任务：

```bash
bash example/kill.sh
```

校验输出：

```bash
python scripts/validate_scene_tracks_npz.py /path/to/observation.images.top_scene_tracks.npz
```

查看输出：

```bash
python scripts/inspect_scene_tracks_npz.py /path/to/observation.images.top_scene_tracks.npz
```

## 17. 验证建议

修改 pipeline 后建议至少执行：

- `python -m compileall scene_flow_tracker scripts example`
- `python -m pytest -q`
- 使用真实 ABC manifest、YOLO 权重、CoTracker 权重处理 `max_episodes=1, segment_id=0`
- 用 `scripts/validate_scene_tracks_npz.py` 校验生成的 episode NPZ
