# SceneFlowTracker 数据处理流程说明

本文档说明当前 SceneFlowTracker 框架如何把 ABC / LeRobot manifest 中指定视角的视频，处理成 episode 级稀疏二维轨迹结果。当前框架的主目标是：

- 从 manifest 中按 `view_key` 读取一个视角的视频片段。
- 将每个 episode 切分为固定长度 segment。
- 对每个 segment 的起始帧做 YOLO 机械臂 bbox 检测。
- 在机械臂 bbox 和环境区域内生成固定数量的 CoTracker query 点。
- 对每个 segment 独立做 CoTracker 点追踪。
- 对轨迹做平滑、运动状态分类和 jitter 过滤。
- 合并为 episode 级 `*_scene_tracks.npz` 与 `*_summary.json`。

当前版本只输出 RGB 主视角/腕部视角上的稀疏二维轨迹，不输出深度场景流。坐标均为原始处理图像像素坐标，顺序为 `(x, y)`。

## 1. 入口与运行方式

推荐入口：

```bash
cd /mnt/workspace/SceneFlowTracker
bash example/run.sh
```

`example/run.sh` 会激活 `co-tracker` conda 环境，然后调用：

```bash
python example/main.py \
  --manifest-path "${MANIFEST_PATH}" \
  --view-key "${VIEW_KEY}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"
```

默认环境变量：

```bash
CONDA_ENV=co-tracker
MANIFEST_PATH=/mnt/data/chachaxu/save/abc_130k_v3/abc_130k_v3_train_all_views.json
VIEW_KEY=observation.images.top
OUTPUT_ROOT=/mnt/data/chachaxu/dataset/abc_130k_v3_sceneflow
```

常用覆盖示例：

```bash
# 只处理 top 视角，输出到指定目录
VIEW_KEY=observation.images.top \
OUTPUT_ROOT=/mnt/data/chachaxu/dataset/abc_130k_v3_sceneflow \
bash example/run.sh

# 只跑一个 episode
bash example/run.sh --episode-index 0

# 只跑一个 segment，适合 smoke test
bash example/run.sh --episode-index 0 --segment-id 0

# 调整吞吐参数
bash example/run.sh \
  --first-frame-decode-workers 8 \
  --sampling-workers 8 \
  --segment-decode-workers 4 \
  --filter-workers 8 \
  --yolo-batch-size 256 \
  --cotracker-segment-batch-size 24
```

`example/main.py` 支持的主要 CLI 参数：

| 参数 | 作用 |
| --- | --- |
| `--config` | 指定基础配置文件，默认 `configs/config.yaml`。 |
| `--manifest-path` | 覆盖输入 manifest 路径。 |
| `--view-key` | 覆盖处理视角，例如 `observation.images.top`。 |
| `--output-root` | 覆盖输出根目录。 |
| `--segment-frames` | 覆盖每段帧数。 |
| `--first-frame-decode-workers` | 覆盖首帧解码并发数。 |
| `--sampling-workers` | 覆盖采样并发数。 |
| `--segment-decode-workers` | 覆盖完整 segment 解码并发数。 |
| `--filter-workers` | 覆盖轨迹过滤并发数。 |
| `--yolo-batch-size` | 覆盖 YOLO 首帧 batch size。 |
| `--cotracker-segment-batch-size` | 覆盖 CoTracker segment batch size。 |
| `--total-query-points` | 覆盖每个 segment 的固定 query 点总数。 |
| `--episode-index` | 只处理指定 `episode_index`。 |
| `--max-episodes` | 从筛选后的列表中最多处理多少个 episode。 |
| `--segment-id` | 只处理指定 segment。 |
| `--resume` / `--no-resume` | 是否跳过已经完成的 episode 结果。 |
| `--debug` | 单 worker、小 batch、DEBUG 日志，适合定位问题。 |

## 2. 配置文件结构

主配置文件：

```text
configs/config.yaml
```

算法细节配置：

```text
configs/sampling.yaml
configs/trajectory_filter.yaml
```

加载顺序为：

```text
DEFAULT_CONFIG -> algorithm_configs 指向的细节配置 -> config.yaml 主配置覆盖 -> CLI 参数覆盖
```

因此优先级从低到高是：默认值、细节配置、主配置、命令行参数。主配置主要保留吞吐量、输入输出格式、模型路径等常用项；采样和轨迹过滤阈值放在单独文件中。

## 3. 主配置参数

### 3.1 input

```yaml
input:
  manifest_path: /mnt/data/chachaxu/save/abc_130k_v3/abc_130k_v3_train_all_views.json
  view_key: observation.images.top
  frame_range_semantics: half_open
  strict_validation: true
```

| 参数 | 说明 |
| --- | --- |
| `manifest_path` | ABC / LeRobot manifest JSON 路径。当前要求 manifest 为 JSON 数组，每条记录包含 `video_segments`。 |
| `view_key` | 只处理一个指定视角。框架会读取 `episode["video_segments"][view_key]`，例如 `observation.images.top`、`observation.images.left_wrist`、`observation.images.right_wrist`。 |
| `frame_range_semantics` | 帧范围语义，当前固定为左闭右开 `[start_frame, end_frame)`。 |
| `strict_validation` | 为 `true` 时，manifest 缺关键字段会报错；为 `false` 时跳过坏记录。 |

### 3.2 video

```yaml
video:
  fps_mode: manifest
  fixed_fps: 30.0
  segment_frames: 30
  tail_policy: keep
```

| 参数 | 说明 |
| --- | --- |
| `fps_mode` | `manifest` 表示使用 manifest 中的视频 fps；`fixed` 表示用 `fixed_fps` 覆盖。 |
| `fixed_fps` | `fps_mode=fixed` 时生效。 |
| `segment_frames` | 每个 segment 的帧数。30 表示在 30fps 下每段 1 秒；15 表示 0.5 秒。 |
| `tail_policy` | `keep` 保留最后不足 `segment_frames` 的尾段；`drop` 丢弃尾段。 |

当前 segment 之间是连续、不重叠的独立片段。每个 segment 的 query 点都在该 segment 起始帧重新采样，CoTracker 也只在该 segment 内追踪。

### 3.3 workers

```yaml
workers:
  first_frame_decode_workers: 8
  sampling_workers: 8
  segment_decode_workers: 4
  filter_workers: 8
```

| 参数 | 说明 |
| --- | --- |
| `first_frame_decode_workers` | 首帧解码并发数。只读取每个 segment 的起始帧，用于 YOLO 和采样。 |
| `sampling_workers` | 采样并发数。根据 YOLO bbox、边缘、颜色先验和可追踪性生成 query 点。 |
| `segment_decode_workers` | 完整 segment 解码并发数。采样成功后才读取整段视频，避免无效内存占用。 |
| `filter_workers` | 轨迹过滤并发数。负责平滑、运动状态分类、jitter 过滤和 usable 标记。 |

这些 worker 是 CPU 侧并发。YOLO 和 CoTracker 的 GPU batch size 分别由 `models.yolo.batch_size` 和 `models.cotracker.segment_batch_size` 控制，可以和 worker 数不同。

### 3.4 pipeline

```yaml
pipeline:
  max_inflight_segments: 64
```

| 参数 | 说明 |
| --- | --- |
| `max_inflight_segments` | 单次推进到 pipeline 内存中的最大 segment 数。调大可提高 GPU batch 填充率和整体吞吐，但会增加 CPU 内存、共享内存和挂载盘读取压力。 |

实际执行时，一个 episode 会先被切成 segments，再按 `max_inflight_segments` 分 chunk 处理。每个 chunk 内执行：首帧解码、YOLO batch、采样、整段解码、CoTracker batch、过滤、cache 写入。

### 3.5 models.yolo

```yaml
models:
  yolo:
    model_path: /mnt/workspace/instance_exp/yolo_detect/arm_detect/runs/yolo11n_arm_abc130k_v3_train_vlm_9_1_640_20260826/weights/best.pt
    device: cuda:0
    batch_size: 256
    imgsz: 224
    conf: 0.5
    iou: 0.7
```

| 参数 | 说明 |
| --- | --- |
| `model_path` | 机械臂 bbox YOLO 权重路径。 |
| `device` | YOLO 推理设备。双卡时通常可放在 `cuda:0`。 |
| `batch_size` | YOLO 首帧 batch size。输入是多个 segment 起始帧。 |
| `imgsz` | YOLO 输入尺寸。当前默认 224。 |
| `conf` | 置信度阈值。低于该阈值的候选框会被 YOLO 结果过滤掉。漏检多时可降低，误检多时可升高。 |
| `iou` | NMS IoU 阈值。多个框重叠过高时，用该阈值控制去重。 |

YOLO 当前只检测每个 segment 的第一帧，不逐帧检测。检测结果经过 slot 分配后只保留到 `left` 和 `right` 两个机械臂槽位。

slot 分配规则：

1. 如果 YOLO 类别名包含 left/right，优先按类别名分配。
2. 否则按 bbox 中心点 x 坐标排序。
3. 只有一个 bbox 时，中心点在图像左半边分配为 `left`，否则分配为 `right`。
4. 没有 bbox 时，`left_bbox=None` 且 `right_bbox=None`。

没有机械臂 bbox 不会让 segment 失败。该机械臂组点数为 0，query 配额转给环境点。

### 3.6 models.cotracker

```yaml
models:
  cotracker:
    source_root: /mnt/data/chachaxu/instance_tracking_environment/co-tracker
    model_path: /mnt/data/chachaxu/model/cotracker3/scaled_offline.pth
    device: cuda:1
    segment_batch_size: 24
    point_chunk_size: 1024
```

| 参数 | 说明 |
| --- | --- |
| `source_root` | CoTracker 官方源码目录，运行时加入 `PYTHONPATH`。 |
| `model_path` | CoTracker checkpoint。 |
| `device` | CoTracker 推理设备。显存主要消耗来自这里。 |
| `segment_batch_size` | segment 维度 batch size。相同 `T/H/W/N` 的 segment 会拼成 `[B,T,C,H,W]` 一次推理。 |
| `point_chunk_size` | 点维度 chunk size。单个 segment query 点特别多时按点拆分，避免一次送入过多点。 |

`segment_batch_size` 和 `point_chunk_size` 是两个维度：

- `segment_batch_size` 控制一次推理几个视频片段。
- `point_chunk_size` 控制每个片段的点数是否拆 chunk。

当前 query 格式为 `[t, x, y]`，所有点都在 segment 起始帧采样，所以 `t=0`。

### 3.7 sampling

```yaml
sampling:
  seed: 0
  query_allocation:
    total_query_points: 300
    points_per_detected_arm: 100
```

| 参数 | 说明 |
| --- | --- |
| `seed` | 采样随机种子。每个 episode/segment 会派生稳定 seed，保证重复运行可复现。 |
| `total_query_points` | 每个 segment 固定 query 点总数。 |
| `points_per_detected_arm` | 每个成功检测到 bbox 的机械臂目标点数。 |

query 分配规则：

```text
left bbox 有效  -> left 目标点数 = points_per_detected_arm
right bbox 有效 -> right 目标点数 = points_per_detected_arm
对应 bbox 缺失 -> 该机械臂目标点数 = 0
environment 点数 = total_query_points - left 实际点数 - right 实际点数
```

如果某个机械臂 bbox 有效但候选点不足，缺额转给环境点。最终 segment 的总 query 数必须等于 `total_query_points`，否则该 segment 才会失败。

### 3.8 cache

```yaml
cache:
  enabled: true
  dirname: .segment_cache
  delete_after_successful_merge: true
  retry_cached_failed_segments: false
```

| 参数 | 说明 |
| --- | --- |
| `enabled` | 是否启用 segment 级缓存。 |
| `dirname` | 每个 episode 输出目录下的缓存目录名。 |
| `delete_after_successful_merge` | episode NPZ 和 summary 成功写入后是否删除 segment cache。 |
| `retry_cached_failed_segments` | 读到 FAILED cache 时是否重新跑该 segment。 |

cache 用于断点续跑。cache 校验包括：文件存在、非空、配置 fingerprint 一致、segment 帧范围一致。

### 3.9 output

```yaml
output:
  output_root: /mnt/data/chachaxu/dataset/abc_130k_v3_sceneflow
  schema_version: "1.2"
  save_npz: true
  save_summary_json: true
  compression: compressed
  save_raw_tracks: true
  save_smooth_tracks: true
  save_features: true
  save_sampling_features: true
  save_filter_features: true
  save_cotracker_confidence: true
  debug_visualization: false
```

| 参数 | 说明 |
| --- | --- |
| `output_root` | 输出根目录。实际 episode 输出路径为 `output_root/dataset/episode_id/`。 |
| `schema_version` | 当前 NPZ schema 版本，当前为 `1.2`。 |
| `save_npz` | 是否保存 episode 级 NPZ。 |
| `save_summary_json` | 是否保存 episode 级 summary JSON。 |
| `compression` | `compressed` 使用 `np.savez_compressed`；`stored` 使用 `np.savez`。 |
| `save_raw_tracks` | 是否保存 CoTracker 原始轨迹。当前 writer 会写入 `*_tracks_raw`。 |
| `save_smooth_tracks` | 是否保存平滑轨迹。当前 writer 会写入 `*_tracks_smooth`。 |
| `save_features` | 兼容开关；建议保持 `true`。 |
| `save_sampling_features` | 是否保存采样阶段特征，例如边缘强度、颜色评分、trackability。 |
| `save_filter_features` | 是否保存轨迹过滤诊断特征。 |
| `save_cotracker_confidence` | 如果 CoTracker wrapper 返回 confidence，则写入 NPZ。 |
| `debug_visualization` | 预留调试开关。批量主流程默认不输出视频可视化。 |

### 3.10 batch

```yaml
batch:
  resume: true
  continue_on_segment_error: true
  group_by_physical_video: true
  atomic_write: true
```

| 参数 | 说明 |
| --- | --- |
| `resume` | 是否扫描已有结果并跳过已完成 episode/view。 |
| `continue_on_segment_error` | segment 失败时是否继续处理后续 segment。 |
| `group_by_physical_video` | manifest job 是否按物理视频路径排序，减少随机 seek。 |
| `atomic_write` | NPZ 和 summary 是否先写临时文件再原子替换。 |

resume 完成判断基于：

```text
<view_safe_name>_scene_tracks.npz
<view_safe_name>_summary.json
```

这两个文件都存在且非空，才认为该 episode/view 完成。

## 4. 采样细节配置

采样细节在：

```text
configs/sampling.yaml
```

### 4.1 机械臂采样

模块：

```text
scene_flow_tracker/algorithms/robot_sampling.py
```

机械臂候选区域来自 YOLO bbox。采样器会综合以下证据给候选点打分：

- bbox 内像素区域。
- Canny 边缘。
- 小连通域过滤。
- 黑/白机械臂颜色先验。
- 边缘两侧颜色对比。
- bbox 拓扑过滤。
- Shi-Tomasi / cornerMinEigenVal 可追踪性。
- 空间均衡采样。

主要配置：

| 参数 | 说明 |
| --- | --- |
| `robot_sampling.left_robot.target_points` | 左机械臂目标采样点数。通常由主配置 `points_per_detected_arm` 覆盖。 |
| `robot_sampling.right_robot.target_points` | 右机械臂目标采样点数。 |
| `robot_sampling.edge.threshold1/threshold2` | Canny 低/高阈值。 |
| `robot_sampling.edge.min_component_pixels` | 过滤小边缘连通域。 |
| `robot_sampling.color.enabled` | 是否启用黑/白颜色先验。 |
| `robot_sampling.color.black_l_max` | LAB 空间中黑色 L 通道上限。 |
| `robot_sampling.color.white_l_min` | LAB 空间中白色 L 通道下限。 |
| `robot_sampling.topology.enabled` | 是否启用 bbox 拓扑过滤，减少穿出 bbox 的背景长边缘。 |
| `robot_sampling.trackability.enabled` | 是否启用可追踪性评分。 |
| `robot_sampling.score_weights.*` | 边缘、颜色、拓扑、可追踪性等证据的综合权重。 |
| `robot_sampling.spatial_sampling.*` | 控制空间网格、最小点距、单格最多点数、候选点上限。 |
| `robot_sampling.candidate_policy.*` | 控制是否允许 robot、ambiguous、background 候选进入补点。 |

输出字段包括：

```text
points_xy [N,2]
sampling_score
edge_strength
trackability_score
color_score
topology_score
candidate_level
stats
```

### 4.2 环境采样

模块：

```text
scene_flow_tracker/algorithms/environment_sampling.py
```

环境采样会排除有效机械臂 bbox，并排除已经被机械臂采样占用的点。候选点优先来自边缘和高可追踪性区域，不足时退化到空间网格和有效图像区域补点。

主要配置：

| 参数 | 说明 |
| --- | --- |
| `environment_sampling.enabled` | 是否启用环境点采样。 |
| `environment_sampling.target_points` | 环境目标点数。实际点数由 query 分配器根据机械臂缺额调整。 |
| `environment_sampling.robot_exclusion_margin_px` | 在机械臂 bbox 外额外排除的 margin。 |
| `environment_sampling.edge.min_component_pixels` | 过滤环境边缘的小连通域。 |
| `environment_sampling.trackability.enabled` | 是否启用可追踪性评分。 |
| `environment_sampling.score_weights.*` | 环境采样评分权重。 |
| `environment_sampling.spatial_sampling.*` | 控制环境点空间均衡采样。 |

环境点有两个作用：

- 给背景/非机械臂区域提供参考轨迹。
- 吸收 YOLO 漏检或机械臂候选不足导致的 query 缺额，保证 CoTracker 每段点数稳定。

## 5. 轨迹过滤配置

轨迹过滤细节在：

```text
configs/trajectory_filter.yaml
```

核心模块：

```text
scene_flow_tracker/algorithms/trajectory_filter.py
scene_flow_tracker/algorithms/trajectory_smoothing.py
scene_flow_tracker/algorithms/trajectory_features.py
scene_flow_tracker/workers/filter_worker.py
```

输入：

```text
tracks_xy_raw [N,T,2]
visibility [N,T]
query_group [N]
```

输出：

```text
tracks_xy_smooth [N,T,2]
track_state [N]
motion_state [N]
usable_for_robot_scene_flow [N]
filter_features
```

当前默认策略是 structured-motion-aware v2：

- 不因为 CoTracker visibility 低直接判 failed。
- 保留 `STATIC`、`MOVING`、`UNCERTAIN`。
- 过滤 `JITTER`。
- `PARTIAL` 轨迹是否保留由 `output_policy.keep_partial_tracks` 控制。

主要配置：

| 参数 | 说明 |
| --- | --- |
| `tracking_validity.enabled` | 是否按 visibility 规则硬判 track failed。当前默认 `false`，避免机械臂遮挡导致大量误删。 |
| `jump_filter.enabled` | 是否启用异常大跳变 MAD 过滤。 |
| `smoothing.enabled` | 是否对轨迹平滑。 |
| `smoothing.method` | 平滑方法，当前默认 `savgol`。 |
| `normalization.method` | 运动幅度归一化方法，当前为 `bbox_diagonal`。 |
| `direction_analysis.min_speed_for_direction` | 方向分析时忽略低速微小位移。 |
| `motion_classification.structured_motion.*` | 保护平滑结构化运动，避免旋转/圆周运动被误判为 jitter。 |
| `motion_classification.jitter_v2.*` | jitter 的残差、转向、jerk 证据阈值。 |
| `motion_classification.static.*` | 静止点判定阈值。 |
| `motion_classification.moving.*` | 运动点判定阈值。 |
| `local_consistency.*` | 局部一致性诊断，默认不作为硬过滤。 |
| `local_affine.*` | 更重的局部仿射一致性诊断，默认关闭。 |
| `output_policy.*` | 控制 static/moving/jitter/uncertain/partial 是否最终 usable。 |

保存的过滤特征包括：

```text
visibility_ratio
net_displacement
path_length
path_efficiency
jitter_rms
jitter_residual_ratio
turn_consistency
turn_angle_mad
normalized_jerk
direction_reversal_ratio
```

## 6. 模块组成

当前正式 pipeline 的主要代码模块如下：

| 模块 | 职责 |
| --- | --- |
| `scene_flow_tracker/config.py` | 加载、合并、校验配置。 |
| `scene_flow_tracker/manifest.py` | 读取 manifest，生成 `EpisodeJob`。 |
| `scene_flow_tracker/segment_planner.py` | 将 episode 切分为 `SegmentJob`。 |
| `scene_flow_tracker/video_decode.py` | 使用 ffmpeg/ffprobe 优先解码视频，OpenCV 兜底。 |
| `scene_flow_tracker/inference/yolo_model.py` | 加载 YOLO，批量检测 segment 起始帧 bbox。 |
| `scene_flow_tracker/inference/cotracker_model.py` | 加载 CoTracker，按 segment batch 和 point chunk 追踪 query。 |
| `scene_flow_tracker/algorithms/query_allocator.py` | 根据 bbox 是否存在分配 left/right/env 点数。 |
| `scene_flow_tracker/algorithms/robot_sampling.py` | 在机械臂 bbox 内选择机械臂 query 点。 |
| `scene_flow_tracker/algorithms/environment_sampling.py` | 在非机械臂区域选择环境 query 点。 |
| `scene_flow_tracker/algorithms/edges.py` | Canny 边缘与连通域处理。 |
| `scene_flow_tracker/algorithms/trackability.py` | 可追踪性评分。 |
| `scene_flow_tracker/algorithms/trajectory_features.py` | 计算轨迹运动特征。 |
| `scene_flow_tracker/algorithms/trajectory_smoothing.py` | 轨迹平滑。 |
| `scene_flow_tracker/algorithms/trajectory_filter.py` | 轨迹状态分类和 usable 判定。 |
| `scene_flow_tracker/workers/*` | 各阶段 worker 封装。 |
| `scene_flow_tracker/orchestration/pipeline_runner.py` | 当前正式调度入口，串联所有阶段。 |
| `scene_flow_tracker/storage/segment_cache.py` | segment cache 读写和校验。 |
| `scene_flow_tracker/storage/resume.py` | 扫描已有 episode/view 输出，实现快速跳过。 |
| `scene_flow_tracker/storage/writers.py` | 写 episode 级 NPZ、summary 和 processing manifest。 |
| `scene_flow_tracker/storage/schema.py` | schema 版本、枚举值、字段约定。 |

历史兼容代码：

```text
scene_flow_tracker/workers/model_worker.py
scene_flow_tracker/pipeline/model_loader.py
scene_flow_tracker/pipeline/segment_processor.py
```

这些文件保留在仓库中，但当前 `example/main.py -> scene_flow_tracker.runner -> orchestration.pipeline_runner` 不再调用旧的统一 `model_worker` 流程。

## 7. 数据处理流程

### 7.1 Manifest 读取

`manifest.py` 从 `input.manifest_path` 读取 JSON 数组。每条记录会转换为 `EpisodeJob`，主要字段包括：

```text
dataset
episode_id
episode_index
task_index
task
instruction
view_key
physical_video_path
source_start_frame
source_end_frame
manifest_fps
effective_fps
raw_record
```

框架只读取 `input.view_key` 指定的一个视角：

```python
episode["video_segments"][view_key]
```

如果需要处理多个视角，需要分别以不同 `VIEW_KEY` 启动任务。

### 7.2 Episode 切分

`segment_planner.py` 根据 `video.segment_frames` 和 `video.tail_policy` 切分 episode。

每个 `SegmentJob` 包含两套帧坐标：

```text
episode_start_frame / episode_end_frame
source_start_frame / source_end_frame
```

二者关系为：

```text
source_start_frame = episode.source_start_frame + episode_start_frame
source_end_frame   = episode.source_start_frame + episode_end_frame
```

所有帧范围都是左闭右开 `[start, end)`。episode 合并时会校验所有 segment 在 episode 坐标和 source 坐标中都连续。

### 7.3 Resume 扫描

如果 `batch.resume=true` 且不是指定单个 `segment_id`，runner 会先调用：

```text
scene_flow_tracker/storage/resume.py
```

扫描策略不是逐条 manifest 做 `stat`，而是先扫描输出根目录中已有 episode 目录，再和本次选中的 `(dataset, episode_id, view)` 求交集。这样在大量 episode 和网络文件系统上更快。

完成条件：

```text
output_root/dataset/episode_id/<safe_view_name>_scene_tracks.npz 存在且非空
output_root/dataset/episode_id/<safe_view_name>_summary.json 存在且非空
```

例如：

```text
/mnt/data/chachaxu/dataset/abc_130k_v3_sceneflow/
└── abc_130k_v3_train/
    └── abc_130k_v3_train__episode_000000/
        ├── observation.images.top_scene_tracks.npz
        └── observation.images.top_summary.json
```

### 7.4 首帧解码

每个 chunk 内先并发读取 segment 起始帧：

```text
decode_first_frame(job)
```

底层优先使用系统 `ffmpeg/ffprobe`，失败时使用 OpenCV 兜底。首帧只用于 YOLO 和采样，不会提前读取整段视频。

### 7.5 YOLO 首帧检测

首帧解码完成后，`YoloModel.predict_batch` 按 `models.yolo.batch_size` 组成 batch，一次检测多个 segment 的起始帧。

输出为 `YoloDetectionResult`，包含：

```text
left_bbox_xyxy
right_bbox_xyxy
left_confidence
right_confidence
left_bbox_valid
right_bbox_valid
raw_detections
assignment_method
image_width
image_height
```

`raw_detections` 是 YOLO 过滤 `conf` 和 NMS 后留下的候选框。slot 分配后会写入 `left` / `right`。

### 7.6 Query 分配与采样

采样 worker 输入首帧和 YOLO 检测结果，输出固定数量 query。

流程：

1. `query_allocator.py` 根据 bbox 是否有效分配 left/right/env 目标点数。
2. `robot_sampling.py` 在有效机械臂 bbox 内生成机械臂候选点并打分。
3. `environment_sampling.py` 在非机械臂区域生成环境候选点并打分。
4. `query_builder.py` 合并三组点，生成统一 `query_xy` 和 `query_group`。

`query_group` 约定：

```text
0 = left
1 = right
2 = environment
```

采样成功后，每个 segment 的总点数固定为 `sampling.query_allocation.total_query_points`。

### 7.7 完整 Segment 解码与共享内存

采样成功后才读取完整 segment：

```text
decode_segment_rgb(item.job)
```

解码得到的帧数组会放入 `multiprocessing.shared_memory`，用 `SharedArrayRef` 在阶段间传递引用：

```text
SharedArrayRef(name, shape, dtype, nbytes, owner, debug_id)
```

这样可以减少 Python 对大视频帧数组的重复拷贝。CoTracker 阶段完成后会释放共享内存。

### 7.8 CoTracker Batch 推理

CoTracker 阶段按 `(T, H, W, N)` 对 segment 分桶，只有相同形状的 segment 才会拼成真正 batch。

batch 输入形状：

```text
video:   [B, T, C, H, W]
queries: [B, N, 3]
```

其中 query 为：

```text
[t, x, y]
```

当前所有 query 都来自 segment 第一帧，所以 `t=0`。尾段如果长度小于 `video.segment_frames`，不会和正常 segment 硬拼 batch，而是单独推理。

输出进入 `TrackResult`，包含：

```text
tracks_xy_raw [N,T,2]
visibility [N,T]
confidence [N,T]  # 如果 CoTracker wrapper 返回
query_xy [N,2]
query_group [N]
```

### 7.9 轨迹过滤

过滤 worker 调用 `filter_track_result`，对每个点轨迹计算特征、平滑并分类。

轨迹状态 `track_state`：

```text
valid
partial
failed
```

运动状态 `motion_state`：

```text
static
moving
jitter
uncertain
```

当前默认不因为 visibility 低直接判 failed。最终 `usable_for_robot_scene_flow` 由 `trajectory_filter.output_policy` 决定，默认保留 static/moving/uncertain，过滤 jitter。

### 7.10 Segment Cache

每个 segment 处理结束后都会写 cache：

```text
output_root/dataset/episode_id/.segment_cache/<safe_view_name>/segment_000000.npz
```

cache 中保存：

```text
cache schema version
config fingerprint
SegmentJob
status / error_code / error_message
detections
sampling
groups
timings
```

如果 chunk 内某个阶段失败，会生成 `FAILED` 的 `SegmentResult` 并写 cache。这样 episode 合并时仍能产出完整 summary，失败 segment 可在 `failed_segments` 中定位。

### 7.11 Episode 合并

当一个 episode 的所有 segment 都有结果后，runner 调用：

```text
write_episode_outputs(output_root, episode, merged_results, atomic, cfg)
```

合并前会做连续性校验：

- `episode_end_frames[:-1] == episode_start_frames[1:]`
- `source_end_frames[:-1] == source_start_frames[1:]`
- 每个 segment 的 source/episode offset 一致。
- padding 区域使用 NaN 或默认枚举值。

写入策略：

1. 先在本地临时文件生成 NPZ，减少挂载盘直接写大文件失败风险。
2. 复制到输出目录的 `.tmp.npz`。
3. `atomic_write=true` 时用 `os.replace` 原子替换正式 NPZ。
4. summary JSON 同样先写临时文件再替换。
5. 如果配置允许，删除 `.segment_cache`。

## 8. 保存结果

### 8.1 Episode 输出目录

每个 episode/view 的结果目录：

```text
output_root/
└── dataset/
    └── episode_id/
        ├── <safe_view_name>_scene_tracks.npz
        └── <safe_view_name>_summary.json
```

`safe_view_name` 由 `view_key` 转换得到，当前实现会替换 `/`，点号保留。例如：

```text
observation.images.top_scene_tracks.npz
observation.images.top_summary.json
```

### 8.2 批处理级输出

输出根目录下还会写：

```text
processing_manifest.jsonl
performance_summary.json
```

`processing_manifest.jsonl` 会记录：

- batch planned 信息。
- 每个 segment 的处理状态。
- 每个 episode 合并完成后的输出路径。

`performance_summary.json` 会记录：

```text
episodes_processed
episodes_failed
segments_processed
segments_failed
invalid_manifest_items
resume_skipped_episodes
average_first_frame_decode_time
average_yolo_time
average_sampling_time
average_segment_decode_time
average_cotracker_time
average_filter_time
total_wall_time
segments_per_second
```

## 9. 数据可视化

批量主流程默认只保存结构化结果，不自动生成视频可视化。后期复现可视化时使用：

```text
scripts/visualize_scene_tracks_npz.py
```

该脚本读取 episode 级 `*_scene_tracks.npz`，从相邻的 `*_summary.json` 中找到 `source_video_path` 和 fps，然后按 NPZ 中保存的 segment 帧号重新解码源视频，把轨迹叠加到原视频帧上，输出 H.264 mp4 和第一帧预览图。

### 9.1 基本命令

示例：对一个 top 视角 episode 生成可视化。

```bash
cd /mnt/workspace/SceneFlowTracker

/root/miniconda3/envs/co-tracker/bin/python scripts/visualize_scene_tracks_npz.py \
  /mnt/data/chachaxu/dataset/abc_130k_v3_sceneflow/abc_130k_v3_train/abc_130k_v3_train__episode_000000/observation.images.top_scene_tracks.npz \
  --output-path /mnt/data/chachaxu/dataset/abc_130k_v3_sceneflow/abc_130k_v3_train/abc_130k_v3_train__episode_000000/observation.images.top_tracks_visualization.mp4
```

输出文件：

```text
observation.images.top_tracks_visualization.mp4
observation.images.top_tracks_visualization_preview.jpg
```

如果需要先做短片段 smoke test：

```bash
/root/miniconda3/envs/co-tracker/bin/python scripts/visualize_scene_tracks_npz.py \
  /path/to/observation.images.top_scene_tracks.npz \
  --start-segment 0 \
  --end-segment 5 \
  --output-path /tmp/top_segments_000_004_viz.mp4
```

### 9.2 可视化内容

默认绘制：

- `left`：红色，左机械臂 usable 点。
- `right`：蓝色，右机械臂 usable 点。
- `env`：绿色，环境 usable 点。
- 每个点会绘制当前 segment 内最近 `trail_frames` 帧的轨迹尾巴。
- 默认只画 `*_usable=True` 的点。

重要参数：

| 参数 | 说明 |
| --- | --- |
| `--groups left,right,env` | 选择要绘制的轨迹组。可以只画 `left,right`，也可以只画 `env`。 |
| `--track-source smooth/raw` | 选择绘制平滑轨迹还是 CoTracker 原始轨迹。默认 `smooth`。 |
| `--draw-non-usable` | 同时绘制 non-usable 点，non-usable 默认用灰色显示。 |
| `--trail-frames 30` | 每帧向前回看多少帧作为轨迹尾巴。 |
| `--max-robot-points 1000` | 每段最多绘制多少机械臂点，避免图像过密。 |
| `--max-env-points 120` | 每段最多绘制多少环境点，默认会对环境点下采样以保持可读性。 |
| `--start-segment` | 起始 segment index。 |
| `--end-segment` | 结束 segment index，左闭右开。 |
| `--source-video` | 手动覆盖源视频路径。默认从 summary 或 NPZ 读取。 |
| `--source-fps` | 手动覆盖源视频 fps。默认从 summary 或 NPZ 读取。 |
| `--output-fps` | 手动覆盖输出视频 fps。默认使用 NPZ 的 `fps`。 |
| `--tmp-dir /tmp` | mp4 编码临时目录。默认 `/tmp`。 |
| `--crf 20` | H.264 质量参数。数值越低质量越高、文件越大。 |
| `--preset veryfast` | x264 编码速度/压缩率预设。 |

由于当前轨迹是按 segment 独立预测的，可视化轨迹尾巴也只在当前 segment 内连续，不会跨 segment 连接身份。这一点和 NPZ 的数据语义一致。

### 9.3 只看机械臂或只看环境

只看机械臂：

```bash
/root/miniconda3/envs/co-tracker/bin/python scripts/visualize_scene_tracks_npz.py \
  /path/to/observation.images.top_scene_tracks.npz \
  --groups left,right \
  --output-path /tmp/top_robot_tracks.mp4
```

只看环境：

```bash
/root/miniconda3/envs/co-tracker/bin/python scripts/visualize_scene_tracks_npz.py \
  /path/to/observation.images.top_scene_tracks.npz \
  --groups env \
  --max-env-points 300 \
  --output-path /tmp/top_env_tracks.mp4
```

对比 raw 与 smooth：

```bash
/root/miniconda3/envs/co-tracker/bin/python scripts/visualize_scene_tracks_npz.py \
  /path/to/observation.images.top_scene_tracks.npz \
  --track-source raw \
  --output-path /tmp/top_raw_tracks.mp4

/root/miniconda3/envs/co-tracker/bin/python scripts/visualize_scene_tracks_npz.py \
  /path/to/observation.images.top_scene_tracks.npz \
  --track-source smooth \
  --output-path /tmp/top_smooth_tracks.mp4
```

### 9.4 复制到本地桌面

如果在 1016 实例上生成了可视化视频，可以从 Windows PowerShell 拉到桌面：

```powershell
scp -P 1016 root@39.101.70.188:/tmp/top_robot_tracks.mp4 C:\Users\34927\Desktop\top_robot_tracks.mp4
```

也可以复制 episode 输出目录中的可视化文件：

```powershell
scp -P 1016 root@39.101.70.188:/mnt/data/chachaxu/dataset/abc_130k_v3_sceneflow/abc_130k_v3_train/abc_130k_v3_train__episode_000000/observation.images.top_tracks_visualization.mp4 C:\Users\34927\Desktop\observation.images.top_tracks_visualization.mp4
```

### 9.5 MP4 打不开的处理

脚本默认会先在 `--tmp-dir` 指定的本地目录生成 mp4，再复制到 `--output-path`。这样做是为了避免在某些挂载目录中直接写 mp4 时，文件尾部索引没有正常写入，导致播放器报错或 `ffprobe` 提示：

```text
moov atom not found
```

如果仍然打不开，建议：

1. 将 `--output-path` 直接设到 `/tmp/xxx.mp4` 测试。
2. 用 `ffprobe /tmp/xxx.mp4` 确认容器有效。
3. 再用 `scp` 拉到本地桌面。

### 9.6 机械臂点很少时如何判断原因

可视化只负责把 NPZ 中已有轨迹画出来。如果视频中机械臂点很少，优先检查 summary：

```bash
/root/miniconda3/envs/co-tracker/bin/python scripts/inspect_scene_tracks_npz.py \
  /path/to/observation.images.top_scene_tracks.npz
```

或者直接看：

```text
segments[i].sampling.yolo_raw_detections
segments[i].sampling.left_bbox
segments[i].sampling.right_bbox
segments[i].left.num_points
segments[i].right.num_points
segments[i].environment.num_points
```

如果某段 `yolo_raw_detections=[]`，说明该段起始帧 YOLO 没检出机械臂；此时可视化里自然不会有对应机械臂轨迹。

## 10. NPZ Schema 1.2

schema 定义在：

```text
scene_flow_tracker/storage/schema.py
```

当前版本：

```text
SCHEMA_VERSION = "1.2"
FRAME_RANGE_SEMANTICS = "half_open"
TRACK_COORDINATE_ORDER = "xy"
TRACK_COORDINATE_SPACE = "original_image_pixels"
GROUPS = ("left", "right", "env")
```

### 10.1 元信息字段

| 字段 | 形状/类型 | 说明 |
| --- | --- | --- |
| `schema_version` | scalar string | schema 版本。 |
| `dataset` | scalar string | 数据集名。 |
| `episode_id` | scalar string | episode id。 |
| `episode_index` | scalar int64 | episode index。 |
| `task_index` | scalar int64 | task index；缺失时为 -1。 |
| `view_key` | scalar string | 当前处理视角。 |
| `fps` | scalar float32 | 有效处理 fps。 |
| `manifest_fps` | scalar float32 | manifest 原始 fps。 |
| `image_width` | scalar int32 | 图像宽度。 |
| `image_height` | scalar int32 | 图像高度。 |
| `source_video_path` | scalar string | 物理视频路径。 |
| `episode_source_start_frame` | scalar int64 | episode 在物理视频中的起始帧。 |
| `episode_source_end_frame` | scalar int64 | episode 在物理视频中的结束帧。 |
| `frame_range_semantics` | scalar string | 当前为 `half_open`。 |
| `track_coordinate_order` | scalar string | 当前为 `xy`。 |
| `track_coordinate_space` | scalar string | 当前为 `original_image_pixels`。 |
| `schema_metadata_json` | scalar string | 枚举值和类别名元数据。 |
| `summary_json` | scalar string | summary JSON 的完整字符串副本。 |

### 10.2 Segment 字段

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `segment_ids` | `[S]` | segment id。 |
| `episode_start_frames` | `[S]` | segment 在 episode 内的起始帧。 |
| `episode_end_frames` | `[S]` | segment 在 episode 内的结束帧。 |
| `source_start_frames` | `[S]` | segment 在物理视频中的起始帧。 |
| `source_end_frames` | `[S]` | segment 在物理视频中的结束帧。 |
| `segment_lengths` | `[S]` | 每个 segment 的真实帧数。 |
| `segment_status` | `[S]` | segment 状态枚举。 |
| `episode_frame_indices` | `[S,Tmax]` | 每个 segment 内每帧对应的 episode 帧号。padding 为 -1。 |
| `source_frame_indices` | `[S,Tmax]` | 每个 segment 内每帧对应的物理视频帧号。padding 为 -1。 |

### 10.3 YOLO 字段

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `yolo_class_ids` | `[2]` | 类别 id，0 为 left_arm，1 为 right_arm。 |
| `yolo_class_names` | `[2]` | 类别名。 |
| `yolo_frame_indices` | `[S,1]` | YOLO 检测帧在物理视频中的帧号。 |
| `yolo_episode_frame_indices` | `[S,1]` | YOLO 检测帧在 episode 内的帧号。 |
| `yolo_frame_valid` | `[S,1]` | 该 segment 是否执行了 YOLO 首帧检测。 |
| `yolo_bboxes` | `[S,1,2,4]` | left/right bbox，格式 `xyxy`。无效为 NaN。 |
| `yolo_bbox_confidence` | `[S,1,2]` | left/right bbox 置信度。无效为 NaN。 |
| `yolo_bbox_valid` | `[S,1,2]` | left/right bbox 是否有效。 |

### 10.4 轨迹组字段

轨迹分为三组：

```text
left
right
env
```

每组字段结构相同。以 `left` 为例：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `left_num_points` | `[S]` | 每个 segment 中 left 真实点数。 |
| `left_point_valid` | `[S,Nmax]` | padding 后每个点是否为真实点。 |
| `left_query_xy` | `[S,Nmax,2]` | query 点坐标。padding 为 NaN。 |
| `left_tracks_raw` | `[S,Tmax,Nmax,2]` | CoTracker 原始轨迹。padding 为 NaN。 |
| `left_tracks_smooth` | `[S,Tmax,Nmax,2]` | 平滑后轨迹。padding 为 NaN。 |
| `left_visibility` | `[S,Tmax,Nmax]` | CoTracker visibility。padding 为 `False`。 |
| `left_track_state` | `[S,Nmax]` | 轨迹状态枚举。 |
| `left_motion_state` | `[S,Nmax]` | 运动状态枚举。 |
| `left_usable` | `[S,Nmax]` | 是否作为最终可用轨迹点。 |
| `left_group_status` | `[S]` | 该组在每个 segment 的状态。 |
| `left_cotracker_confidence` | `[S,Tmax,Nmax]` | 可选，CoTracker confidence。 |
| `left_sampling_score` | `[S,Nmax]` | 可选，采样综合评分。 |
| `left_edge_strength` | `[S,Nmax]` | 可选，边缘强度。 |
| `left_trackability_score` | `[S,Nmax]` | 可选，可追踪性评分。 |
| `left_color_score` | `[S,Nmax]` | 可选，颜色先验评分。 |
| `left_topology_score` | `[S,Nmax]` | 可选，拓扑评分。 |
| `left_candidate_level` | `[S,Nmax]` | 可选，候选级别枚举。 |
| `left_visibility_ratio` | `[S,Nmax]` | 可选，轨迹可见比例。 |
| `left_net_displacement` | `[S,Nmax]` | 可选，首尾净位移。 |
| `left_path_length` | `[S,Nmax]` | 可选，路径总长度。 |
| `left_path_efficiency` | `[S,Nmax]` | 可选，路径效率。 |
| `left_jitter_rms` | `[S,Nmax]` | 可选，抖动 RMS。 |
| `left_jitter_residual_ratio` | `[S,Nmax]` | 可选，抖动残差比例。 |
| `left_turn_consistency` | `[S,Nmax]` | 可选，转向一致性。 |
| `left_turn_angle_mad` | `[S,Nmax]` | 可选，转角 MAD。 |
| `left_normalized_jerk` | `[S,Nmax]` | 可选，归一化 jerk。 |
| `left_direction_reversal_ratio` | `[S,Nmax]` | 可选，方向反转比例。 |

`right_*` 与 `env_*` 同理。环境组没有颜色和拓扑等机械臂专属采样特征，主要保存：

```text
env_sampling_score
env_edge_strength
env_trackability_score
```

### 10.5 Padding 规则

由于不同 segment 可能有不同点数，NPZ 使用定长 padding：

- `*_num_points[s]` 表示第 `s` 个 segment 的真实点数。
- `*_point_valid[s]` 标记真实点位置。
- `*_query_xy` 和 `*_tracks_*` 的 padding 使用 NaN。
- `*_visibility` padding 使用 `False`。
- `*_track_state`、`*_motion_state`、`*_group_status` padding 使用枚举默认值。
- 禁止 `dtype=object`，方便后续 NumPy / PyTorch 批量读取。

## 11. 状态枚举

### 11.1 SegmentStatus

```text
0 UNKNOWN_OR_PADDING
1 OK
2 PARTIAL
3 FAILED
```

### 11.2 GroupStatus

```text
0 UNKNOWN
1 OK
2 NO_DETECTION
3 NO_CANDIDATES
4 TRACK_FAILED
5 PROCESSING_FAILED
```

`NO_DETECTION` 常见于某个机械臂 slot 没有 YOLO bbox。它不表示整个 segment 失败，只表示该组没有机械臂检测结果。

### 11.3 TrackState

```text
0 INVALID_OR_PADDING
1 VALID
2 PARTIAL
3 FAILED
```

当前默认配置下，visibility 低不会直接让点进入 `FAILED`，除非其他过滤逻辑或处理异常触发。

### 11.4 MotionState

```text
0 UNKNOWN_OR_PADDING
1 STATIC
2 MOVING
3 JITTER
4 UNCERTAIN
```

默认输出策略保留：

```text
STATIC
MOVING
UNCERTAIN
PARTIAL
```

默认过滤：

```text
JITTER
```

## 12. Summary JSON

每个 episode 会保存一个可读 summary：

```text
<safe_view_name>_summary.json
```

顶层字段包括：

```text
schema_version
created_at
dataset
episode_id
episode_index
task_index
task
instruction
view_key
source_video_path
episode_source_start_frame
episode_source_end_frame
fps
manifest_fps
image_width
image_height
segment_count
segments_ok
segments_partial
segments_failed
failed_segment_ids
left
right
environment
schema
processing
segments
```

`left/right/environment` 是组级统计，例如：

```text
total_queries
usable
track_valid
track_partial
track_failed
static
moving
jitter
uncertain
```

`segments` 是每个 segment 的详细摘要，包含：

```text
segment_id
status
episode_start_frame
episode_end_frame
source_start_frame
source_end_frame
frame_count
error_code
error_message
sampling
left
right
environment
timings
```

其中 `sampling` 会保存该段的 YOLO bbox、raw detections、点数分配和采样统计。定位漏检时，优先看：

```text
segments[i].sampling.yolo_raw_detections
segments[i].sampling.left_bbox
segments[i].sampling.right_bbox
segments[i].left.num_points
segments[i].right.num_points
segments[i].environment.num_points
```

## 13. 常见现象与排查

### 13.1 机械臂点很少

先看 summary 中每段：

```text
yolo_raw_detections
left_bbox
right_bbox
left_count
right_count
environment_count
```

如果 `yolo_raw_detections=[]` 且 `left_bbox/right_bbox=None`，说明 YOLO 首帧没有检出机械臂。此时该段机械臂点数为 0，配额会转给环境点。

可能原因：

- `models.yolo.conf` 太高。
- YOLO 权重对当前视角泛化不好。
- 当前 segment 起始帧机械臂遮挡、过小或不完整。
- 只检测起始帧，而机械臂在后续帧才明显出现。

### 13.2 环境点很多

这是 query 分配的预期行为。为了保证每段 CoTracker 输入点数固定，缺失的机械臂配额会转给环境点。

### 13.3 可视化没有轨迹

批量主流程默认不保存可视化视频，只保存 NPZ 和 summary。需要额外脚本读取 `*_tracks_smooth` 或 `*_tracks_raw` 叠到视频上。注意 MP4 最好先写到实例本地磁盘如 `/tmp`，再复制到挂载目录；直接在某些挂载目录写 mp4 可能导致 `moov atom` 丢失，播放器打不开。

### 13.4 单卡显存没有吃满

优先调大：

```yaml
models.cotracker.segment_batch_size
pipeline.max_inflight_segments
```

如果单段点数增加很多，再关注：

```yaml
models.cotracker.point_chunk_size
sampling.query_allocation.total_query_points
```

`workers.*` 主要影响 CPU 解码、采样和过滤，不直接等价于 GPU 显存占用。

## 14. 当前框架边界

当前实现有几个明确边界：

- 每次运行只处理一个 `view_key`。
- YOLO 只检测每个 segment 的第一帧，不逐帧检测。
- segment 之间独立采样和追踪，当前没有跨 segment 轨迹身份连接。
- 输出是二维像素轨迹，不是三维场景流。
- `depth_enabled=false`，深度图相关流程未进入当前正式 pipeline。
- 批量主流程不自动生成可视化视频。

这些边界是当前框架的工程取舍，后续如果要提高机械臂召回，可优先考虑：降低 YOLO conf、增加每段内多帧检测、引入 VLM/SAM bbox 或 mask 辅助、或者做跨 segment 的检测结果传播。
