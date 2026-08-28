# SceneFlowTracker

SceneFlowTracker 是面向 LeRobot / ABC-130K 视频数据的批量二维稀疏场景轨迹处理框架。

它负责把 `episodes.json` 中的 episode 记录转换成 segment 任务，通过独立的解码 worker 和模型 worker 完成：

- 按 `video_segments` 精确读取物理视频片段；
- 按左闭右开 `[start_frame, end_frame)` 规划 segment；
- 使用 YOLO 检测左右机械臂 bbox；
- 复用已有机械臂采样、环境采样、CoTracker 预测和轨迹过滤模块；
- 输出每个 episode 独立的 NPZ、`summary.json`、批处理状态和性能统计。

## 指定视角

当前框架只处理 YAML 中指定的一个视角，例如：

```yaml
input:
  view_key: observation.images.top
```

框架会只读取：

```python
episode["video_segments"]["observation.images.top"]
```

如果要处理腕部视角，只需要改配置：

```yaml
input:
  view_key: observation.images.left_wrist
```

或：

```yaml
input:
  view_key: observation.images.right_wrist
```

注意：`video_segments[view_key]` 是物理视频路径和帧范围的权威来源；顶层 `video_path` 只作为元数据参考。

## 运行

批量运行：

```bash
python scripts/run_batch_scene_tracking.py \
  --config configs/config.yaml
```

调试单个 episode：

```bash
python scripts/run_batch_scene_tracking.py \
  --config configs/config.yaml \
  --episode-index 0 \
  --debug
```

调试单个 segment：

```bash
python scripts/run_batch_scene_tracking.py \
  --config configs/config.yaml \
  --episode-index 0 \
  --segment-id 0 \
  --debug
```

## 输出

每个 episode 的输出目录形如：

```text
output_root/
└── dataset/
    └── episode_id/
        ├── observation.images.top_scene_tracks.npz
        └── observation.images.top_summary.json
```

批处理级别还会输出：

- `processing_manifest.jsonl`
- `performance_summary.json`

## 断点续跑

`batch.resume: true` 时，框架会先扫描一次 `output_root` 下已经存在的 `*_scene_tracks.npz` / `*_summary.json`，构建完成 episode×view 集合，然后用 O(1) 查询跳过已完成结果。这样不会对 manifest 中每个 episode 逐个做文件探测，适合 ABC-130K 这种大规模批处理。

## NPZ 格式

当前 writer 只写 `schema_version = "1.2"`。存储单位是 `episode × view`，所有帧范围统一为左闭右开 `[start_frame, end_frame)`。

核心索引字段：

- `segment_ids`
- `episode_start_frames` / `episode_end_frames`
- `source_start_frames` / `source_end_frames`
- `episode_frame_indices [S,Tmax]`
- `source_frame_indices [S,Tmax]`

三路点组会物理分开保存：

- 左臂：`left_query_xy [S,NLmax,2]`，`left_tracks_raw [S,Tmax,NLmax,2]`
- 右臂：`right_query_xy [S,NRmax,2]`，`right_tracks_raw [S,Tmax,NRmax,2]`
- 环境：`env_query_xy [S,NEmax,2]`，`env_tracks_raw [S,Tmax,NEmax,2]`

每组同时保存 `tracks_smooth`、`visibility`、`track_state`、`motion_state`、`usable` 和 `group_status`。变长数据由 padding、`*_num_points` 和 `*_point_valid` 表示，禁止 `dtype=object`。

状态枚举集中定义在 `scene_flow_tracker/storage/schema.py`：

- `TrackState`: `0=INVALID_OR_PADDING, 1=VALID, 2=PARTIAL, 3=FAILED`
- `MotionState`: `0=UNKNOWN_OR_PADDING, 1=STATIC, 2=MOVING, 3=JITTER, 4=UNCERTAIN`
- `SegmentStatus`: `0=UNKNOWN_OR_PADDING, 1=OK, 2=PARTIAL, 3=FAILED`
- `GroupStatus`: `0=UNKNOWN, 1=OK, 2=NO_DETECTION, 3=NO_CANDIDATES, 4=TRACK_FAILED, 5=PROCESSING_FAILED`

检查一个 NPZ：

```bash
python scripts/inspect_scene_tracks_npz.py /path/to/observation.images.top_scene_tracks.npz
```

校验一个 NPZ：

```bash
python scripts/validate_scene_tracks_npz.py /path/to/observation.images.top_scene_tracks.npz
```

代码中读取推荐使用 `EpisodeSceneTrackReader`，它会自动根据 `segment_lengths`、`*_num_points` 和 `*_point_valid` 去掉 padding，返回自然 shape。

更完整的实现说明见 `AUDIT_REPORT.md`。
