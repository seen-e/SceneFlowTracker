# SceneFlowTracker

SceneFlowTracker 是面向 LeRobot / ABC-130K 主视角或腕部视角视频的批量二维稀疏轨迹处理框架。

当前版本已经重构为分阶段 pipeline：

```text
manifest -> EpisodeJob -> SegmentJob
  -> 首帧解码
  -> YOLO batch 检测机械臂 bbox
  -> 固定总数 query 采样
  -> 完整 segment 解码
  -> CoTracker 多 segment batch 追踪
  -> 轨迹过滤
  -> segment cache
  -> episode NPZ + summary.json
```

正式 pipeline 不再导入外部业务采样/过滤代码；只依赖官方 CoTracker、Ultralytics、OpenCV、NumPy、PyTorch 和本仓库内部算法模块。

## 指定视角

在 `configs/config.yaml` 中设置：

```yaml
input:
  view_key: observation.images.top
```

框架只会读取：

```python
episode["video_segments"]["observation.images.top"]
```

如果要处理腕部视角，可改为：

```yaml
input:
  view_key: observation.images.left_wrist
```

或：

```yaml
input:
  view_key: observation.images.right_wrist
```

## 运行

推荐入口：

```bash
cd /mnt/workspace/SceneFlowTracker
bash example/run.sh
```

调试单个 episode：

```bash
bash example/run.sh --episode-index 0 --no-resume --debug
```

调试单个 segment：

```bash
bash example/run.sh --episode-index 0 --segment-id 0 --no-resume --debug
```

等价 Python 入口：

```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

性能测试：

```bash
python scripts/benchmark_pipeline.py --config configs/config.yaml --max-episodes 1 --no-resume
```

停止任务：

```bash
bash example/kill.sh
```

## 关键配置

主配置只保留常用运行参数：

- `input.manifest_path`：ABC/LeRobot manifest。
- `input.view_key`：本次处理的单个视角。
- `video.segment_frames`：每个 segment 的帧数。
- `workers.*`：CPU 解码、采样、过滤并发数。
- `models.yolo.batch_size`：YOLO 首帧 batch size。
- `models.cotracker.segment_batch_size`：CoTracker 的 segment 维度 batch size。
- `models.cotracker.point_chunk_size`：CoTracker 的 query 点维度 chunk size。
- `sampling.query_allocation.total_query_points`：每段固定 query 总数。
- `cache.*`：segment 级缓存与断点续跑策略。
- `output.output_root`：输出根目录。

采样阈值和轨迹过滤阈值放在：

```text
configs/sampling.yaml
configs/trajectory_filter.yaml
```

## 输出

每个 episode 输出到：

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

segment 级缓存默认在：

```text
output_root/dataset/episode_id/.segment_cache/
```

episode 合并成功后会按配置自动删除。

## NPZ 格式

当前 schema 版本为 `1.2`。核心字段包括：

- segment 索引：`segment_ids`、`episode_start_frames`、`source_start_frames`、`segment_lengths`
- YOLO 检测：`yolo_bboxes`、`yolo_bbox_confidence`、`yolo_bbox_valid`
- 左臂轨迹：`left_query_xy`、`left_tracks_raw`、`left_tracks_smooth`、`left_usable`
- 右臂轨迹：`right_query_xy`、`right_tracks_raw`、`right_tracks_smooth`、`right_usable`
- 环境轨迹：`env_query_xy`、`env_tracks_raw`、`env_tracks_smooth`、`env_usable`

检查输出：

```bash
python scripts/inspect_scene_tracks_npz.py /path/to/observation.images.top_scene_tracks.npz
```

校验输出：

```bash
python scripts/validate_scene_tracks_npz.py /path/to/observation.images.top_scene_tracks.npz
```

更详细的数据流说明见：

```text
docs/data_processing_pipeline.md
```
