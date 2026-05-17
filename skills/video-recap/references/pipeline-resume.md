# 断点续跑与局部重跑

Pipeline 会在 `work_dir` 下用 `.step_*.done` 标记已完成阶段。

| 标记 | 阶段 |
|------|------|
| `.step_extract.done` | 帧提取 |
| `.step_detect.done` | 场景检测 |
| `.step_asr.done` | ASR |
| `.step_silence.done` | 静音检测 |
| `.step_vlm.done` | VLM 分析 |
| `.step_script.done` | Agent 写好的 `narration.json` 已验证；cut 模式还要求 `clip_plan.json` |
| `.step_edit.done` | cut 模式剪辑源视频与时间轴映射已生成 |
| `.step_tts.done` | TTS 合成 |
| `.step_assemble.done` | 视频组装 |

## 写好 narration.json 后继续

```bash
python3 scripts/video_recap.py <video> --resume work_dir
```

## 改解说词后重新配音

```bash
rm -rf work_dir/tts_segments/ work_dir/.step_tts.done \
  work_dir/.step_assemble.done work_dir/tts_meta.json
python3 scripts/video_recap.py <video> --resume work_dir
```

cut 模式下，如果只改了 `clip_plan.json` 或 `narration.json`，续跑会自动重建 `clip_plan_validated.json`、`edited_source.mp4`、`narration_mapped.json`。如果想强制重建，也可以删：

```bash
rm -f work_dir/.step_edit.done work_dir/clip_plan_validated.json \
  work_dir/narration_mapped.json work_dir/edited_source.mp4
```

## 换音色

```bash
rm -rf work_dir/tts_segments/ work_dir/.step_tts.done \
  work_dir/.step_assemble.done work_dir/tts_meta.json
python3 scripts/video_recap.py <video> --resume work_dir --voice zh-CN-YunxiNeural
```

## 重新做 VLM 分析

```bash
rm -f work_dir/.step_vlm.done work_dir/.step_script.done \
  work_dir/.step_tts.done work_dir/.step_assemble.done
rm -f work_dir/vlm_analysis.json work_dir/narration.json work_dir/tts_meta.json
rm -rf work_dir/tts_segments/
OPENAI_MODEL=新模型 python3 scripts/video_recap.py <video> --resume work_dir
```
