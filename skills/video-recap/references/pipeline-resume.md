# 断点续跑与局部重跑

Pipeline 会在 `work_dir` 下写 `workflow_state.json` 记录统一步骤状态、耗时、参数指纹和输入视频内容指纹；同时保留 `.step_*.done` 作为兼容旧工作目录和手动清理的标记。

组装阶段还会写 `assemble_meta.json`，记录是否压制字幕、压制字幕样式、是否强制重编码等渲染设置，以及 TTS meta / 输入视频的内容指纹。续跑时这些设置或内容指纹变化会自动触发重新组装，避免复用旧的 `output.mp4`。

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

## 内容指纹缓存

新工作目录会记录输入视频的快速内容指纹（文件大小 + 首尾采样 hash），不再只依赖路径或 mtime。相同视频复制到不同路径时，TTS/组装等尾部产物可以通过内容指纹判断是否仍可复用；视频内容变化时，`workflow_state.json` 会清空旧步骤状态并要求重新分析。

`tts_meta.json` 和 `assemble_meta.json` 也会记录相关输入文件的内容指纹。改 `narration.json`、`narration_mapped.json`、TTS 设置或组装设置后，续跑会自动判断尾部缓存是否过期。

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

## 压制字幕

```bash
python3 scripts/video_recap.py <video> --resume work_dir --burn-subtitles
```

CLI 会继续生成 `subtitles.srt`，并额外生成用于压制的 `subtitles.ass`。压制需要当前 `ffmpeg` 带 `subtitles`/libass 滤镜；缺少时会在组装前报错。若只切换 `--burn-subtitles` 或调整 `SUBTITLE_FONT_SIZE` / `SUBTITLE_MARGIN_V` 等字幕样式环境变量，通常不需要手动删除 `.step_assemble.done`；`assemble_meta.json` 会让组装阶段自动重跑。

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
