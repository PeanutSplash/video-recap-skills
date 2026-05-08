---
name: video-recap
description: >
  视频自动解说 skill。输入视频，输出带中文旁白的解说视频。
  使用场景：(1) 为视频生成中文解说旁白 (2) 视频摘要或解说 (3) 制作 recap 视频。
  触发词: video-recap, 视频解说, 视频旁白, 视频recap, 生成解说, recap
---

**解说词必须由 agent（你）亲自撰写，不要调用 LLM API 来生成解说词。**

## 流程

### 1. 运行前置 pipeline

```bash
OPENAI_API_KEY=xxx OPENAI_API_URL=xxx \
  python3 ~/.claude/skills/video-recap/scripts/video_recap.py <video> \
  --tts edge-tts --context "背景信息"
```

Pipeline 自动完成：帧提取 → 场景检测 → ASR → 静音检测 → VLM 分析。
到"解说脚本生成"步骤时暂停，等待 agent 写解说词。

### 2. Agent 写解说词

1. 读取工作目录中的 `vlm_analysis.json`、`asr_result.json`、`silence_periods.json`
2. 根据安静窗口识别解说区（合并相邻窗口，gap < 3s）
3. **亲自撰写解说词**——基于 VLM 深度分析和 ASR 对白，写有洞察力的故事性解说
4. 写入 `work_dir/narration.json`：
```json
[{"start": 秒数, "end": 秒数, "narration": "解说文本", "pause_after_ms": 600, "overlaps_speech": false}]
```
5. 标记完成 + 清理缓存 + 重新跑 TTS：
```bash
touch work_dir/.step_script.done
rm -rf work_dir/tts_segments/ work_dir/.step_tts.done work_dir/.step_assemble.done work_dir/tts_meta.json
python3 ~/.claude/skills/video-recap/scripts/video_recap.py <video> --resume work_dir
```

**每次改完 narration.json 后必须删 `tts_segments/`**，否则会复用旧音频。

## 解说词写作要求

- **禁止看图说话**：观众看得见画面，不要描述动作和表情
- **讲故事**：揭示角色意图、潜台词、关系动态
- **大段解说 + 原声交替**：解说区集中在安静窗口，原声区保留原始对白
- **字数预算**：每段字数 ≤ (end - start - 0.6) × 3，超限会被截断
- 如果提供了 `--context`，使用角色名（如 Big、凯莉）

## Pipeline 恢复

Pipeline 用 `.step_<name>.done` 标记文件控制跳过：

| 删哪些文件 | 重跑什么 |
|---|---|
| `.step_tts.done` + `tts_meta.json` | 重跑 TTS |
| `.step_tts.done` + `tts_meta.json` + `tts_segments/` | 强制重新合成所有 TTS |
| `.step_assemble.done` | 重跑视频组装 |

**快速 resume**（只改了某段）：删 `tts_segments/narr_00N.wav` + `.step_tts.done` + `tts_meta.json`，然后 `--resume`。

## 参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `video` | 输入视频文件路径 | (必填) |
| `--tts` | TTS 引擎: auto/indextts2/edge-tts/say | auto |
| `--style` | 解说风格: 短剧/电视剧/电影/纪录片/科普视频 | 纪录片 |
| `--context` | 额外上下文（节目名、角色名等） | "" |
| `--scene-threshold` | 场景检测阈值 0.0-1.0 | 0.1 |
| `--resume` | 从已有工作目录继续 | - |
| `--burn-subtitles` | 烧录字幕到视频（需重编码） | false |
| `--output, -o` | 输出目录 | 视频所在目录/output |
| `--step` | 仅执行: extract/detect/asr/analyze/script/tts/assemble | 全部 |

## 输出

- `recap_<视频名>.mp4` — 最终视频
- `subtitles.srt` — SRT 字幕
- 工作目录保留中间文件：`vlm_analysis.json`、`silence_periods.json`、`narration.json` 等

## 参考

- 解说词详细规则和风格模板见 [references/prompt-templates.md](references/prompt-templates.md)
- 用户要求调整 CONFIG 参数、Zone/Ducking 行为时，参考 [references/internal-config.md](references/internal-config.md)
