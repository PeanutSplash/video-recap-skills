import re
from pathlib import Path

from config import CONFIG
from common import log


# ── Agent narration preparation and validation helpers ────────────────


def _format_frame_facts(scene):
    """将帧动作描述格式化为可注入 agent brief 的文本。"""
    facts = scene.get("frame_facts", {})
    if not facts:
        return ""
    lines = []
    for ts in sorted(facts.keys(), key=lambda x: float(x)):
        actions = facts[ts]
        lines.append(f"    {ts}s: {'; '.join(actions)}")
    return "\n  帧动作:\n" + "\n".join(lines)


def _text_char_count(text):
    """计算文本的有效字数（去除标点和空白，这些不占 TTS 朗读时间）。"""
    return len(re.sub(r'[，。！？、；：…“”‘’《》〈〉\s"\'「」『』（）()【】\[\]—～·,.!?;:\\-]', '', text or ""))


def _truncate_at_sentence(text, max_chars):
    """在句子边界截断，不产生残句。max_chars 按有效字符计（不含标点空白）。"""
    if _text_char_count(text) <= max_chars:
        return text
    eff = 0
    cutoff = len(text)
    for i, ch in enumerate(text):
        eff += 1 if _text_char_count(ch) else 0
        if eff > max_chars:
            cutoff = i + 1
            break
    for sep in ['。', '！', '？', '!', '?']:
        idx = text[:cutoff].rfind(sep)
        if idx > 0:
            return text[:idx + 1]
    for sep in ['，', '、', '；', ',']:
        idx = text[:cutoff].rfind(sep)
        if idx > 3:
            return text[:idx] + '。'
    return ""


def _char_bigrams(text):
    return {text[i:i + 2] for i in range(len(text) - 1) if text[i:i + 2].strip()}


def _post_dedup_narration(narration):
    """去除相邻相似解说段（bigram Jaccard >40% 则合并）。"""
    if len(narration) < 2:
        return narration
    result = [narration[0]]
    for seg in narration[1:]:
        prev = result[-1]
        if not prev.get("narration", "").strip() or not seg.get("narration", "").strip():
            result.append(seg)
            continue
        set_a, set_b = _char_bigrams(prev["narration"]), _char_bigrams(seg["narration"])
        if not set_a or not set_b:
            result.append(seg)
            continue
        overlap = len(set_a & set_b) / min(len(set_a), len(set_b))
        if overlap > 0.4:
            if len(seg["narration"]) > len(prev["narration"]):
                prev["narration"] = seg["narration"]
            prev["end"] = seg["end"]
            prev["pause_after_ms"] = seg.get("pause_after_ms", prev.get("pause_after_ms", 600))
            log(f"  去重合并: {prev['start']:.0f}-{prev['end']:.0f}s")
        else:
            result.append(seg)
    removed = len(narration) - len(result)
    if removed:
        log(f"  去重: {len(narration)} → {len(result)} 段 (合并 {removed} 段)")
    return result


def _scene_available_seconds(start, end, pause_after_ms=None):
    pause = (CONFIG.get("breath_ms", 600) if pause_after_ms is None else pause_after_ms) / 1000
    return max(0.0, float(end) - float(start) - pause)


def _recommended_char_budget(start, end, pause_after_ms=None):
    effective_rate = CONFIG["speech_rate"] * CONFIG["speech_safety_margin"]
    available = _scene_available_seconds(start, end, pause_after_ms)
    return max(0, int(available * effective_rate))


def _find_scene_for_midpoint(scenes_analysis, start, end):
    mid = (float(start) + float(end)) / 2
    for scene in scenes_analysis:
        if scene["start"] <= mid <= scene["end"]:
            return scene
    return None


def _normalise_narration_segment(seg, scenes_analysis=None):
    if not isinstance(seg, dict):
        return None
    try:
        start = float(seg.get("start"))
        end = float(seg.get("end"))
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    text = str(seg.get("narration", "")).strip()
    if not text:
        return None
    pause = seg.get("pause_after_ms", CONFIG.get("breath_ms", 600))
    try:
        pause = int(pause)
    except (TypeError, ValueError):
        pause = CONFIG.get("breath_ms", 600)
    item = {
        "start": round(start, 2),
        "end": round(end, 2),
        "narration": text,
        "pause_after_ms": pause,
        "overlaps_speech": bool(seg.get("overlaps_speech", False)),
    }
    for optional_key in ("source_start", "source_end", "source_clip_id"):
        if optional_key in seg:
            item[optional_key] = seg[optional_key]
    if scenes_analysis:
        parent = _find_scene_for_midpoint(scenes_analysis, item["start"], item["end"])
        if parent:
            item["start"] = round(max(parent["start"], item["start"]), 2)
            item["end"] = round(min(parent["end"], item["end"]), 2)
            if item["end"] <= item["start"]:
                return None
    return item


def _validate_narration_budget(narration, scenes_analysis):
    """Validate agent-written narration against timing budgets; trim impossible text safely."""
    if not isinstance(narration, list):
        raise ValueError("narration.json 必须是 JSON 数组")

    cleaned = []
    for raw in narration:
        item = _normalise_narration_segment(raw, scenes_analysis)
        if not item:
            continue
        max_chars = _recommended_char_budget(item["start"], item["end"], item.get("pause_after_ms"))
        if max_chars < 5:
            log(f"  丢弃过短解说段 {item['start']:.1f}-{item['end']:.1f}s")
            continue
        if _text_char_count(item["narration"]) > max_chars * 1.25:
            truncated = _truncate_at_sentence(item["narration"], max_chars)
            if truncated and _text_char_count(truncated) >= 5:
                log(f"  解说超预算，已截短: {item['start']:.1f}-{item['end']:.1f}s")
                item["narration"] = truncated
            else:
                log(f"  解说超预算且无法安全截断，已丢弃: {item['start']:.1f}-{item['end']:.1f}s")
                continue
        item["narration"] = _clean_narration_punctuation(item["narration"])
        if item["narration"].strip()[-1] in "，：、；,—…":
            item["narration"] += "。"
        cleaned.append(item)

    cleaned.sort(key=lambda n: n["start"])
    deduped = []
    for item in cleaned:
        if deduped and item["start"] < deduped[-1]["end"]:
            prev = deduped[-1]
            log(
                f"  解说时间重叠: {item['start']:.1f}-{item['end']:.1f}s vs "
                f"{prev['start']:.1f}-{prev['end']:.1f}s"
            )
            if _text_char_count(item["narration"]) > _text_char_count(prev["narration"]):
                deduped[-1] = item
        else:
            deduped.append(item)
    return _post_dedup_narration(deduped)


def _clean_narration_punctuation(text):
    text = re.sub(r'\s+', ' ', text or '').strip()
    text = re.sub(r'[，：、；,]["\']?[。！？]', '。', text)
    text = re.sub(r'["\']。$', '。', text)
    return text


def _align_narration_to_quiet(narration, scenes_analysis, silence_periods):
    """将解说段移到同场景内的安静窗口，标记是否与语音重叠。"""
    if not silence_periods:
        for n in narration:
            n["overlaps_speech"] = False
        return _validate_narration_budget(narration, scenes_analysis)

    quiet_windows = [qp for qp in silence_periods if not qp.get("has_speech", False)]

    for n in narration:
        seg_start = n["start"]
        seg_end = n["end"]
        seg_dur = seg_end - seg_start
        best_window = None
        best_overlap = 0

        for qw in quiet_windows:
            overlap_start = max(seg_start, qw["start"])
            overlap_end = min(seg_end, qw["end"])
            overlap = overlap_end - overlap_start
            if overlap > best_overlap:
                best_overlap = overlap
                best_window = qw

        if best_window and best_overlap > 0:
            new_start = max(best_window["start"], seg_start - (seg_dur * 0.5))
            new_start = min(new_start, best_window["end"] - seg_dur)
            parent_scene = _find_scene_for_midpoint(scenes_analysis, seg_start, seg_end)
            if parent_scene:
                new_start = max(parent_scene["start"], new_start)
                new_start = min(new_start, parent_scene["end"] - seg_dur)
            new_start = max(0.0, new_start)
            new_end = round(new_start + seg_dur, 2)
            new_start = round(new_start, 2)
            if new_end > new_start:
                n["start"] = new_start
                n["end"] = new_end
                n["overlaps_speech"] = False
            else:
                n["overlaps_speech"] = True
        else:
            n["overlaps_speech"] = True

    narration.sort(key=lambda x: x["start"])
    for i in range(1, len(narration)):
        prev = narration[i - 1]
        curr = narration[i]
        min_gap = 0.3
        min_start = prev["end"] + min_gap
        if curr["start"] < min_start:
            seg_dur = curr["end"] - curr["start"]
            curr["start"] = round(min_start, 2)
            curr["end"] = round(min_start + seg_dur, 2)
            parent_scene = _find_scene_for_midpoint(scenes_analysis, curr["start"], curr["end"])
            if parent_scene and curr["end"] > parent_scene["end"]:
                curr["end"] = round(parent_scene["end"], 2)
            if curr["end"] - curr["start"] < 1.5:
                curr["narration"] = ""

    return _validate_narration_budget(narration, scenes_analysis)


def _scene_asr_lines(asr_result, scene):
    lines = []
    for seg in asr_result or []:
        try:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", start))
        except (TypeError, ValueError):
            continue
        if scene["start"] < end and scene["end"] > start:
            text = str(seg.get("text", "")).strip()
            if text:
                lines.append(f"    [{start:.1f}-{end:.1f}] {text}")
    return lines


def _quiet_windows_for_scene(silence_periods, scene):
    windows = []
    for qp in silence_periods or []:
        if qp.get("has_speech", False):
            continue
        if qp["start"] < scene["end"] and qp["end"] > scene["start"]:
            start = max(qp["start"], scene["start"])
            end = min(qp["end"], scene["end"])
            if end > start:
                windows.append((start, end))
    return windows


def build_agent_brief(scenes_analysis, asr_result, silence_periods, video_duration, work_dir, style="纪录片"):
    """Write a compact brief that tells the agent exactly how to author recap artifacts."""
    effective_rate = CONFIG["speech_rate"] * CONFIG["speech_safety_margin"]
    breath_sec = CONFIG.get("breath_ms", 600) / 1000
    edit_mode = CONFIG.get("edit_mode", "full")
    target_duration = CONFIG.get("target_duration") or "(not set)"
    lines = [
        "# Agent Narration Brief",
        "",
        "Write the required JSON artifact(s) manually from the analysis files in this work directory.",
        "The CLI will not generate final narration text; it will only validate timing, run TTS, and assemble the video.",
        "",
        f"- Style: {style}",
        f"- Edit mode: {edit_mode}",
        f"- Source video duration: {video_duration:.1f}s",
        f"- Target duration (cut mode): {target_duration}",
        f"- Effective speech budget: {effective_rate:.2f} Chinese chars/sec after {breath_sec:.1f}s pause allowance",
        f"- Context: {CONFIG.get('context_info') or '(none)'}",
        "",
    ]

    if edit_mode == "cut":
        lines.extend([
            "## Required files for cut mode",
            "",
            "First write `clip_plan.json` to choose source footage, then write `narration.json` using ORIGINAL source timestamps inside those clips.",
            "The CLI maps source timestamps to the edited timeline after concatenating clips.",
            "",
            "### clip_plan.json shape",
            "",
            "```json",
            "{",
            "  \"target_duration\": \"10m\",",
            "  \"clips\": [",
            "    {\"start\": 12.0, \"end\": 38.0, \"reason\": \"关键冲突开端\"}",
            "  ]",
            "}",
            "```",
            "",
            "### narration.json shape (source timestamps)",
            "",
            "```json",
            "[",
            "  {\"start\": 14.0, \"end\": 21.0, \"narration\": \"解说文本。\", \"pause_after_ms\": 600, \"overlaps_speech\": false}",
            "]",
            "```",
        ])
    else:
        lines.extend([
            "## Required JSON shape",
            "",
            "```json",
            "[",
            "  {\"start\": 5.0, \"end\": 12.0, \"narration\": \"解说文本。\", \"pause_after_ms\": 600, \"overlaps_speech\": false}",
            "]",
            "```",
        ])

    lines.extend([
        "",
        "## Writing rules",
        "",
        "1. Do not describe what the viewer can already see; explain intent, stakes, subtext, and story logic.",
        "2. Prefer quiet windows. Use `overlaps_speech=true` only when narration intentionally overlaps original dialogue.",
        "3. Keep each line under its max character budget; shorter is safer for edge-tts.",
        "4. Preserve original audio moments that carry important dialogue or atmosphere.",
        "5. In cut mode, select clips for plot causality, key dialogue, reveals, and emotional turns; avoid filler and repeated shots.",
        "6. After writing, run: `python3 skills/video-recap/scripts/video_recap.py <video> --resume <work_dir>`.",
        "",
        "## Scene timing guide",
        "",
    ])

    for scene in scenes_analysis:
        duration = scene["end"] - scene["start"]
        max_chars = max(5, int(max(1.0, duration - breath_sec) * effective_rate))
        quiets = _quiet_windows_for_scene(silence_periods, scene)
        quiet_text = ", ".join(f"{s:.1f}-{e:.1f}s" for s, e in quiets) or "none"
        lines.extend([
            f"### Scene {scene['scene_id'] + 1}: {scene['start']:.1f}-{scene['end']:.1f}s",
            f"- Duration: {duration:.1f}s; max budget if fully narrated: {max_chars} chars",
            f"- Quiet windows: {quiet_text}",
            f"- Description: {scene.get('description', '')}",
        ])
        if scene.get("depth_analysis"):
            lines.append(f"- Deeper analysis: {scene['depth_analysis']}")
        facts = _format_frame_facts(scene)
        if facts:
            lines.append(facts.rstrip())
        asr_lines = _scene_asr_lines(asr_result, scene)
        if asr_lines:
            lines.append("- ASR overlap:")
            lines.extend(asr_lines[:8])
        lines.append("")

    brief_path = Path(work_dir) / "agent_narration_brief.md"
    brief_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"已写入 Agent 解说写作 brief: {brief_path}")
    return brief_path
