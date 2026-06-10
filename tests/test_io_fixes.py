"""Regression tests for asr/tts/extract/doctor IO-robustness fixes (BUG 11 batch)."""
import json
import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills' / 'video-recap' / 'scripts'))

import asr
import doctor
import extract
import tts
from tts import _tts_edge


# ── asr.py ────────────────────────────────────────────────────────────

def test_segment_cut_failure_yields_empty_text_not_stale_transcription(monkeypatch, tmp_path):
    """切分失败的段应返回空文本，而不是对磁盘陈旧音频转录。"""
    segments_dir = tmp_path / "audio_segments"
    segments_dir.mkdir()
    audio_wav = tmp_path / "audio.wav"
    audio_wav.write_bytes(b"")

    def fake_run_cmd(cmd, **kwargs):
        # 第一段切分成功，第二段切分失败
        seg_target = cmd[-1] if isinstance(cmd, list) else ""
        if "seg_000.wav" in str(seg_target):
            return CompletedProcess(cmd, 0, stdout="", stderr="")
        return CompletedProcess(cmd, 1, stdout="", stderr="cut failed")

    def fake_run_asr(wav_path):
        # 如果切分失败仍调用 ASR，会返回这段污染文本
        return "STALE-GARBAGE"

    monkeypatch.setattr("asr.run_cmd", fake_run_cmd)
    monkeypatch.setattr("asr._run_asr", fake_run_asr)

    results = asr._segment_and_transcribe(audio_wav, segments_dir, total_duration=60.0, segment_length=30)

    assert len(results) == 2
    assert results[0]["text"] == "STALE-GARBAGE"   # 成功段照常转录
    assert results[1]["text"] == ""                # 失败段不应有陈旧文本


def test_zero_duration_does_not_fabricate_180s_timestamps(monkeypatch, tmp_path):
    """get_video_duration 返回 0 时应警告并返回空 ASR，而不是伪造 0-180s 时间戳。"""
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"")

    def fake_run_cmd(cmd, **kwargs):
        # 音频提取这一步成功，其余不应被调用
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("asr.run_cmd", fake_run_cmd)
    monkeypatch.setattr("asr.get_video_duration", lambda path: 0.0)

    def boom(*args, **kwargs):
        raise AssertionError("时长为 0 时不应进行任何转录")

    monkeypatch.setattr("asr._run_asr", boom)
    monkeypatch.setattr("asr._segment_and_transcribe", boom)

    result = asr.transcribe_audio(video_path, tmp_path)

    assert result == []
    saved = json.loads((tmp_path / "asr_result.json").read_text())
    assert saved == []
    # 绝不应出现伪造的 0-180s 时间戳
    assert not any(s.get("end") == 180.0 for s in saved)


# ── tts.py ────────────────────────────────────────────────────────────

def test_edge_mp3_to_wav_failure_surfaces_error_and_keeps_mp3(monkeypatch, tmp_path):
    """mp3->wav 转换非零退出应抛错，且不应删除 mp3 源。"""
    output_wav = tmp_path / "narr_000.wav"
    mp3_path = tmp_path / "narr_000.mp3"

    monkeypatch.setitem(tts.CONFIG, "edge_tts_voice", "zh-CN-YunxiNeural")
    monkeypatch.setitem(tts.CONFIG, "tts_timeout", 90)

    def fake_run_cmd(cmd, **kwargs):
        if cmd[0] == "edge-tts":
            # 模拟 edge-tts 成功写出 mp3
            Path(mp3_path).write_bytes(b"fake-mp3")
            return CompletedProcess(cmd, 0, stdout="", stderr="")
        # ffmpeg 转换失败
        return CompletedProcess(cmd, 1, stdout="", stderr="conversion broke")

    removed = []
    monkeypatch.setattr("tts.run_cmd", fake_run_cmd)
    monkeypatch.setattr("tts.os.remove", lambda p: removed.append(p))

    with pytest.raises(RuntimeError, match="mp3 转 WAV 失败"):
        _tts_edge("测试文本", output_wav)

    # mp3 源不能在转换成功前被删除
    assert removed == []
    assert mp3_path.exists()


# ── extract.py ────────────────────────────────────────────────────────

def test_extract_frames_returns_only_current_run_frames(monkeypatch, tmp_path):
    """复用 work_dir 时，上一次更高编号的陈旧帧不应泄漏进结果。"""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # 上一次高 fps 运行残留的陈旧帧
    for n in range(1, 6):
        (frames_dir / f"frame_{n:05d}.jpg").write_bytes(b"stale")

    monkeypatch.setitem(extract.CONFIG, "fps", 1)

    def fake_run_cmd(cmd, **kwargs):
        # 本次只产出 2 帧
        (frames_dir / "frame_00001.jpg").write_bytes(b"new")
        (frames_dir / "frame_00002.jpg").write_bytes(b"new")
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("extract.run_cmd", fake_run_cmd)

    frames = extract.extract_frames(tmp_path / "video.mp4", tmp_path, fps=1)

    assert len(frames) == 2
    assert [f.name for f in frames] == ["frame_00001.jpg", "frame_00002.jpg"]


# ── doctor.py ─────────────────────────────────────────────────────────

def test_doctor_smoke_timeout_is_handled_not_raised(monkeypatch):
    """edge-tts 网络挂起（TimeoutExpired）应被吞掉转为 skipped，而不是抛出 traceback。"""
    monkeypatch.setattr("doctor._command_path", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd, *, timeout=20):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr("doctor._run", fake_run)

    result = doctor._check_tts_smoke("zh-CN-YunxiNeural")

    assert result.get("skipped") is True
    assert result.get("ok") is False


def test_doctor_skipped_smoke_not_marked_as_failure(monkeypatch):
    """跳过的冒烟测试应记为 warning，doctor 不应因此判为 FAILED。"""
    # 系统工具齐备，TTS 可用，仅冒烟测试被跳过
    monkeypatch.setattr("doctor._ffmpeg_filters", lambda: {"subtitles", "ass"})
    monkeypatch.setattr(
        "doctor._command_path",
        lambda name: f"/usr/bin/{name}" if name in ("ffmpeg", "ffprobe") else None,
    )
    monkeypatch.setitem(doctor.CONFIG, "mimo_tts_api_key", "tp-test-key")
    monkeypatch.setattr(
        "doctor._check_tts_smoke",
        lambda voice: {"ok": False, "skipped": True, "reason": "edge-tts not found"},
    )

    report = doctor.build_report(tts_smoke=True)

    assert report["ok"] is True
    assert "edge-tts smoke test failed" not in report["failures"]
    assert any("smoke test skipped" in w for w in report["warnings"])
