"""Regression tests for resume-settings persistence/restore (bugs 1, 3, 8)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills' / 'video-recap' / 'scripts'))

from config import CONFIG
from pipeline import (
    _asr_step_params,
    _detect_step_params,
    _extract_step_params,
    _has_runtime_override,
    _load_run_settings,
    _persist_run_settings,
    _resume_command,
)


# ── BUG 1: edit-mode / target-duration / clip-padding survive resume ──────

def test_has_runtime_override_true_after_cli_source(monkeypatch):
    # Without a *_source marker the override check must be False (config default).
    monkeypatch.setitem(CONFIG, "edit_mode_source", "default")
    assert _has_runtime_override("edit_mode") is False
    # An explicit CLI value must win over persisted run settings.
    monkeypatch.setitem(CONFIG, "edit_mode_source", "cli")
    assert _has_runtime_override("edit_mode") is True
    monkeypatch.setitem(CONFIG, "target_duration_source", "env")
    assert _has_runtime_override("target_duration") is True


def test_persist_restore_edit_mode_target_clip_padding(monkeypatch, tmp_path):
    # Persist a cut-mode run, then simulate a fresh resume process with default
    # config and no *_source markers: persisted values must be restored.
    monkeypatch.setitem(CONFIG, "edit_mode", "cut")
    monkeypatch.setitem(CONFIG, "edit_mode_source", "cli")
    monkeypatch.setitem(CONFIG, "target_duration", "8m")
    monkeypatch.setitem(CONFIG, "target_duration_source", "cli")
    monkeypatch.setitem(CONFIG, "clip_padding", 0.75)
    monkeypatch.setitem(CONFIG, "clip_padding_source", "cli")
    _persist_run_settings(tmp_path)

    # Resume process: defaults + no override markers.
    monkeypatch.setitem(CONFIG, "edit_mode", "full")
    monkeypatch.setitem(CONFIG, "edit_mode_source", "default")
    monkeypatch.setitem(CONFIG, "target_duration", "")
    monkeypatch.setitem(CONFIG, "target_duration_source", "default")
    monkeypatch.setitem(CONFIG, "clip_padding", 0.0)
    monkeypatch.setitem(CONFIG, "clip_padding_source", "default")
    _load_run_settings(tmp_path)

    assert CONFIG["edit_mode"] == "cut"
    assert CONFIG["target_duration"] == "8m"
    assert CONFIG["clip_padding"] == 0.75


def test_explicit_cli_edit_mode_wins_over_persisted(monkeypatch, tmp_path):
    monkeypatch.setitem(CONFIG, "edit_mode", "cut")
    monkeypatch.setitem(CONFIG, "edit_mode_source", "cli")
    _persist_run_settings(tmp_path)

    # Resume with an explicit CLI edit_mode=full → CLI must win.
    monkeypatch.setitem(CONFIG, "edit_mode", "full")
    monkeypatch.setitem(CONFIG, "edit_mode_source", "cli")
    _load_run_settings(tmp_path)
    assert CONFIG["edit_mode"] == "full"


# ── BUG 3: context / fps / scene-threshold / style survive resume ─────────

def test_persist_restore_context_fps_scene_threshold_style(monkeypatch, tmp_path):
    monkeypatch.setitem(CONFIG, "context_info", "节目名 X")
    monkeypatch.setitem(CONFIG, "context_info_source", "cli")
    monkeypatch.setitem(CONFIG, "fps", 0)  # RAW fps (0 = auto), persisted before resolution
    monkeypatch.setitem(CONFIG, "fps_source", "default")
    monkeypatch.setitem(CONFIG, "scene_threshold", 0.3)
    monkeypatch.setitem(CONFIG, "scene_threshold_source", "cli")
    monkeypatch.setitem(CONFIG, "style", "电影")
    monkeypatch.setitem(CONFIG, "style_source", "cli")
    settings = _persist_run_settings(tmp_path)
    assert settings["context_info"] == "节目名 X"
    assert settings["fps"] == 0
    assert settings["scene_threshold"] == 0.3
    assert settings["style"] == "电影"

    # Resume process: defaults + no override markers.
    monkeypatch.setitem(CONFIG, "context_info", "")
    monkeypatch.setitem(CONFIG, "context_info_source", "default")
    monkeypatch.setitem(CONFIG, "scene_threshold", 0.1)
    monkeypatch.setitem(CONFIG, "scene_threshold_source", "default")
    monkeypatch.setitem(CONFIG, "style", "纪录片")
    monkeypatch.setitem(CONFIG, "style_source", "default")
    _load_run_settings(tmp_path)

    assert CONFIG["context_info"] == "节目名 X"
    assert CONFIG["scene_threshold"] == 0.3
    assert CONFIG["style"] == "电影"
    assert CONFIG["fps"] == 0


def test_explicit_cli_context_and_style_win_over_persisted(monkeypatch, tmp_path):
    monkeypatch.setitem(CONFIG, "context_info", "旧上下文")
    monkeypatch.setitem(CONFIG, "context_info_source", "cli")
    monkeypatch.setitem(CONFIG, "style", "电影")
    monkeypatch.setitem(CONFIG, "style_source", "cli")
    _persist_run_settings(tmp_path)

    monkeypatch.setitem(CONFIG, "context_info", "新上下文")
    monkeypatch.setitem(CONFIG, "context_info_source", "cli")
    monkeypatch.setitem(CONFIG, "style", "科普视频")
    monkeypatch.setitem(CONFIG, "style_source", "cli")
    _load_run_settings(tmp_path)

    assert CONFIG["context_info"] == "新上下文"
    assert CONFIG["style"] == "科普视频"


def test_resume_command_appends_semantic_flags(monkeypatch, tmp_path):
    monkeypatch.setitem(CONFIG, "edit_mode", "full")
    monkeypatch.setitem(CONFIG, "tts_engine", "auto")
    monkeypatch.setitem(CONFIG, "burn_subtitles", False)
    monkeypatch.setitem(CONFIG, "context_info", "节目 A")
    monkeypatch.setitem(CONFIG, "style", "电影")
    monkeypatch.setitem(CONFIG, "scene_threshold", 0.25)
    monkeypatch.setitem(CONFIG, "fps", 1.5)
    cmd = _resume_command(Path("cli.py"), Path("in.mp4"), tmp_path)
    assert "--context" in cmd and "'节目 A'" in cmd
    assert "--style" in cmd and "电影" in cmd
    assert "--scene-threshold 0.25" in cmd
    assert "--fps 1.5" in cmd


def test_resume_command_omits_default_semantic_flags(monkeypatch, tmp_path):
    monkeypatch.setitem(CONFIG, "edit_mode", "full")
    monkeypatch.setitem(CONFIG, "tts_engine", "auto")
    monkeypatch.setitem(CONFIG, "burn_subtitles", False)
    monkeypatch.setitem(CONFIG, "context_info", "")
    monkeypatch.setitem(CONFIG, "style", "纪录片")
    monkeypatch.setitem(CONFIG, "scene_threshold", 0.1)
    monkeypatch.setitem(CONFIG, "fps", 0)
    cmd = _resume_command(Path("cli.py"), Path("in.mp4"), tmp_path)
    assert "--context" not in cmd
    assert "--style" not in cmd
    assert "--scene-threshold" not in cmd
    assert "--fps" not in cmd


# ── BUG 8: single-step and full-pipeline param fingerprints match ─────────

def test_detect_step_params_helper_includes_junk_luma(monkeypatch):
    monkeypatch.setitem(CONFIG, "scene_junk_dark_luma", 8)
    monkeypatch.setitem(CONFIG, "scene_junk_bright_luma", 245)
    params = _detect_step_params(0.1)
    # The single-step dispatch previously dropped these keys → fingerprint drift.
    assert "scene_junk_dark_luma" in params
    assert "scene_junk_bright_luma" in params
    assert params["scene_threshold"] == 0.1


def test_asr_step_params_helper_includes_bin_and_model_dir(monkeypatch):
    monkeypatch.setitem(CONFIG, "asr_bin", "local_transcribe")
    monkeypatch.setitem(CONFIG, "asr_model_dir", "/models/asr")
    params = _asr_step_params(False)
    assert "asr_bin" in params
    assert "asr_model_dir" in params
    assert params["skip_asr"] is False


def test_single_step_and_full_pipeline_detect_params_identical(monkeypatch):
    # Both the single-step dispatch and full pipeline now call _detect_step_params,
    # so their fingerprints (the dicts) are byte-identical for the same inputs.
    monkeypatch.setitem(CONFIG, "fps", 2)
    single = _detect_step_params(0.1)
    full = _detect_step_params(0.1)
    assert single == full


def test_single_step_and_full_pipeline_asr_params_identical(monkeypatch):
    single = _asr_step_params(True)
    full = _asr_step_params(True)
    assert single == full
    assert _extract_step_params() == _extract_step_params()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
