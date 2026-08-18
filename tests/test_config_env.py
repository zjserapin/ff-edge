"""The `.env` loader, and the one property that makes it safe to add mid-season.

`config._load_env_file` was added on 2026-08-17 because a FantasyPros key had
been sitting in `.env` unread — nothing in `src/` called a dotenv loader, so the
file was decoration. Adding one five days before a draft is only safe if it
cannot change how anything already-working resolves, which is exactly what these
pin.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from src import config


@pytest.fixture(autouse=True)
def _restore_config() -> object:
    """Put `config` back the way it was found.

    These tests `importlib.reload(config)`, which rebinds attributes on the one
    module object every other module imported. Without this the last reload's
    environment would leak into the rest of the suite — and because config holds
    seasons and paths, a leak there is the kind that produces a plausible wrong
    answer somewhere unrelated rather than an error here.
    """
    yield
    importlib.reload(config)


def _write_env(tmp_path: Path, body: str) -> Path:
    """A fake repo root with a .env in it, shaped the way config expects."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / ".env").write_text(body)
    return root


def _load_from(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real loader against `root` by pointing __file__ at root/src."""
    monkeypatch.setattr(config, "__file__", str(root / "src" / "config.py"))
    config._load_env_file()


def test_a_real_export_beats_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The safety property.** A command-line export must still win.

    This is the whole reason the loader uses `setdefault`. If it ever overrode,
    `FF_EDGE_LEAGUE_ID=... uv run ...` would silently resolve to whatever stale
    id happened to be in `.env`, which is the 2026 superflex bug's shape: a
    plausible number from the wrong source, and nothing raises.
    """
    monkeypatch.setenv("FF_EDGE_TEST_KEY", "from-the-shell")
    root = _write_env(tmp_path, "FF_EDGE_TEST_KEY=from-the-file\n")

    _load_from(root, monkeypatch)

    assert os.environ["FF_EDGE_TEST_KEY"] == "from-the-shell"


def test_a_name_absent_from_the_environment_is_filled_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FF_EDGE_TEST_ABSENT", raising=False)
    root = _write_env(tmp_path, "FF_EDGE_TEST_ABSENT=from-the-file\n")

    _load_from(root, monkeypatch)

    assert os.environ["FF_EDGE_TEST_ABSENT"] == "from-the-file"


@pytest.mark.parametrize(
    "line, expected",
    [
        ("FF_EDGE_T=plain", "plain"),
        ("FF_EDGE_T='single'", "single"),
        ('FF_EDGE_T="double"', "double"),
        ("  FF_EDGE_T = spaced  ", "spaced"),
        ("FF_EDGE_T=has=equals", "has=equals"),
    ],
)
def test_line_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str, expected: str
) -> None:
    """Quotes stripped, whitespace trimmed, and only the FIRST `=` splits.

    The last case is the one that matters: an API key containing `=` must not be
    truncated at its own padding.
    """
    monkeypatch.delenv("FF_EDGE_T", raising=False)
    root = _write_env(tmp_path, line + "\n")

    _load_from(root, monkeypatch)

    assert os.environ["FF_EDGE_T"] == expected


def test_comments_and_blanks_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FF_EDGE_REAL", raising=False)
    root = _write_env(
        tmp_path, "\n# FF_EDGE_COMMENTED=nope\n\nFF_EDGE_REAL=yes\nnot-a-pair\n"
    )

    _load_from(root, monkeypatch)

    assert os.environ["FF_EDGE_REAL"] == "yes"
    assert "FF_EDGE_COMMENTED" not in os.environ


def test_a_missing_env_file_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh clone has no `.env` and must import config anyway."""
    root = tmp_path / "bare"
    (root / "src").mkdir(parents=True)

    _load_from(root, monkeypatch)  # must not raise


def test_the_fantasypros_key_accepts_either_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`.env` in this repo already said `fantasypros_api`; both names resolve.

    Reimports config so the module-level constant is recomputed under the
    patched environment.
    """
    monkeypatch.delenv("FF_EDGE_FP_API_KEY", raising=False)
    monkeypatch.setenv("fantasypros_api", "legacy-spelling")
    reloaded = importlib.reload(config)
    assert reloaded.FANTASYPROS_API_KEY == "legacy-spelling"

    monkeypatch.setenv("FF_EDGE_FP_API_KEY", "canonical-spelling")
    reloaded = importlib.reload(config)
    assert reloaded.FANTASYPROS_API_KEY == "canonical-spelling"


def test_no_key_is_empty_string_not_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same convention as ANTHROPIC_API_KEY, so callers can test truthiness.

    Both names are set to `""` rather than deleted, because `reload` re-executes
    the module and re-runs the real `_load_env_file` — patching that function out
    would not survive the reload that is supposed to observe it. `setdefault`
    leaves an existing empty value alone, so this reaches the no-key branch even
    on a machine whose `.env` holds a real key.
    """
    monkeypatch.setenv("FF_EDGE_FP_API_KEY", "")
    monkeypatch.setenv("fantasypros_api", "")
    reloaded = importlib.reload(config)
    assert reloaded.FANTASYPROS_API_KEY == ""


def test_the_free_tier_cap_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the probed limit so a future session does not re-derive it by hand.

    Ten is not a number this project chose; it is what
    `/nfl/2026/consensus-rankings` returns alongside `public_api_limited: True`
    while reporting `count: 270`.
    """
    reloaded = importlib.reload(config)
    assert reloaded.FANTASYPROS_FREE_TIER_LIMIT == 10
