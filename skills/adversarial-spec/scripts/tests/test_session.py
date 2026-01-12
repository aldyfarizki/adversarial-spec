"""Tests for session module."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from session import SessionState, save_checkpoint


class TestSessionState:
    def test_create_session_state(self):
        session = SessionState(
            session_id="test-session",
            spec="# Test Spec\n\nContent here.",
            round=1,
            doc_type="tech",
            models=["gpt-4o", "gemini/gemini-2.0-flash"],
        )
        assert session.session_id == "test-session"
        assert session.round == 1
        assert session.doc_type == "tech"
        assert len(session.models) == 2

    def test_session_with_optional_fields(self):
        session = SessionState(
            session_id="test",
            spec="spec",
            round=2,
            doc_type="prd",
            models=["gpt-4o"],
            focus="security",
            persona="security-engineer",
            preserve_intent=True,
        )
        assert session.focus == "security"
        assert session.persona == "security-engineer"
        assert session.preserve_intent is True

    def test_save_and_load_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"

            with patch("session.SESSIONS_DIR", sessions_dir):
                session = SessionState(
                    session_id="save-test",
                    spec="test spec content",
                    round=3,
                    doc_type="tech",
                    models=["gpt-4o"],
                    focus="performance",
                )
                session.save()

                # Verify file exists
                assert (sessions_dir / "save-test.json").exists()

                # Load and verify
                loaded = SessionState.load("save-test")
                assert loaded.session_id == "save-test"
                assert loaded.spec == "test spec content"
                assert loaded.round == 3
                assert loaded.focus == "performance"
                assert loaded.updated_at != ""

    def test_load_nonexistent_session_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()

            with patch("session.SESSIONS_DIR", sessions_dir):
                try:
                    SessionState.load("nonexistent")
                    assert False, "Should have raised FileNotFoundError"
                except FileNotFoundError as e:
                    assert "nonexistent" in str(e)

    def test_list_sessions_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"

            with patch("session.SESSIONS_DIR", sessions_dir):
                sessions = SessionState.list_sessions()
                assert sessions == []

    def test_list_sessions_returns_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()

            with patch("session.SESSIONS_DIR", sessions_dir):
                # Create two sessions
                s1 = SessionState(
                    session_id="first",
                    spec="spec1",
                    round=1,
                    doc_type="prd",
                    models=["gpt-4o"],
                )
                s1.save()

                s2 = SessionState(
                    session_id="second",
                    spec="spec2",
                    round=2,
                    doc_type="tech",
                    models=["gemini/gemini-2.0-flash"],
                )
                s2.save()

                sessions = SessionState.list_sessions()
                assert len(sessions) == 2
                # Most recent first
                assert sessions[0]["id"] == "second"
                assert sessions[1]["id"] == "first"

    def test_session_history_append(self):
        session = SessionState(
            session_id="history-test",
            spec="spec",
            round=1,
            doc_type="tech",
            models=["gpt-4o"],
        )
        assert session.history == []

        session.history.append(
            {
                "round": 1,
                "all_agreed": False,
                "models": [{"model": "gpt-4o", "agreed": False}],
            }
        )
        assert len(session.history) == 1
        assert session.history[0]["round"] == 1


class TestSaveCheckpoint:
    def test_save_checkpoint_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"

            with patch("session.CHECKPOINTS_DIR", checkpoint_dir):
                save_checkpoint("# Test Spec", 1)

                assert (checkpoint_dir / "round-1.md").exists()
                content = (checkpoint_dir / "round-1.md").read_text()
                assert content == "# Test Spec"

    def test_save_checkpoint_with_session_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"

            with patch("session.CHECKPOINTS_DIR", checkpoint_dir):
                save_checkpoint("# Test Spec", 2, session_id="my-session")

                assert (checkpoint_dir / "my-session-round-2.md").exists()
