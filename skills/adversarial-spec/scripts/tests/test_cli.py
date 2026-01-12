"""Tests for CLI argument parsing and command routing."""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from io import StringIO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCLIProviders:
    def test_providers_command(self):
        """Test that providers command runs without error."""
        import debate

        with patch("sys.argv", ["debate.py", "providers"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                debate.main()
                output = mock_stdout.getvalue()
                assert "OpenAI" in output
                assert "OPENAI_API_KEY" in output


class TestCLIFocusAreas:
    def test_focus_areas_command(self):
        """Test that focus-areas command lists all areas."""
        import debate

        with patch("sys.argv", ["debate.py", "focus-areas"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                debate.main()
                output = mock_stdout.getvalue()
                assert "security" in output
                assert "scalability" in output
                assert "performance" in output


class TestCLIPersonas:
    def test_personas_command(self):
        """Test that personas command lists all personas."""
        import debate

        with patch("sys.argv", ["debate.py", "personas"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                debate.main()
                output = mock_stdout.getvalue()
                assert "security-engineer" in output
                assert "oncall-engineer" in output


class TestCLISessions:
    def test_sessions_command_empty(self):
        """Test sessions command with no sessions."""
        import debate

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("session.SESSIONS_DIR", Path(tmpdir) / "sessions"):
                with patch("debate.SESSIONS_DIR", Path(tmpdir) / "sessions"):
                    with patch("sys.argv", ["debate.py", "sessions"]):
                        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                            debate.main()
                            output = mock_stdout.getvalue()
                            assert "No sessions found" in output


class TestCLIDiff:
    def test_diff_command(self):
        """Test diff between two files."""
        import debate

        with tempfile.TemporaryDirectory() as tmpdir:
            prev = Path(tmpdir) / "prev.md"
            curr = Path(tmpdir) / "curr.md"
            prev.write_text("line1\nline2\n")
            curr.write_text("line1\nmodified\n")

            with patch(
                "sys.argv",
                ["debate.py", "diff", "--previous", str(prev), "--current", str(curr)],
            ):
                with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                    debate.main()
                    output = mock_stdout.getvalue()
                    assert "-line2" in output
                    assert "+modified" in output


class TestCLISaveProfile:
    def test_save_profile_command(self):
        """Test saving a profile."""
        import debate

        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir) / "profiles"

            with patch("providers.PROFILES_DIR", profiles_dir):
                with patch(
                    "sys.argv",
                    [
                        "debate.py",
                        "save-profile",
                        "test-profile",
                        "--models",
                        "gpt-4o,gemini/gemini-2.0-flash",
                        "--focus",
                        "security",
                    ],
                ):
                    with patch("sys.stdout", new_callable=StringIO):
                        debate.main()

                        # Verify profile was saved
                        profile_path = profiles_dir / "test-profile.json"
                        assert profile_path.exists()

                        data = json.loads(profile_path.read_text())
                        assert data["models"] == "gpt-4o,gemini/gemini-2.0-flash"
                        assert data["focus"] == "security"


class TestCLICritique:
    @patch("debate.call_models_parallel")
    def test_critique_with_json_output(self, mock_call):
        """Test critique command with JSON output."""
        import debate
        from models import ModelResponse

        mock_call.return_value = [
            ModelResponse(
                model="gpt-4o",
                response="Critique here.\n[SPEC]\n# Revised\n[/SPEC]",
                agreed=False,
                spec="# Revised",
                input_tokens=100,
                output_tokens=50,
                cost=0.01,
            )
        ]

        with patch("sys.stdin", StringIO("# Test Spec\n\nContent here.")):
            with patch(
                "sys.argv", ["debate.py", "critique", "--models", "gpt-4o", "--json"]
            ):
                with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                    with patch("sys.stderr", new_callable=StringIO):
                        debate.main()
                        output = mock_stdout.getvalue()

                        data = json.loads(output)
                        assert data["round"] == 1
                        assert data["models"] == ["gpt-4o"]
                        assert len(data["results"]) == 1
                        assert data["results"][0]["model"] == "gpt-4o"

    @patch("debate.call_models_parallel")
    def test_critique_with_all_agree(self, mock_call):
        """Test critique when all models agree."""
        import debate
        from models import ModelResponse

        mock_call.return_value = [
            ModelResponse(
                model="gpt-4o",
                response="[AGREE]\n[SPEC]\n# Final\n[/SPEC]",
                agreed=True,
                spec="# Final",
                input_tokens=100,
                output_tokens=50,
                cost=0.01,
            ),
            ModelResponse(
                model="gemini/gemini-2.0-flash",
                response="[AGREE]\n[SPEC]\n# Final\n[/SPEC]",
                agreed=True,
                spec="# Final",
                input_tokens=80,
                output_tokens=40,
                cost=0.005,
            ),
        ]

        with patch("sys.stdin", StringIO("# Test Spec")):
            with patch(
                "sys.argv",
                [
                    "debate.py",
                    "critique",
                    "--models",
                    "gpt-4o,gemini/gemini-2.0-flash",
                    "--json",
                ],
            ):
                with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                    with patch("sys.stderr", new_callable=StringIO):
                        debate.main()
                        output = mock_stdout.getvalue()

                        data = json.loads(output)
                        assert data["all_agreed"] is True

    @patch("debate.call_models_parallel")
    def test_critique_passes_options(self, mock_call):
        """Test that CLI options are passed to model calls."""
        import debate
        from models import ModelResponse

        mock_call.return_value = [
            ModelResponse(
                model="gpt-4o",
                response="[AGREE]\n[SPEC]\n# Spec\n[/SPEC]",
                agreed=True,
                spec="# Spec",
            )
        ]

        with patch("sys.stdin", StringIO("# Spec")):
            with patch(
                "sys.argv",
                [
                    "debate.py",
                    "critique",
                    "--models",
                    "gpt-4o",
                    "--focus",
                    "security",
                    "--persona",
                    "security-engineer",
                    "--preserve-intent",
                    "--json",
                ],
            ):
                with patch("sys.stdout", new_callable=StringIO):
                    with patch("sys.stderr", new_callable=StringIO):
                        debate.main()

                        # Verify options were passed (positional args)
                        # call_models_parallel(models, spec, round_num, doc_type, press,
                        #                      focus, persona, context, preserve_intent, ...)
                        call_args = mock_call.call_args[0]
                        assert call_args[0] == ["gpt-4o"]  # models
                        assert call_args[5] == "security"  # focus
                        assert call_args[6] == "security-engineer"  # persona
                        assert call_args[8] is True  # preserve_intent


class TestCLIBedrock:
    def test_bedrock_status_not_configured(self):
        """Test bedrock status when not configured."""
        import debate

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with patch("sys.argv", ["debate.py", "bedrock", "status"]):
                    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                        debate.main()
                        output = mock_stdout.getvalue()
                        assert "Not configured" in output

    def test_bedrock_enable(self):
        """Test enabling bedrock mode."""
        import debate

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "adversarial-spec" / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with patch(
                    "sys.argv",
                    ["debate.py", "bedrock", "enable", "--region", "us-east-1"],
                ):
                    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                        debate.main()
                        output = mock_stdout.getvalue()
                        assert "enabled" in output.lower()

                        # Verify config was written
                        assert config_path.exists()
                        data = json.loads(config_path.read_text())
                        assert data["bedrock"]["enabled"] is True
                        assert data["bedrock"]["region"] == "us-east-1"
