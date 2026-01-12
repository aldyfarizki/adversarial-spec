"""Tests for prompts module."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts import (
    get_system_prompt,
    get_doc_type_name,
    SYSTEM_PROMPT_PRD,
    SYSTEM_PROMPT_TECH,
    SYSTEM_PROMPT_GENERIC,
    FOCUS_AREAS,
    PERSONAS,
    PRESERVE_INTENT_PROMPT,
)


class TestGetSystemPrompt:
    def test_prd_returns_prd_prompt(self):
        result = get_system_prompt("prd")
        assert result == SYSTEM_PROMPT_PRD
        assert "Product Requirements Document" in result

    def test_tech_returns_tech_prompt(self):
        result = get_system_prompt("tech")
        assert result == SYSTEM_PROMPT_TECH
        assert "Technical Specification" in result

    def test_unknown_returns_generic_prompt(self):
        result = get_system_prompt("unknown")
        assert result == SYSTEM_PROMPT_GENERIC

    def test_known_persona_returns_persona_prompt(self):
        result = get_system_prompt("tech", persona="security-engineer")
        assert result == PERSONAS["security-engineer"]
        assert "security engineer" in result

    def test_unknown_persona_returns_custom_prompt(self):
        result = get_system_prompt("tech", persona="fintech auditor")
        assert "fintech auditor" in result
        assert "adversarial spec development" in result

    def test_persona_overrides_doc_type(self):
        result = get_system_prompt("prd", persona="oncall-engineer")
        assert result == PERSONAS["oncall-engineer"]
        assert "on-call engineer" in result


class TestGetDocTypeName:
    def test_prd(self):
        assert get_doc_type_name("prd") == "Product Requirements Document"

    def test_tech(self):
        assert get_doc_type_name("tech") == "Technical Specification"

    def test_unknown(self):
        assert get_doc_type_name("other") == "specification"


class TestFocusAreas:
    def test_all_focus_areas_exist(self):
        expected = [
            "security",
            "scalability",
            "performance",
            "ux",
            "reliability",
            "cost",
        ]
        for area in expected:
            assert area in FOCUS_AREAS

    def test_focus_areas_contain_critical_focus(self):
        for name, content in FOCUS_AREAS.items():
            assert "CRITICAL FOCUS" in content


class TestPersonas:
    def test_all_personas_exist(self):
        expected = [
            "security-engineer",
            "oncall-engineer",
            "junior-developer",
            "qa-engineer",
            "site-reliability",
            "product-manager",
            "data-engineer",
            "mobile-developer",
            "accessibility-specialist",
            "legal-compliance",
        ]
        for persona in expected:
            assert persona in PERSONAS

    def test_personas_are_non_empty(self):
        for name, content in PERSONAS.items():
            assert len(content) > 50


class TestPreserveIntentPrompt:
    def test_contains_key_instructions(self):
        assert "PRESERVE ORIGINAL INTENT" in PRESERVE_INTENT_PROMPT
        assert "ASSUME the author had good reasons" in PRESERVE_INTENT_PROMPT
        assert "ERRORS" in PRESERVE_INTENT_PROMPT
        assert "RISKS" in PRESERVE_INTENT_PROMPT
        assert "PREFERENCES" in PRESERVE_INTENT_PROMPT
