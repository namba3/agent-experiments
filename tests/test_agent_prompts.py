"""Tests for shared classification and workflow prompt templates."""

from __future__ import annotations

import unittest

from agent_prompts import (
    answer_prompt,
    draft_prompt,
    research_prompt,
    route_prompt,
    summary_prompt,
    verification_prompt,
)


class PromptTemplateTests(unittest.TestCase):
    def test_route_prompt_lists_supported_labels_and_fallback(self) -> None:
        prompt = route_prompt("2026-10-04T12:00:00+09:00")

        for label in ("SIMPLE", "STANDARD", "DEEP", "RESEARCH"):
            with self.subTest(label=label):
                self.assertIn(label, prompt)
        self.assertNotIn("MODERATED", prompt)
        self.assertNotIn("COMPLICATED", prompt)
        self.assertNotIn("DIRECT", prompt)
        self.assertIn("Chat start date and time:", prompt)

    def test_research_prompt_includes_refinement_feedback(self) -> None:
        prompt = research_prompt(
            started_at="start",
            refine_count=1,
            max_refine_loops=2,
            verification_notes="Check the source date.",
        )

        self.assertIn("attempt 2/3", prompt)
        self.assertIn("Check the source date.", prompt)

    def test_draft_prompt_distinguishes_standard_and_deep_reasoning(self) -> None:
        standard = draft_prompt(started_at="start", standard=True)
        deep = draft_prompt(started_at="start", standard=False)

        self.assertIn("standard drafting phase", standard)
        self.assertIn("deep-reasoning drafting phase", deep)
        self.assertIn("verification phase to check", standard)
        self.assertIn("verification phase to check", deep)

    def test_verification_prompt_requires_a_final_status(self) -> None:
        prompt = verification_prompt(
            started_at="start",
            material_label="draft answer",
            material="candidate",
        )

        self.assertIn("Draft answer:\ncandidate", prompt)
        self.assertIn("STATUS: OK", prompt)
        self.assertIn("STATUS: NEEDS_REVISION", prompt)

    def test_answer_prompt_discloses_unverified_material(self) -> None:
        prompt = answer_prompt(
            answer_language="Japanese",
            started_at="start",
            verification_notes="uncertain date",
            verification_status="INVALID",
        )

        self.assertIn("Answer entirely in Japanese", prompt)
        self.assertIn("uncertain date", prompt)
        self.assertIn("Do not present them as verified facts", prompt)

    def test_summary_prompt_retains_decisions_and_prior_notes(self) -> None:
        prompt = summary_prompt(
            "existing summary",
            research_notes="source finding",
            draft_notes="draft content",
            verification_notes="open issue",
            transcript="user: keep this requirement",
        )

        for text in (
            "existing summary",
            "source finding",
            "draft content",
            "open issue",
            "user: keep this requirement",
        ):
            with self.subTest(text=text):
                self.assertIn(text, prompt)


if __name__ == "__main__":
    unittest.main()
