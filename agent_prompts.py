"""Prompt templates shared by the LangGraph and Agent Framework workflows."""

from __future__ import annotations


def language_detection_prompt() -> str:
    return (
        "Determine the response language for the user's message. "
        "Reply with exactly one word: Japanese or English. "
        "Only return 'Japanese' if the message is clearly in Japanese; "
        "otherwise return 'English'."
    )


def summary_prompt(
    previous_summary: str,
    *,
    research_notes: str = "",
    draft_notes: str = "",
    verification_notes: str = "",
    transcript: str = "",
) -> str:
    prompt = (
        "Create a concise, factual conversation summary for the main agent. Preserve user "
        "requirements, decisions, relevant tool results, errors, and unresolved tasks. "
        "Do not add new facts. Summarize supplied conversation messages, including "
        "information conveyed by attached images.\n\n"
        f"Previous summary:\n{previous_summary or '(none)'}"
    )
    if research_notes or draft_notes or verification_notes or transcript:
        prompt += (
            f"\n\nResearch notes:\n{research_notes or '(none)'}"
            f"\n\nDraft answer:\n{draft_notes or '(none)'}"
            f"\n\nVerification notes:\n{verification_notes or '(none)'}"
        )
    if transcript:
        prompt += f"\n\nMessages to compact:\n{transcript}"
    return prompt


def session_time_instruction(started_at: str) -> str:
    return f"Chat start date and time: {started_at}.\n"


def route_prompt(started_at: str) -> str:
    return (
        session_time_instruction(started_at)
        + "Classify the user's request. Reply with exactly one word:\n"
        "RESEARCH if it needs external facts, current information, MCP tools, or verification against real-world data.\n"
        "COMPLICATED if it does NOT need external research or tools, but is a non-trivial reasoning, math, logic, coding, or writing task where a careful draft should be checked and refined before answering (e.g. multi-step problems, proofs, code that must be correct, precise or high-stakes writing).\n"
        "DIRECT for casual conversation, simple translation, simple rewriting, or other trivial tasks that need no verification.\n"
        "For anything else or if uncertain, reply MODERATED."
    )


def _refinement_instruction(
    *,
    refine_count: int,
    max_refine_loops: int,
    verification_notes: str,
    material_name: str,
) -> str:
    if not refine_count:
        return ""
    return (
        f"\n\nThis is a refinement pass (attempt {refine_count + 1}/{max_refine_loops + 1}). "
        f"The verification phase found problems with the previous {material_name}. "
        f"Address the following issues specifically before returning updated {material_name}:\n"
        f"{verification_notes}"
    )


def research_prompt(
    *,
    started_at: str,
    refine_count: int = 0,
    max_refine_loops: int = 2,
    verification_notes: str = "",
) -> str:
    return (
        session_time_instruction(started_at)
        + "You are the research phase of the main agent. Investigate the user's request "
        "and use available tools when useful. Return factual findings and unresolved points."
        + _refinement_instruction(
            refine_count=refine_count,
            max_refine_loops=max_refine_loops,
            verification_notes=verification_notes,
            material_name="research",
        )
    )


def draft_prompt(
    *,
    started_at: str,
    moderated: bool,
    refine_count: int = 0,
    max_refine_loops: int = 2,
    verification_notes: str = "",
) -> str:
    if moderated:
        base = (
            "You are the moderated drafting phase of the main agent. This task requires "
            "some careful consideration but does NOT require external research or tools. "
            "Produce a candidate answer for the verification phase to check."
        )
        material_name = "draft"
    else:
        base = (
            "You are the drafting phase of the main agent. This task is complicated but "
            "does NOT require external research or tools. Work through it carefully, step "
            "by step, and produce a complete candidate answer or solution for the "
            "verification phase to check."
        )
        material_name = "draft"
    return (
        session_time_instruction(started_at)
        + base
        + _refinement_instruction(
            refine_count=refine_count,
            max_refine_loops=max_refine_loops,
            verification_notes=verification_notes,
            material_name=material_name,
        )
    )


def verification_prompt(*, started_at: str, material_label: str, material: str) -> str:
    return (
        session_time_instruction(started_at)
        + f"You are the verification phase. Check the {material_label} below for contradictions, "
        "missing information, unsupported claims, dates, times, units, logical errors, and any "
        "requirements from the user's request that were not met. Return concise verification "
        "notes and corrections.\n\n"
        f"{material_label.capitalize()}:\n{material}\n\n"
        "End your reply with exactly one final line, with no other text on it: 'STATUS: OK' if "
        "this is sound and sufficient to answer the user, or 'STATUS: NEEDS_REVISION' if it "
        "must be corrected or completed before answering."
    )


def answer_prompt(
    *,
    answer_language: str,
    started_at: str,
    research_notes: str = "",
    draft_notes: str = "",
    verification_notes: str = "",
    verification_status: str = "OK",
) -> str:
    notes_section = ""
    if research_notes:
        notes_section += f"Research notes:\n{research_notes}\n\n"
    if draft_notes:
        notes_section += f"Draft answer:\n{draft_notes}\n\n"
    if verification_notes:
        notes_section += f"Verification notes:\n{verification_notes}\n\n"
    instructions = (
        session_time_instruction(started_at)
        + "You are the main agent. "
        + ("Answer the user's request using the notes below." if notes_section else "Answer the user's request.")
        + " Do not mention internal phases or hidden reasoning. Answer entirely in "
        f"{answer_language}.\n\n{notes_section}"
    )
    if verification_status != "OK":
        instructions += (
            "The verification phase did not confirm the material as sound. Be transparent about "
            "any unresolved or unverified points in your answer. Do not present them as verified facts.\n"
        )
    return instructions
