"""
Module 7 — Explain the Recommendation (LLM Layer)
====================================================
Responsibility: turn structured JSON from Modules 3-6 into plain-language
answers to a fixed set of manager questions.
"""

import os
import json


SYSTEM_PROMPT = (
    "You are a retail inventory analyst. You will be given structured JSON data "
    "about a demand forecast, a store allocation decision, or driver explanations. "
    "Answer the manager's question in 2-4 plain-English sentences. "
    "Use ONLY the numbers given in the JSON — never invent, estimate, or look up numbers "
    "that are not present in the data provided."
)


def call_llm(prompt: str) -> str:
    """
    Calls the Anthropic API if ANTHROPIC_API_KEY is set; otherwise
    returns a clearly-labeled offline fallback so the module still runs
    end-to-end without network access or credentials.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[offline fallback — no ANTHROPIC_API_KEY set]\n" + _naive_fill(prompt)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
    except Exception as e:
        return f"[LLM call failed: {e}]\n" + _naive_fill(prompt)


def _naive_fill(prompt: str) -> str:
    """
    A deterministic non-LLM fallback: just echoes the structured data in
    a readable form. Not a substitute for the real workflow output, but
    keeps a full offline demo possible.
    """
    return "Structured data for this query:\n" + prompt.split("DATA:\n", 1)[-1]


# Workflow 1: Explain Decision
def explain_decision(store_id, allocation_row: dict, drivers: list) -> str:
    prompt = (
        f"A manager asked: \"Why is Store {store_id} receiving "
        f"{allocation_row.get('allocated_units')} units?\"\n\n"
        f"DATA:\n{json.dumps({'allocation': allocation_row, 'top_drivers': drivers}, indent=2)}"
    )
    return call_llm(prompt)


# Workflow 2: Compare Stores
def compare_stores(store_a_id, store_a_row: dict, store_b_id, store_b_row: dict) -> str:
    prompt = (
        f"A manager asked: \"Why is Store {store_a_id} different from Store {store_b_id}?\"\n\n"
        f"DATA:\n{json.dumps({'store_a': store_a_row, 'store_b': store_b_row}, indent=2)}"
    )
    return call_llm(prompt)


# Workflow 3: Explain Forecast
def explain_forecast(forecast_row: dict, drivers: list) -> str:
    prompt = (
        "A manager asked: \"What factors increased demand this week?\"\n\n"
        f"DATA:\n{json.dumps({'forecast': forecast_row, 'top_drivers': drivers}, indent=2)}"
    )
    return call_llm(prompt)


# Workflow 4: Executive Summary
def executive_summary(kpis: dict, insights: list, allocation_summary: dict) -> str:
    prompt = (
        "A manager asked: \"Summarize this week's inventory decisions.\"\n\n"
        f"DATA:\n{json.dumps({'kpis': kpis, 'insights': insights, 'allocation_summary': allocation_summary}, indent=2)}"
    )
    return call_llm(prompt)


if __name__ == "__main__":
    demo_allocation = {"store_id": 12, "allocated_units": 300, "effective_demand": 320, "fill_rate_pct": 93.8}
    demo_drivers = [
        {"feature": "Holiday season", "direction": "increase", "approx_pct_impact": 21.0},
        {"feature": "Promotion", "direction": "increase", "approx_pct_impact": 14.0},
    ]
    print("--- Explain Decision ---")
    print(explain_decision(12, demo_allocation, demo_drivers))
