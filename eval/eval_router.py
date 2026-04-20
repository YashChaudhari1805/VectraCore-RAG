"""
eval/eval_router.py
-------------------
Measures routing accuracy across 20 labeled test posts.

Each test post has a defined `expected_top_bot` — the bot that should
rank #1 by cosine similarity. The eval measures:

  - Top-1 Accuracy : correct bot ranked #1
  - Any-Match Rate : correct bot appears in matched results at all
  - Mean Similarity: average cosine score of the correct bot
  - Per-bot breakdown

Run from project root:
    python -m eval.eval_router
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from core.router import build_index, route_post, get_all_scores

# ── 20 labeled test posts ─────────────────────────────────────────────────────

TEST_POSTS = [
    # Bot A — Tech Maximalist
    {"post": "GPT-5 just dropped and it's already replacing junior devs at top firms.",          "expected_top_bot": "Bot_A_TechMaximalist"},
    {"post": "SpaceX just landed its 30th booster. Reusable rockets are the future.",            "expected_top_bot": "Bot_A_TechMaximalist"},
    {"post": "Elon Musk says Neuralink will let humans merge with AI by 2030.",                  "expected_top_bot": "Bot_A_TechMaximalist"},
    {"post": "Bitcoin just hit a new all-time high. Crypto is the future of money.",             "expected_top_bot": "Bot_A_TechMaximalist"},
    {"post": "AI will cure cancer, climate change, and poverty. The future is now.",             "expected_top_bot": "Bot_A_TechMaximalist"},

    # Bot B — Doomer
    {"post": "Meta is selling your private messages to advertisers without consent.",            "expected_top_bot": "Bot_B_Doomer"},
    {"post": "Billionaires paid zero taxes last year while workers struggle.",                   "expected_top_bot": "Bot_B_Doomer"},
    {"post": "Amazon rainforest hit a new deforestation record. Corporations win, nature loses.","expected_top_bot": "Bot_B_Doomer"},
    {"post": "Facial recognition cameras are being installed in every major city.",              "expected_top_bot": "Bot_B_Doomer"},
    {"post": "Social media is engineered to be addictive and it's destroying mental health.",    "expected_top_bot": "Bot_B_Doomer"},

    # Bot C — Finance Bro
    {"post": "Fed signals two rate cuts in Q3. Time to rotate into equities.",                   "expected_top_bot": "Bot_C_FinanceBro"},
    {"post": "S&P 500 hits all-time high on strong jobs data. Bull market confirmed.",           "expected_top_bot": "Bot_C_FinanceBro"},
    {"post": "Nvidia earnings beat expectations by 40%. Semiconductors are the new oil.",        "expected_top_bot": "Bot_C_FinanceBro"},
    {"post": "Hedge funds are shorting regional banks ahead of the next credit cycle.",          "expected_top_bot": "Bot_C_FinanceBro"},
    {"post": "10-year treasury yield just inverted again. Recession probability now at 70%.",    "expected_top_bot": "Bot_C_FinanceBro"},

    # Mixed / ambiguous — tests nuance
    {"post": "OpenAI is now valued at $300B. Is this a bubble or justified growth?",             "expected_top_bot": "Bot_A_TechMaximalist"},
    {"post": "Apple fined €1.8B by EU. Big Tech regulation is finally happening.",              "expected_top_bot": "Bot_B_Doomer"},
    {"post": "Bitcoin ETF approved. Institutional money is flooding into crypto markets.",       "expected_top_bot": "Bot_C_FinanceBro"},
    {"post": "Elon Musk's net worth surges $20B in one day on Tesla stock rally.",               "expected_top_bot": "Bot_A_TechMaximalist"},
    {"post": "AI startup funding hits $50B globally. Where are the best risk-adjusted returns?", "expected_top_bot": "Bot_C_FinanceBro"},
]

# ── Evaluation ────────────────────────────────────────────────────────────────

def run_eval(threshold: float = 0.18) -> dict:
    print("=" * 65)
    print("GRID07 — ROUTER EVALUATION")
    print(f"Threshold: {threshold} | Test posts: {len(TEST_POSTS)}")
    print("=" * 65)

    build_index()

    top1_correct   = 0
    any_match      = 0
    correct_scores = []
    per_bot        = {bot: {"total": 0, "top1": 0, "any": 0} for bot in [
        "Bot_A_TechMaximalist", "Bot_B_Doomer", "Bot_C_FinanceBro"
    ]}

    print(f"\n{'#':<4} {'Expected':<24} {'Got #1':<24} {'Score':>7}  {'✓'}")
    print("─" * 65)

    for i, test in enumerate(TEST_POSTS, 1):
        post     = test["post"]
        expected = test["expected_top_bot"]
        scores   = get_all_scores(post)   # all bots, no threshold

        # Top-1: which bot ranked highest?
        top1     = scores[0]["bot_id"] if scores else "none"
        top1_sim = scores[0]["similarity"] if scores else 0.0

        # Score of the expected bot
        expected_score = next((s["similarity"] for s in scores if s["bot_id"] == expected), 0.0)
        correct_scores.append(expected_score)

        # Matched (above threshold)
        matched_ids = [s["bot_id"] for s in scores if s["similarity"] >= threshold]

        is_top1 = (top1 == expected)
        is_any  = (expected in matched_ids)

        if is_top1:  top1_correct += 1
        if is_any:   any_match    += 1

        per_bot[expected]["total"] += 1
        if is_top1: per_bot[expected]["top1"] += 1
        if is_any:  per_bot[expected]["any"]  += 1

        tick = "✅" if is_top1 else ("⚠️ " if is_any else "❌")
        short_expected = expected.replace("Bot_", "").replace("_", " ")
        short_top1     = top1.replace("Bot_", "").replace("_", " ")
        print(f"{i:<4} {short_expected:<24} {short_top1:<24} {expected_score:>6.4f}  {tick}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n              = len(TEST_POSTS)
    top1_pct       = top1_correct / n * 100
    any_pct        = any_match    / n * 100
    mean_sim       = sum(correct_scores) / n

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Top-1 Accuracy  : {top1_correct}/{n}  ({top1_pct:.1f}%)")
    print(f"  Any-Match Rate  : {any_match}/{n}  ({any_pct:.1f}%)")
    print(f"  Mean Similarity : {mean_sim:.4f}")

    print("\nPer-bot breakdown:")
    for bot_id, stats in per_bot.items():
        t    = stats["total"]
        top1 = stats["top1"]
        any_ = stats["any"]
        name = bot_id.replace("Bot_", "").replace("_", " ")
        print(f"  {name:<20} Top-1: {top1}/{t}  Any-match: {any_}/{t}")

    print("\nLegend:  ✅ Top-1 correct   ⚠️  In results but not #1   ❌ Not matched")
    print("=" * 65)

    return {
        "top1_accuracy":    round(top1_pct, 1),
        "any_match_rate":   round(any_pct, 1),
        "mean_similarity":  round(mean_sim, 4),
        "per_bot":          per_bot,
        "threshold":        threshold,
        "total_tests":      n,
    }


if __name__ == "__main__":
    results = run_eval(threshold=0.18)
