"""Controlled diagnostic seed for measuring verification performance.

    python seed_eval.py --reset        # wipe, then ingest the benchmark corpus
    python seed_eval.py                # ingest on top of existing data (not advised)

Unlike seed_news.py (a noisy real-news snapshot) this is a SMALL, ENGINEERED
corpus. Each scenario is entity-disjoint and built so the *correct* verification
behaviour is known in advance, and several scenarios deliberately set up a
DIVERGENCE between the system's likely status and the warranted truth — those are
what give the eval real precision AND recall signal, which a corpus where every
inference is true cannot.

Scenarios (see EVAL_SEED_NOTES.md for the full rationale + what to watch):

  A  Veridia        -> exercises the DEFEATER path (status should reach 'contested')
  B  Marran         -> dense, well-evidenced causal chain (true positive)
  C  Pelora         -> TOPICAL-vs-CAUSAL trap: co-movement with no causal evidence.
                       If it lands 'corroborated', that is a measurable PRECISION miss.
  D  Khelas         -> reproduces the CONVERGENCE-OVER-CAP bug: two unsupported,
                       independently-grounded derivations of the same claim. Watch
                       for status='unverified' with confidence pushed ABOVE 0.55.
  E  Tovar          -> thin + speculative (true negative / unverifiable)
  F  Anvaria/Coastal-> warranted & TRUE but lacks corroborating evidence. If it
                       lands 'unverified', that is a measurable RECALL miss.
  G  Sandar         -> the CONTROL for D: two SUPPORTED, independent derivations of
                       a true claim — convergence here is legitimately earned.

Mirrors seed_news.py's ingestion conventions: real event_date per node (so
coverage / time-window / convergence reason over dates) and an optional source
weight (only persisted when ENCELADUS_SOURCE_WEIGHTS=1).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import db
import ingestion
import seed_large  # reuse reset()
import seed_roots
import sources
from ingestion import ingest_text, run_inference_batch

try:  # real source names can carry non-cp1252 chars; keep Windows stdout safe
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# All events live in one recent month so every within-scenario pair sits inside
# the 30-day coverage / time-window. `day` is an offset (in days) from BASE.
BASE = datetime(2026, 6, 2, 9, 0, 0, tzinfo=timezone.utc)


def _iso(day_offset: int, hour: int = 9) -> str:
    return (BASE + timedelta(days=day_offset, hours=hour - 9)).strftime("%Y-%m-%dT%H:%M:%SZ")


# Each event: (text, source, day_offset, role). `role` is documentation only —
# it records what the event is *for* so the corpus is auditable. The engine never
# sees it; it only sees text / source / date.
#
#   premise   one of the two events the target inference should be derived from
#   support   evidence that should be classified supports_inference (corroboration)
#   topical   shares the actors/subject but does NOT establish the causal claim
#             (the Pelora trap — present to see if the classifier rewards overlap)
#   defeater  evidence that should be classified supports_alternative (contradicts)
#   context   neutral density (affects coverage; not support or defeater)

SCENARIOS: dict[str, list[tuple[str, str, int, str]]] = {
    # === A — DEFEATER path: the corpus contradicts a derived narrative ========
    "A_veridia_contested": [
        ("Veridia's central bank raised its benchmark interest rate by 50 basis points, the third increase this year, citing stubborn inflation.",
         "Reuters", 0, "premise"),
        ("Veridia's economy expanded 3.2 percent last quarter, beating forecasts, official figures showed.",
         "Bloomberg", 1, "premise"),
        # Direct contradiction of any 'Veridia is in a tightening cycle' conclusion:
        ("Veridia's central bank unexpectedly cut its benchmark rate and declared an end to its tightening cycle, citing recession risk.",
         "Reuters", 8, "defeater"),
        ("Veridia's inflation rate eased to 4.1 percent in May from 4.8 percent in April.",
         "Financial Times", 4, "context"),
        ("Veridia's finance minister said fiscal policy would stay supportive of growth for now.",
         "Bloomberg", 5, "context"),
        ("Economists at a Veridia lender said recent rate moves reflect data-dependence, not a fixed tightening path.",
         "CNBC", 6, "context"),
    ],

    # === B — dense, well-evidenced causal chain: TRUE POSITIVE ================
    "B_marran_corroborated_true": [
        ("Marran banned all exports of rare-earth magnets, citing national security concerns.",
         "Reuters", 0, "premise"),
        ("Global prices for rare-earth magnets surged 42 percent over the past week, traders reported.",
         "Bloomberg", 2, "premise"),
        ("Automakers warned of production cuts as rare-earth magnet supplies tightened following Marran's export ban.",
         "Reuters", 3, "support"),
        ("Electronics manufacturers said Marran's export ban left them scrambling for alternative magnet suppliers.",
         "Financial Times", 3, "support"),
        ("Analysts attributed the spike in magnet prices directly to Marran's sudden export halt.",
         "CNBC", 4, "support"),
        ("Marran controls roughly 80 percent of global rare-earth magnet production, industry data shows.",
         "The Economist", 1, "context"),
        ("Wind-turbine makers reported rising costs after the Marran export restrictions.",
         "Bloomberg", 5, "support"),
        ("Several governments convened emergency talks over the rare-earth supply shock triggered by Marran.",
         "Associated Press", 5, "support"),
        ("Marran's commerce ministry confirmed the export ban would remain in force until further notice.",
         "Reuters", 6, "context"),
        ("Defense contractors flagged supply risk for magnet-dependent systems after the Marran ban.",
         "The Wall Street Journal", 6, "support"),
        ("Recyclers reported a jump in demand for reclaimed rare-earth magnets amid the Marran-driven shortage.",
         "Reuters", 7, "support"),
        ("Commodity indices tracking rare-earth materials hit record highs this week.",
         "Financial Times", 7, "context"),
    ],

    # === C — TOPICAL-vs-CAUSAL trap: dense overlap, NO causal evidence ========
    # The two premises co-move; the surrounding evidence points to a COMMON cause
    # (capital flight / election uncertainty), not currency -> stocks causation.
    # Correct truth for "the dram's fall caused the stock drop" is false/unverifiable.
    "C_pelora_topical_trap": [
        ("Pelora's currency, the dram, fell 3.1 percent against the dollar on Tuesday.",
         "Reuters", 0, "premise"),
        ("Pelora's benchmark stock index dropped 2.4 percent on Tuesday.",
         "Bloomberg", 0, "premise"),
        ("Foreign investors withdrew funds from Pelora's equity market this quarter, data showed.",
         "Financial Times", 1, "topical"),
        ("Pelora's government bond yields climbed amid fiscal-deficit concerns.",
         "Bloomberg", 1, "topical"),
        ("Pelora's dram has been among the region's most volatile currencies this month.",
         "CNBC", 2, "topical"),
        ("Analysts cited election uncertainty as weighing broadly on Pelora's financial markets.",
         "Reuters", 2, "topical"),
        ("Pelora's central bank intervened to slow the dram's slide.",
         "Bloomberg", 3, "topical"),
        ("Trading volume on Pelora's stock exchange rose as investors repositioned.",
         "CNBC", 3, "topical"),
        ("Pelora's finance ministry downplayed the market moves as short-term noise.",
         "Reuters", 4, "topical"),
        ("Credit-rating agencies left Pelora's outlook unchanged despite the turbulence.",
         "Financial Times", 4, "context"),
        ("Pelora's dram recovered slightly on Wednesday as the dollar eased.",
         "Bloomberg", 5, "topical"),
    ],

    # === D — CONVERGENCE-OVER-CAP bug repro: two unsupported, independent ======
    # derivations of the same claim. No node states the conclusion (-> support 0,
    # unverified, capped 0.55). No node contradicts it (-> defeater 0). The two
    # pairs share NO raw grounding, so convergence fires and the bonus lifts
    # confidence above the 0.55 cap. EXPECT: status 'unverified', conf ~0.65-0.70,
    # converged >= 1. After the status-gated fix, conf should stay at 0.55.
    "D_khelas_convergence_bug": [
        # pair 1 (army)
        ("Khelas cancelled all leave for army personnel and recalled reservists this week.",
         "Reuters", 0, "premise"),
        ("Khelas moved armored units toward its northern frontier, satellite images suggest.",
         "Associated Press", 1, "premise"),
        # pair 2 (navy + air force — disjoint raw grounding from pair 1)
        ("Khelas ordered its naval fleet back to home ports without explanation.",
         "Reuters", 2, "premise"),
        ("Khelas's air force carried out unannounced large-scale readiness drills.",
         "Bloomberg", 3, "premise"),
        # neutral — explicitly does NOT corroborate an 'escalation' conclusion
        ("Khelas's defense ministry declined to comment on recent troop movements.",
         "Reuters", 3, "context"),
    ],

    # === E — thin + speculative: TRUE NEGATIVE / unverifiable =================
    "E_tovar_unverifiable": [
        ("Analysts speculate Tovar's incoming prime minister could steer the country toward the Eastern Bloc.",
         "CNBC", 0, "premise"),
        ("Tovar's new cabinet held its first formal meeting on Monday.",
         "Reuters", 1, "premise"),
        ("Tovar's prime minister pledged to focus first on domestic economic reform.",
         "Reuters", 2, "context"),
    ],

    # === F — warranted & TRUE but uncorroborated: RECALL miss =================
    # Both sides ratified -> the deal clearing its final hurdle is well-warranted
    # and true, but there is no separate 'entered into force' node to act as
    # support. EXPECT: likely 'unverified' despite truth=true -> a false negative.
    "F_anvaria_recall_miss": [
        ("Anvaria's parliament ratified the free-trade agreement with the Coastal Union by a wide margin.",
         "Reuters", 0, "premise"),
        ("The Coastal Union's council formally approved the same free-trade agreement with Anvaria.",
         "Bloomberg", 1, "premise"),
        ("Business groups in Anvaria welcomed the trade deal's progress.",
         "Financial Times", 2, "context"),
    ],

    # === G — CONTROL for D: two SUPPORTED independent derivations of a true =====
    # claim. Convergence here is legitimately earned (both inferences are
    # themselves corroborated). EXPECT: both 'corroborated', converged >= 1, true.
    "G_sandar_corroborated_convergence": [
        # pair 1
        ("Sandar imposed rolling blackouts across its capital this week.",
         "Reuters", 0, "premise"),
        ("Sandar's largest power station went offline for emergency repairs.",
         "Bloomberg", 1, "premise"),
        # pair 2 (disjoint grounding)
        ("Sandar's government urged industrial users to cut power consumption by 30 percent.",
         "Financial Times", 2, "premise"),
        ("Sandar began importing emergency electricity from a neighboring country.",
         "Associated Press", 2, "premise"),
        # supporting density (entail the shortage)
        ("Sandar's energy minister warned the power crunch could last for weeks.",
         "Reuters", 3, "support"),
        ("Hospitals across Sandar switched to backup generators amid the outages.",
         "BBC News", 3, "support"),
        ("Sandar's grid operator reported demand far exceeding available supply.",
         "Bloomberg", 4, "support"),
        ("Businesses in Sandar reported losses from the prolonged power cuts.",
         "CNBC", 4, "context"),
        ("Sandar's reservoirs hit record lows, cutting hydroelectric output.",
         "Reuters", 5, "support"),
        ("Residents in Sandar queued for fuel to run private generators.",
         "Associated Press", 5, "context"),
        ("Sandar's opposition blamed years of underinvestment in the grid.",
         "The Guardian", 6, "context"),
        ("Sandar declared a state of energy emergency on Friday.",
         "Reuters", 6, "support"),
    ],
}


def _ingest_all() -> int:
    n = 0
    for key, events in SCENARIOS.items():
        print(f"\n--- {key}  ({len(events)} events) ---")
        for text, source, day, role in events:
            weight = sources.weight_for(source) if ingestion.USE_SOURCE_WEIGHTS else None
            try:
                node_id = ingest_text(
                    text,
                    source_url=f"https://example.com/{key}/{n}",
                    event_date=_iso(day),
                    source_weight=weight,
                )
                n += 1
                print(f"  [{role:>8}] {node_id}  {text[:62]}")
            except Exception as exc:
                print(f"  [{role:>8}] FAILED: {exc}  ({text[:44]})")
    return n


def main() -> None:
    if "--reset" in sys.argv:
        seed_large.reset()
    else:
        print("NOTE: running without --reset; a clean diagnostic read needs an "
              "empty DB. Re-run with --reset for interpretable numbers.\n")

    print("seeding topic roots ...")
    seed_roots.seed_roots()

    total = sum(len(v) for v in SCENARIOS.values())
    print(f"ingesting {total} engineered events across {len(SCENARIOS)} scenarios ...")
    ingested = _ingest_all()

    print("\nflushing inference tail ...")
    print(run_inference_batch(force=True))

    c = db.client()
    nodes = c.table("nodes").select("id", count="exact").execute().count
    raw = c.table("nodes").select("id", count="exact").eq("node_category", "raw_input").execute().count
    inf = c.table("nodes").select("id", count="exact").eq("node_category", "inference").execute().count
    entities = c.table("entities").select("id", count="exact").execute().count
    edges = c.table("edges").select("id", count="exact").execute().count

    print(
        f"\ndone — ingested {ingested}/{total} | entities: {entities} | "
        f"nodes: {nodes} (raw: {raw}, inference: {inf}) | edges: {edges}"
    )
    print("\nNext: open /dashboard (or `python eval/label_dump.py`) and compare each")
    print("inference's status/confidence against EVAL_SEED_NOTES.md. Then label and")
    print("run `python eval/run_eval.py --compare`.")


if __name__ == "__main__":
    main()
