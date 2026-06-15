"""Large, diverse seed for exercising corroboration + convergence.

Unlike seed.py (small, for smoke testing), this corpus is built to be DENSE
within thematic clusters so coverage clears the corroboration floor (>= 0.5),
and to carry parallel/independent evidence for the same conclusions so the
convergence detector (independence-guarded) can fire.

    python seed_large.py            # ingest on top of existing data
    python seed_large.py --reset    # wipe everything first, then ingest

Clusters (deliberately overlapping via oil/energy/sanctions):
  A. Iran nuclear / Persian Gulf
  B. Russia-Ukraine war / European energy
  C. China-Taiwan / semiconductors & trade
  D. Global oil markets / OPEC / inflation
  E. Climate, energy transition, supply chains
"""
from __future__ import annotations

import sys

import db
from ingestion import ingest_text, run_inference_batch

SAMPLES: list[tuple[str, str]] = [
    # ---- Cluster A: Iran nuclear / Persian Gulf ----
    ("Iran announced it will resume nuclear talks with European powers in Geneva next month.",
     "https://example.com/a1"),
    ("Iran's foreign minister insisted the country's nuclear program is entirely peaceful and denied any weapons ambitions.",
     "https://example.com/a2"),
    ("The United States warned it would impose fresh sanctions on Iran if the nuclear negotiations collapse.",
     "https://example.com/a3"),
    ("Israel said it would not rule out military action against Iran's nuclear facilities.",
     "https://example.com/a4"),
    ("The IAEA reported that Iran increased its stockpile of enriched uranium beyond agreed limits.",
     "https://example.com/a5"),
    ("Iran began enriching uranium to 60 percent purity at the Fordow facility, the IAEA confirmed.",
     "https://example.com/a6"),
    ("Iran deployed additional naval vessels to the Strait of Hormuz amid rising tensions with Washington.",
     "https://example.com/a7"),
    ("The United States moved an aircraft carrier strike group into the Persian Gulf as a deterrent to Iran.",
     "https://example.com/a8"),
    ("Iran threatened to close the Strait of Hormuz to oil tankers if its exports were blocked.",
     "https://example.com/a9"),
    ("Saudi Arabia warned that an Iranian nuclear weapon would force it to pursue its own nuclear deterrent.",
     "https://example.com/a10"),
    ("European diplomats said the Geneva talks were the last chance to avoid a wider Middle East confrontation.",
     "https://example.com/a11"),

    # ---- Cluster B: Russia-Ukraine war / European energy ----
    ("Russia announced a partial withdrawal of troops from the eastern front near Kharkiv.",
     "https://example.com/b1"),
    ("Ukraine's military denied that Russian forces had withdrawn, calling it a tactical repositioning.",
     "https://example.com/b2"),
    ("The European Union pledged an additional 5 billion euros in military aid to Ukraine.",
     "https://example.com/b3"),
    ("Russia halted natural gas supplies to three EU member states, citing unpaid balances.",
     "https://example.com/b4"),
    ("Germany announced it would fast-track two LNG import terminals to replace Russian gas.",
     "https://example.com/b5"),
    ("Russia rerouted its crude oil exports to India and China after European buyers withdrew.",
     "https://example.com/b6"),
    ("China and Russia signed a 30-year natural gas supply agreement, deepening ties under Western sanctions.",
     "https://example.com/b7"),
    ("Ukraine launched drone strikes that damaged pumping stations on Russia's Druzhba oil pipeline.",
     "https://example.com/b8"),
    ("Russia reduced oil exports through the Druzhba pipeline after the Ukrainian drone strikes.",
     "https://example.com/b9"),
    ("North Korea shipped artillery shells and ballistic missiles to Russia, according to US intelligence.",
     "https://example.com/b10"),
    ("Poland reported a sharp increase in Russian cyber attacks targeting its energy grid infrastructure.",
     "https://example.com/b11"),

    # ---- Cluster C: China-Taiwan / semiconductors & trade ----
    ("China conducted large-scale military drills around Taiwan following a foreign diplomatic visit.",
     "https://example.com/c1"),
    ("Taiwan's defense ministry said it detected 20 Chinese aircraft crossing the median line.",
     "https://example.com/c2"),
    ("The United States reaffirmed its commitment to Taiwan's defense amid rising regional tensions.",
     "https://example.com/c3"),
    ("China imposed export restrictions on gallium and germanium, critical materials for semiconductors.",
     "https://example.com/c4"),
    ("The United States expanded chip export controls on China, citing national security over AI chips.",
     "https://example.com/c5"),
    ("Apple accelerated supply-chain diversification away from China, shifting production to India and Vietnam.",
     "https://example.com/c6"),
    ("Taiwan Semiconductor Manufacturing Company said demand for advanced chips remained strong despite tensions.",
     "https://example.com/c7"),
    ("China summoned the US ambassador to protest the latest semiconductor export controls.",
     "https://example.com/c8"),
    ("Japan and the Netherlands agreed to align with US restrictions on chipmaking equipment exports to China.",
     "https://example.com/c9"),
    ("Taiwan pledged billions in subsidies to keep advanced chip fabrication onshore amid security fears.",
     "https://example.com/c10"),

    # ---- Cluster D: Global oil markets / OPEC / inflation ----
    ("Saudi Arabia announced a unilateral oil production cut of one million barrels per day.",
     "https://example.com/d1"),
    ("OPEC+ extended its output cuts through the end of the year to support crude prices.",
     "https://example.com/d2"),
    ("Iran-backed Houthi forces in Yemen attacked a Saudi Aramco oil facility, halting some production.",
     "https://example.com/d3"),
    ("Global crude oil prices surged above 95 dollars a barrel after the supply disruptions.",
     "https://example.com/d4"),
    ("The US released crude from its strategic petroleum reserve to dampen rising fuel prices.",
     "https://example.com/d5"),
    ("The International Energy Agency warned that oil supply shocks were stoking global inflation.",
     "https://example.com/d6"),
    ("The Federal Reserve signaled it may keep interest rates higher for longer as energy costs lifted inflation.",
     "https://example.com/d7"),
    ("Rising diesel prices pushed European manufacturing costs to a two-year high.",
     "https://example.com/d8"),
    ("India increased discounted crude purchases from Russia to shield its economy from price spikes.",
     "https://example.com/d9"),

    # ---- Cluster E: Climate, energy transition, supply chains ----
    ("The European Union's carbon border adjustment mechanism is expected to cut Russian fossil fuel revenues.",
     "https://example.com/e1"),
    ("China dominated global solar panel manufacturing, supplying over 80 percent of the world's modules.",
     "https://example.com/e2"),
    ("The United States passed subsidies to onshore clean-energy and battery manufacturing away from China.",
     "https://example.com/e3"),
    ("A drought in key mining regions disrupted the global supply of lithium for electric-vehicle batteries.",
     "https://example.com/e4"),
    ("Automakers warned that critical-mineral shortages could delay the electric-vehicle transition.",
     "https://example.com/e5"),
    ("The IEA said record renewable installations were beginning to slow growth in global oil demand.",
     "https://example.com/e6"),
    ("Germany accelerated wind and solar buildout to reduce its dependence on imported fossil fuels.",
     "https://example.com/e7"),
]


def reset() -> None:
    nil = "00000000-0000-0000-0000-000000000000"
    c = db.client()
    c.table("edges").delete().neq("id", nil).execute()
    c.table("inference_meta").delete().neq("node_id", nil).execute()
    c.table("node_topics").delete().neq("node_id", nil).execute()
    c.table("node_entities").delete().neq("node_id", nil).execute()
    c.table("nodes").delete().neq("id", nil).execute()
    c.table("entities").delete().neq("id", nil).execute()
    c.table("topics").delete().neq("id", nil).execute()
    print("reset: cleared all tables")


def main() -> None:
    if "--reset" in sys.argv:
        reset()

    for i, (text, url) in enumerate(SAMPLES, 1):
        try:
            node_id = ingest_text(text, url)
            print(f"[{i}/{len(SAMPLES)}] {node_id}  {text[:66]}")
        except Exception as exc:
            print(f"[{i}/{len(SAMPLES)}] FAILED: {exc}  ({text[:46]})")

    # Flush any remaining unprocessed tail (auto-batches fire during ingestion).
    print("\nflushing inference tail ...")
    print(run_inference_batch(force=True))

    c = db.client()
    nodes    = c.table("nodes").select("id", count="exact").execute().count
    raw      = c.table("nodes").select("id", count="exact").eq("node_category", "raw_input").execute().count
    inf      = c.table("nodes").select("id", count="exact").eq("node_category", "inference").execute().count
    entities = c.table("entities").select("id", count="exact").execute().count
    topics   = c.table("topics").select("id", count="exact").execute().count
    edges    = c.table("edges").select("id", count="exact").execute().count
    print(
        f"\ndone — entities: {entities} | topics: {topics} | "
        f"nodes: {nodes} (raw: {raw}, inference: {inf}) | edges: {edges}"
    )


if __name__ == "__main__":
    main()
