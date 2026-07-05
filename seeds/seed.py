"""Seed the knowledge graph with sample headlines.

    python seed.py            # ingest the samples on top of existing data
    python seed.py --reset    # wipe entities/nodes/edges/topics first, then ingest

Samples are clustered across geopolitical theatres with deliberate domain
overlaps (economic + military, diplomatic + energy, etc.) so the expanded
inference pool — entity-neighbors + domain-neighbors — has cross-domain
material to reason over.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path

import db
from ingestion import ingest_text

SAMPLES: list[tuple[str, str]] = [
    # --- Iran nuclear + sanctions cluster ---
    ("Iran announced it will resume nuclear talks with European powers in Geneva next month.",
     "https://example.com/iran-talks"),
    ("Iran's foreign minister said the country's nuclear program is entirely peaceful and denied any weapons ambitions.",
     "https://example.com/iran-denial"),
    ("The United States warned it would impose fresh sanctions on Iran if the nuclear negotiations fail.",
     "https://example.com/us-iran-sanctions"),
    ("Israel said it would not rule out military action against Iran's nuclear facilities.",
     "https://example.com/israel-iran"),
    ("The IAEA reported that Iran has increased its stockpile of enriched uranium beyond agreed limits.",
     "https://example.com/iaea-iran"),
    ("The European Union imposed new sanctions targeting Iranian oil exports following the IAEA report.",
     "https://example.com/eu-iran-oil-sanctions"),
    ("Iran announced it is increasing oil exports to China to offset Western sanctions revenue losses.",
     "https://example.com/iran-china-oil"),
    ("Iran deployed additional naval vessels to the Strait of Hormuz amid rising tensions with the US.",
     "https://example.com/iran-hormuz"),

    # --- Russia / Ukraine + energy cluster ---
    ("Russia announced a partial withdrawal of troops from the eastern front near Kharkiv.",
     "https://example.com/russia-withdrawal"),
    ("Ukraine's military denied that Russian forces had withdrawn, calling it a tactical repositioning.",
     "https://example.com/ukraine-denial"),
    ("The European Union pledged an additional 5 billion euros in military aid to Ukraine.",
     "https://example.com/eu-ukraine-aid"),
    ("Russia halted natural gas supplies to three EU member states citing unpaid balances.",
     "https://example.com/russia-gas-cutoff"),
    ("Germany announced it would fast-track construction of two LNG import terminals to replace Russian gas.",
     "https://example.com/germany-lng"),
    ("Poland reported a sharp increase in Russian cyber attacks targeting its energy grid infrastructure.",
     "https://example.com/poland-cyberattack"),
    ("North Korea shipped artillery shells and ballistic missiles to Russia for use in Ukraine, according to US intelligence.",
     "https://example.com/north-korea-russia-arms"),

    # --- China / Taiwan + semiconductor / economic cluster ---
    ("China conducted large-scale military drills around Taiwan following a foreign diplomatic visit.",
     "https://example.com/china-drills"),
    ("Taiwan's defense ministry said it detected 20 Chinese aircraft crossing the median line.",
     "https://example.com/taiwan-aircraft"),
    ("The United States reaffirmed its commitment to Taiwan's defense amid rising regional tensions.",
     "https://example.com/us-taiwan"),
    ("China imposed export restrictions on gallium and germanium, critical materials for semiconductor manufacturing.",
     "https://example.com/china-semiconductor-materials"),
    ("The United States expanded chip export controls to China, citing national security concerns over AI chip access.",
     "https://example.com/us-chip-controls"),
    ("Apple announced it is accelerating supply chain diversification away from China, shifting production to India and Vietnam.",
     "https://example.com/apple-supply-chain"),
    ("Taiwan Semiconductor Manufacturing Company reported that demand for advanced chips remains strong despite geopolitical risks.",
     "https://example.com/tsmc-demand"),

    # --- Cross-domain events (economic + military / diplomatic + energy) ---
    ("Russia reduced oil exports through the Druzhba pipeline after Ukrainian drone strikes damaged pumping stations.",
     "https://example.com/russia-oil-drones"),
    ("Saudi Arabia announced a unilateral oil production cut of one million barrels per day amid tensions with the United States over Yemen.",
     "https://example.com/saudi-oil-cut"),
    ("Iran-backed Houthi forces in Yemen attacked a Saudi Aramco oil facility, causing a temporary production shutdown.",
     "https://example.com/houthi-aramco"),
    ("The United States deployed an aircraft carrier strike group to the Persian Gulf in response to Iranian threats to oil shipping.",
     "https://example.com/us-carrier-gulf"),
    ("The European Union's carbon border adjustment mechanism is expected to reduce Russian fossil fuel export revenues by 12 percent.",
     "https://example.com/eu-carbon-russia"),
    ("China and Russia signed a 30-year natural gas supply agreement, deepening economic ties as Western sanctions on Russia intensify.",
     "https://example.com/china-russia-gas"),
    ("Israel struck Hezbollah weapons depots in Lebanon, with analysts warning of potential disruption to Mediterranean shipping lanes.",
     "https://example.com/israel-hezbollah-shipping"),
    ("The World Food Programme warned that the conflict in Sudan is pushing 8 million people toward famine, with supply routes blocked by armed groups.",
     "https://example.com/sudan-famine"),
]


def reset() -> None:
    """Delete all rows. Cascade order: edges → node_topics → node_entities → nodes → entities → topics."""
    nil = "00000000-0000-0000-0000-000000000000"
    c = db.client()
    c.table("edges").delete().neq("id", nil).execute()
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
            print(f"[{i}/{len(SAMPLES)}] {node_id}  {text[:70]}")
        except Exception as exc:
            print(f"[{i}/{len(SAMPLES)}] FAILED: {exc}  ({text[:50]})")

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
