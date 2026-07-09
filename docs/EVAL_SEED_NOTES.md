# Diagnostic seed — what each scenario tests and the correct behavior

`seed_eval.py` is not a realistic news dump. It is a controlled benchmark: seven
entity-disjoint scenarios where the *correct* verification outcome is known in
advance. Three of them deliberately set up a **divergence** between the system's
likely `status` and the **warranted truth** — those are what make precision and
recall measurable. (Your previous label set was ~14 true / 2 false, so precision
rested on two examples and recall was effectively unmeasurable.)

Run: `python seeds/seed_eval.py --reset`, then read `/dashboard` (or
`python eval/label_dump.py`) against the table below.

Two distinct things are being checked per scenario:
- **status** — what the pipeline outputs (`corroborated` / `contested` / `unverified`).
- **truth** — what a human should label it, judging whether the conclusion is
  warranted by its premises and evidence (the same standard `judge_inference` uses).

A correct verifier makes `status == corroborated` line up with `truth == true`.
Where they're designed to diverge, that gap is the signal.

---

## A — Veridia (defeater path)

Designed inference: *"Veridia is in a monetary-tightening cycle"* (from three rate
hikes + strong growth).
The corpus then contains a direct reversal: the bank **cuts** and declares the
tightening cycle over.

- **Expected status:** `contested` — the cut should classify as `supports_alternative` (a defeater).
- **Expected truth:** `false` or `unverifiable` (the tightening narrative is overtaken).
- **Diagnoses:** whether the adversarial half works *at all*. In your last sample,
  defeaters were `0` on every card. If this one shows `defeaters 0` and lands
  `corroborated`/`unverified`, the defeater path is the thing to fix — either
  retrieval isn't surfacing the contradicting node, or the classifier won't emit
  `supports_alternative`.

## B — Marran (true positive)

Designed inference: *"Marran's export ban drove the rare-earth magnet price surge."*
Dense corpus (~12 nodes), multiple sources explicitly attributing the spike to the ban.

- **Expected status:** `corroborated`, high confidence (coverage near saturation).
- **Expected truth:** `true`.
- **Diagnoses:** the happy path. If this *isn't* corroborated, coverage or the
  support classifier is too conservative and your recall problem is upstream of
  the trickier cases.

## C — Pelora (topical-vs-causal trap)  ← key precision probe

Designed inference: *"The dram's fall caused Pelora's stock index to drop."*
The two move together on the same day, and the corpus is **dense with nodes that
share the actors/subject** — but none establish the *causal direction*. The
evidence actually points to a **common cause** (capital flight, election
uncertainty), which would explain both without one causing the other.

- **Expected status if the classifier is sound:** `unverified` (no evidence for the
  specific causal claim).
- **Likely actual status:** `corroborated`, because the support classifier rewards
  topical overlap.
- **Expected truth:** `false` / `unverifiable`.
- **Diagnoses:** this is the "what is `support N` actually counting?" question from
  the review, made measurable. If C lands `corroborated`, it is a **false positive**
  and your precision number now has a real negative to move it. Pull the support
  node ids for this inference and check: do any assert the dram *caused* the stock
  move, or do they just co-mention Pelora's markets?

## D — Khelas (convergence-over-cap bug)  ← bug reproduction

Two **independent** derivations of *"Khelas is preparing a major military
escalation"*: one from army signals, one from navy + air-force signals, sharing no
raw grounding. **No node states or corroborates the escalation conclusion**
(support should be `0`), and none contradicts it (defeaters `0`).

- **Expected status:** `unverified`.
- **The bug:** with `support 0`, `_verify_inference` caps confidence at
  `UNVERIFIED_CONFIDENCE_CAP` (0.55). But `_persist_inference` then adds
  `CONVERGENCE_BONUS` (0.15) *after* the cap and only re-ceilings against coverage —
  so confidence climbs to ~0.65–0.70 on an unsupported claim.
- **Watch for:** `status = unverified`, `support 0`, `converged ≥ 1`, **confidence > 0.55**.
  That is the same shape as the `0.90 → 0.70` card in your sample.
- **Expected truth:** `unverifiable` (two independent *hints* of preparation don't
  evidence the escalation itself).
- **After the fix** (gate the bonus on the converging node's status, or apply it
  before the cap): confidence should stay pinned at 0.55. This scenario is your
  regression test for that fix.

## E — Tovar (true negative / unverifiable)

Designed inference: *"Tovar's foreign policy will shift toward the Eastern Bloc."*
Pure speculation, thin corpus, nothing for or against.

- **Expected status:** `unverified`, low confidence (~0.40, the coverage floor).
- **Expected truth:** `unverifiable`.
- **Diagnoses:** that thin + speculative correctly stays low and doesn't get
  swept up by anything.

## F — Anvaria / Coastal Union (recall miss)  ← key recall probe

Designed inference: *"The Anvaria–Coastal-Union trade deal has cleared its final
ratification hurdle."* Both sides ratified — this is **well-warranted and true** —
but there is no separate "deal entered into force" node to serve as corroborating
support, and coverage is thin.

- **Expected status:** likely `unverified` (no support node, thin coverage).
- **Expected truth:** `true`.
- **Diagnoses:** a **false negative** — true but not corroborated. This is what gives
  recall something to measure. If you want corroborated-vs-true to have teeth, you
  need rows like this where the truth is `true` but the evidence is sparse.

## G — Sandar (control for D)

Two **independent, supported** derivations of *"Sandar is facing a severe energy
shortage"* — one from blackouts + plant outage, one from the rationing order +
emergency imports — over a dense, consistent corpus.

- **Expected status:** both `corroborated`, `converged ≥ 1`.
- **Expected truth:** `true`.
- **Diagnoses:** the contrast that proves D is a *bug* and not convergence being
  broken in general. Here both converging inferences are themselves corroborated,
  so the bonus is legitimately earned. D and G differ only in whether the
  converging derivations have evidentiary support — which is exactly the
  distinction the fix should encode.

## H — Cael (specific-value overreach)  ← soundness probe

Designed inference: *"Cael's banking crisis began between June 20 and 22"* — an
interval bound invented from two **effect** reports (a June-20 market plunge, a
June-22 emergency meeting). Effects only *upper*-bound a start; the premises say
nothing about when the crisis began. The corpus also contains the real, earlier
start ("first erupted in late May"), and the supports confirm the crisis EXISTS
without dating it.

- **Expected status without the soundness gate:** likely `corroborated` (topical
  supports + no defeater on the *is-it-real* axis) — a **false positive**, the
  live Iran-war-dating failure reproduced.
- **With `ENCELADUS_SOUNDNESS_GATE=1`:** the premise-only check should flag the
  interval as not-following-from-premises and demote to `unverified`.
- **Expected truth:** `false`.
- **Diagnoses:** whether a conclusion can be caught as *logically unsound from its
  own premises*, independent of world knowledge or corpus retrieval. Note the LLM
  reasoner may instead form a sound inference here (e.g. "Cael faces a banking
  crisis"); judge whichever inference the run actually produces — the target is any
  date/interval-overreach card.

---

## Expectation matrix (the planted divergences)

| Scenario | Expected status        | Expected truth | This row measures        |
|----------|------------------------|----------------|--------------------------|
| A        | contested              | false/unverif. | defeater path fires      |
| B        | corroborated           | true           | true positive            |
| C        | corroborated (likely)  | false/unverif. | **precision** (FP)       |
| D        | unverified, conf >0.55 | unverifiable   | convergence-cap **bug**  |
| E        | unverified             | unverifiable   | true negative            |
| F        | unverified (likely)    | true           | **recall** (FN)          |
| G        | corroborated           | true           | convergence done right   |
| H        | corroborated w/o gate  | false          | **soundness** (overreach)|

## Reading the eval output

After labeling (`eval/label_dump.py` → fill `truth` → `eval/run_eval.py`):

- **C** should appear in the confusion matrix as `predicted=corroborated, truth=false`
  — a false positive that drags precision below 1.0. If precision stays 1.0, either
  C wasn't corroborated (good — your classifier is causally careful) or it wasn't
  labeled `false` (check the label).
- **F** should appear as `predicted=unverified, truth=true` — a false negative that
  pulls recall down.
- **D's** confidence belongs in the calibration table: an `unverified` claim sitting
  at ~0.70 is the calibration error the convergence bug introduces. Re-run after the
  fix; ECE in that bin should improve.

## Caveats

- These are *targets*. The engine pairs by similarity and an LLM does the reasoning,
  so it may also form adjacent inferences (e.g. B might separately conclude "Marran
  dominates the magnet market"). Judge each inference the dashboard actually
  produces; the table tells you which one per scenario is the one to watch.
- Scenarios are entity-disjoint and noise-free on purpose, so behavior is
  interpretable. Once you trust the diagnostics here, re-run them against the real
  `news_snapshot.jsonl` corpus, where noisy entities and aggregator churn will
  stress retrieval and entity resolution in ways this clean set deliberately doesn't.
- If you turn on `ENCELADUS_SOURCE_WEIGHTS=1`, A/C/D's lower-weight sources won't
  matter much (they're mostly wires here); the weighting effect shows up most in B
  and G, where high-reliability wires dominate the support set.
