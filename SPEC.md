# Codex Task Specs — Dynamic Persistence Diagnostic Pipeline

**Goal:** Figure 1 for the residency application within ~1 week: bare-model persistence curve on paired intervention trajectories, teacher-forced protocol, Oasis-500M.

**Critical path:** T0 (schema) → T1 (recorder) + T2 (model runner, parallel) → T3 (readout) → T4 (figure). T5 is post-submission.

**Working agreement with Codex (paste into every session):**
- Build against the data contracts in this doc; never invent schema fields.
- Mock-first: every GPU-dependent component gets a `--mock` mode (returns noise frames) so the full pipeline is testable on CPU before renting a GPU.
- Small tasks, each with a runnable acceptance test. No task is "done" until its acceptance command passes.
- Do not make experiment-design decisions (which metrics, which scenarios, thresholds). Flag ambiguities; a human decides.

---

## Repo layout

```
persistence-probe/
  configs/            # model constants, scenario params (yaml)
  schema/             # episode schema + validator
  recorder/           # T1: MineRL episode recorder
  runner/             # T2: world-model inference wrappers (oasis/, mineworld/)
  readout/            # T3: pixel->state readout + metrics
  figures/            # T4: plotting + qualitative grids
  episodes/           # data (gitignored)
  results/            # generations + metrics (gitignored)
  scripts/            # end-to-end orchestration
  PREREG.md           # metric definitions + directional predictions (human-written, frozen before large runs)
```

---

## T0 — Episode schema + validator (Day 1)

Everything codes against this contract.

```
episodes/{scenario}_{pair_id}_{arm}/     # arm ∈ {intervene, control}
  meta.json      # {scenario, pair_id, arm, N_away, world_seed, start_pos,
                 #  probe_block: {pos, type}, fps, resolution, model_target}
  frames/%06d.png            # native model resolution + fps (from configs/)
  actions.jsonl              # one line per frame, model's native action space
  state.jsonl                # per frame: {probe_block_state, time_of_day,
                             #  player_pos, player_yaw_pitch}
  events.jsonl               # {frame_idx, event: "intervention"|"look_away"|"return_start"}
```

**Spec:** JSON-schema files + `python -m schema.validate <episode_dir>` + a generator for one synthetic dummy episode (random frames, plausible logs).
**Accept:** validator passes on dummy, fails loudly on a corrupted copy.

---

## T1 — MineRL scripted recorder (Days 1–3, highest risk, 2-day timebox)

**Spec:** Headless MineRL (xvfb) controller executing parameterized probe scripts and logging per the schema:

1. Fixed `world_seed`, superflat/plains start; place a high-contrast probe block (e.g., gold) at a fixed offset. Verify frames look in-distribution for Minecraft gameplay (HUD present, normal FOV).
2. Script: `approach_and_observe(K1 frames)` → `intervene()` (break probe block) **or no-op** → `look_away(N frames)` (rotate 180° + walk) → `return_and_observe(K2 frames)`.
3. **Paired counterfactual:** intervene/control arms share seed, spawn, and the exact action sequence except the intervention frames. Assert action-log equality outside the intervention window.
4. CLI: `python -m recorder.record --scenario break_gold --N 8 16 32 64 128 256 --pairs 20`
5. Log ground truth from the env handler/observation dict every frame (block state at probe coords, time_of_day, player pose).

**Accept:** 2 pairs at N=16 recorded end-to-end; validator passes; a contact-sheet script renders key frames (pre-intervention, intervention, mid-away, return) for human eyeballing.
**Gotchas for Codex:** deterministic playback (fixed seeds everywhere, no async input); frame rate must match the model config (resample if the env ticks differently); resolution letterboxing per model config; xvfb + Java flags for headless.
**Timebox rule:** if MineRL install/determinism burns >2 days, stop and escalate (fallbacks: Malmo directly, or pin to a known-good MineRL docker image). Do not silently switch to a non-Minecraft renderer — domain match with the model's training data is a hard requirement.

---

## T2 — Model runner with context control (Days 2–4, parallel with T1)

**Spec:** Wrapper over open-oasis (oasis-500m) inference:

1. First subtask: **extract model constants** (native resolution, fps, max context frames, action encoding) from the repo/weights into `configs/oasis.yaml`. Do not hard-code guessed numbers.
2. Interface: `generate(context_frames, context_actions, future_actions, seed, ctx_limit=None) -> frames`. `ctx_limit` truncates visible context (for the context-boundary sweep).
3. **Teacher-forced protocol:** context = real frames+actions up to `return_start`; generate K2 frames using the recorded return actions.
4. Batch orchestrator: `python -m runner.run --model oasis --episodes episodes/ --protocol teacher_forced` — resumable, caches to `results/{episode}/{model}/gen_%06d.png`, records seeds + config hash.
5. `--mock` mode for CPU testing.

**Accept:** end-to-end on the dummy episode with `--mock` on CPU; deterministic outputs under fixed seed with the real model on GPU.
**Second model (MineWorld) is a copy of this interface — defer until after Figure 1.**

---

## T3 — Readout + metrics (Days 3–5)

**Spec:**

1. **Probe-region localization:** under teacher forcing the return camera pose is forced, so the probe block's screen region in generated frames matches its region in the *real* return frames of the same episode. Compute the crop box from the real frames (template match the probe block in the control arm), reuse it for generated frames.
2. **State classifier:** decide "block present vs. absent" in the crop via template/SSIM match against reference crops harvested from real frames (labels are free from `state.jsonl`). Threshold chosen on real frames only; report accuracy on held-out real frames (must be >95% before touching generated frames).
3. **Per-episode outcome:** majority vote over the K2 return frames → `correct_state ∈ {0,1}` vs. engine ground truth.
4. **Metrics table + JSON:** P(correct | intervene) and P(correct | control) per N; paired Δ; stale-resurrection rate (intervene arm rendered as intact); auxiliary LPIPS/DINO similarity of generated-vs-real return frames.
5. **Human QA hook:** `python -m readout.audit --sample 50` dumps crops + classifier decisions to an HTML sheet for manual verification.

**Accept:** full metric run on mock generations completes; audit sheet renders.

---

## T4 — Figure 1 + prelim report (Days 5–7)

**Spec:**

1. Main plot: x = N (log scale), y = P(correct state); two curves (intervene, control); vertical line at the model's context boundary; error bars via binomial CI over pairs. Paper-style matplotlib (single column, no chartjunk).
2. Qualitative grid: rows = N values, cols = [real return | generated (control) | generated (intervene)], probe region boxed.
3. Side-by-side demo clip builder (real vs. generated return sequence) for the strongest failure case.
4. `PRELIM.md` auto-fills a results table; a human writes the 1-page narrative around it.

**Accept:** one command regenerates figure + grid from `results/`.

---

## T5 — Post-submission (do NOT start before the application is out)

- Arm 2 (retrieval): re-inject the *pre-intervention* real frames of the revisited location as extra context on return → measures negative transfer.
- Arm 3 (tracker): same injection, but frames procedurally edited to the tracked post-intervention state.
- Autoregressive-absence protocol + relocalization gating (SuperPoint+LightGlue match count) + context sweep.
- Linear probes on cached activations ("is the block still present?").

---

## Exemplary Codex prompt (T1 — copy this level of specificity for every task)

> You are working in the repo `persistence-probe` (structure and data contracts in SPEC.md — read it first; never deviate from the episode schema).
> Implement `recorder/` per T1: a headless MineRL-based recorder producing paired intervention/control episodes.
> Requirements: (1) fixed world seed + spawn, superflat, place a gold block at a fixed offset before recording starts; (2) scripted phases approach → [break block | no-op] → rotate-and-walk-away for N frames → return along the reverse path → observe K2=32 frames; (3) intervene/control arms must have byte-identical action logs outside the intervention window — add an assertion; (4) log frames (per `configs/oasis.yaml` resolution+fps), actions, and per-frame ground truth (probe block state, time_of_day, player pose) per schema; (5) CLI `record --scenario break_gold --N 8 16 32 64 128 256 --pairs 20`; (6) headless via xvfb; pin all versions in `recorder/requirements.txt`.
> Acceptance: record 2 pairs at N=16; `python -m schema.validate` passes on all four episode dirs; `python -m recorder.contact_sheet <dir>` renders key frames. Write a smoke test that runs a 1-pair N=8 recording in CI-mock mode (env stubbed).
> Do not decide scenario parameters beyond the above; if MineRL's API forces a deviation from the schema, stop and list options instead of improvising.