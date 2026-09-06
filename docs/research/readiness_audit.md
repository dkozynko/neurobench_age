# Scientific experiment readiness audit

Date: 2026-09-05

## Verdict

The repository is code-ready for the real-data engineering pilot and HBN
frozen-representation run. It is not yet evidence-complete: no real MIPDB
primary subject has been evaluated, the 40 real head runs do not yet exist, and
the external study has not been sealed or started.

The prospective claim therefore remains undecided. Existing HBN R5 results are
retrospective secondary evidence only.

## Implemented controls

- One strict executable protocol fixes the encoder, layers, four heads, ten
  seeds, preprocessing, checkpoint selection, inferential procedure, and
  decision threshold.
- The MIPDB inventory is model-free and content-addressed. HBN age support is derived from HBN
  training subjects in the canonical manifest rather than supplied as a free
  runtime override.
- The ten-subject MIPDB pilot is deterministic, permanently disjoint from the
  external cohorts, target-free in its output, and create-only.
- A second target-free pass applies the frozen signal QC contract to every
  non-pilot candidate and publishes the finalized cohort before sealing.
- MIPDB `task-block01` parsing requires marker `90`, alternating `20`/`30`
  condition boundaries, exact `E1`--`E128` EEG channels, no mapped bad channel,
  no interpolation, and no cross-condition window.
- HBN representation materialization excludes R5 before resolving signal
  paths, rejects duplicate subjects, uses one global channel order, and caches
  only layers `-2` and `-1` from a verified frozen encoder.
- Head fitting receives cached train/validation representations only. Its CLI
  exposes no head, seed, or test-split override and requires the exact 40-run
  matrix.
- The external runner validates the sealed lock and complete checkpoint
  inventory, writes `evaluation_started.json`, and only then invokes lazy MIPDB
  signal loading and REVE extraction. Existing predictions are immutable.
- Confirmatory analysis requires the complete matched subject/seed inventory
  and applies the predeclared hierarchical bootstrap, exact randomization test,
  Holm correction, and joint stability rule.
- Adequate primary cohort size is part of that joint rule; underpowered results
  remain descriptive and cannot populate `established_heads`.
- Production sealing derives every lock field from verified artifacts and
  cross-checks raw acquisition hashes, pilot/final QC, HBN training evidence,
  and all 40 run manifests/checkpoints.
- Runtime outputs, datasets, checkpoints, package-build directories, and caches
  are ignored or required outside the versioned canonical evidence tree.

## Verification evidence

The following checks were run from the current uncommitted source tree:

- Complete default-environment suite: 565 tests passed before the final audit
  fixes.
- Current default environment: 578 tests passed, including a complete synthetic
  artifact-derived sealed workflow with 40 miniature runs and 2,000 immutable
  predictions.
- Clean package build: source distribution and wheel built successfully.
- Lock consistency: `uv lock --check` resolved 115 packages without drift.
- Python byte-compilation: all files under `src`, `scripts`, and `tests`
  compiled successfully.
- Shell syntax: both shell launchers passed `bash -n`.
- Diff hygiene: `git diff --check HEAD` reported no whitespace errors.
- Secret scan: no Hugging Face token, cloud access-key pattern, or private-key
  header was found in the worktree (excluding the dependency lock).
- Size audit: the largest tracked blob is approximately 171 KB; the Git pack is
  under 1 MB. The approximately 572 KB `uv.lock` is expected dependency
  metadata, not experiment data or a model artifact.

PyTorch needs shared-memory access on this macOS host. A sandboxed direct import
failed with an OpenMP shared-memory error; the same isolated environment passed
the full suite outside that sandbox. This is a host sandbox constraint, not a
repository failure, but GPU execution should still begin with the pilot smoke
test on the actual compute host.

## Required execution sequence

1. Mount HBN and the pinned MIPDB/NEMAR snapshot outside Git.
2. Build the content-addressed MIPDB draft inventory from the canonical HBN manifest.
3. Run only the ten-subject MIPDB pilot and manually inspect channel, event,
   duration, window, and representation-shape QC.
4. If the pilot fails, diagnose the adapter without viewing or retaining an age
   prediction metric. A protocol change requires a new protocol identity.
5. Run target-free QC over all remaining candidates and publish the finalized
   MIPDB cohort and power status.
6. Materialize HBN train/validation representations; confirm R5 was untouched.
7. Train the four heads for seeds 33--42 and audit the exact 40 checkpoints.
8. From a clean worktree, run artifact-derived sealing over the environment,
   acquisition manifests, QC reports, subject lists, encoder state, and runs.
9. Seal the study. Do not modify code, dependencies, cohorts, thresholds, or
   checkpoints afterward.
10. Launch the external runner once. Resume only after an infrastructure failure
   and only against the same lock and immutable artifacts.
11. Run aggregate confirmatory analysis only after the complete prediction
    inventory and completion marker exist.

Exact commands and path variables are maintained in
`docs/research/article_ready_protocol.md`.

## Remaining evidence, not code, blockers

- The exact MIPDB release snapshot and real channel/event compatibility have
  not been verified by the production pilot.
- Real REVE encoder state hash and HBN cache identities have not been recorded.
- The exact 40 real head checkpoints have not been produced.
- Effective MIPDB primary cohort size and the underpowered flag are unknown.
- Primary MIPDB inference has not started; no prospective external metric or
  superiority conclusion exists.
