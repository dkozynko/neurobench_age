# Clean-server reproduction runbook

This document describes how to reconstruct the public research environment on
a clean machine. It is a runbook, not a replacement for the scientific
execution contract. The authoritative protocol is
[`article_ready_protocol.md`](article_ready_protocol.md); the current evidence
boundary and hashes are recorded in
[`readiness_audit.md`](readiness_audit.md), and the submission checklist is in
[`submission_readiness.md`](submission_readiness.md).

The repository is intentionally aggregate-only. It contains code, schemas,
configuration, manuscript assets, and compact result summaries. Raw EEG,
participant-level metadata, participant-level predictions, representation
caches, and checkpoints must remain in access-controlled storage.

## What can be reproduced from Git

Without restricted data, a clean checkout can reproduce the public aggregate
artifacts and:

- the Python contract and regression tests;
- schema and provenance validation;
- aggregate analysis and manuscript asset validation;
- the LaTeX manuscript build;
- the interpretation and claim boundary documented by the retained JSON,
  tables, figures, and hash manifests.

The full subject-level computation requires the exact access-controlled inputs
listed below. Aggregate JSON is sufficient to rebuild the public tables and
figures, but it cannot recreate raw signals, subject-level predictions, cached
representations, or selected model checkpoints.

## 1. Start from a clean checkout

Use a new working directory and record the repository revision before running
anything:

```bash
git clone <repository-url> neurobench_age
cd neurobench_age
git rev-parse HEAD
git status --short
```

The clean run should begin with an empty `git status --short` output. Do not
place datasets, credentials, caches, checkpoints, predictions, or logs inside
the repository.

## 2. Create the locked environment

The project requires Python 3.12 or newer. Install `uv` using the operating
system's approved package or bootstrap mechanism, then create the environment
from the committed lock file:

```bash
uv sync --frozen --extra test
```

For the full REVE/data pipeline and manuscript assets, include the optional
groups:

```bash
uv sync --frozen \
  --extra test \
  --extra reve \
  --extra external \
  --extra manuscript
```

The installation step does not download HBN, MIPDB, ds006780, or pretrained
weights. Those inputs are obtained separately under their applicable access and
reuse terms.

## 3. Run the public verification checks

Run the contract and regression suite before adding private inputs:

```bash
uv run --frozen --extra test pytest -q
```

The current repository baseline is 808 passing tests. A different count after
future changes is acceptable only when the complete suite still passes and the
diff explains the change.

Run the layer-wise article audit when the aggregate layer-wise evidence is
present:

```bash
uv run --frozen --extra test python scripts/audit_layerwise_article.py
```

Run the manuscript verification target from the repository root:

```bash
make --directory=manuscript verify
```

This target regenerates deterministic manuscript assets from retained evidence,
runs the LaTeX build, and rejects undefined references. Local TeX font warnings
or `Underfull \\hbox` diagnostics are layout warnings, not evidence changes;
they should still be visually inspected before release.

## 4. Public aggregate asset checks

The retained secondary ds006780 bundle contains aggregate analysis and a
precision-gate report. Its deterministic table, figure, and LaTeX macro assets
can be regenerated into a temporary absolute output directory:

```bash
export DS006780_ANALYSIS="$PWD/results/extensions/ds006780_external_v5/external_analysis.json"
export DS006780_GATE="$PWD/results/extensions/ds006780_external_v5/precision_gate.json"
export DS006780_ASSETS="/absolute/path/to/temporary/ds006780_assets"

uv run --frozen --extra test --extra manuscript \
  python scripts/build_ds006780_manuscript_assets.py \
  --analysis "$DS006780_ANALYSIS" \
  --precision-gate "$DS006780_GATE" \
  --output-dir "$DS006780_ASSETS"
```

The builder validates the analysis hash, precision-gate hash, schema, and
provenance before writing output. Compare regenerated files with the retained
asset manifest before replacing any tracked artifact.

The expected interpretation of the v5 bundle is fixed: the n=800 rich-head
point estimate is positive across all ten seeds, but its subject-aware interval
has width `0.1272` against the predeclared `0.06` threshold. It is secondary,
uncertainty-limited transfer evidence, not a stable-superiority claim.

## 5. Inputs required for a full rerun

The following inputs belong outside Git and must be supplied through restricted
storage. Use absolute paths and preserve the source identity and hashes.

| Input | Purpose |
| --- | --- |
| HBN release and canonical subject manifest | Development subjects and fixed age-support interval |
| MIPDB release, BIDS tree, and finalized cohort manifest | Sealed primary external evaluation |
| ds006780 pinned BIDS snapshot and eligible-subject manifest | Secondary cross-cohort transfer |
| Pinned REVE checkpoint and channel mapping | Frozen encoder inference |
| HBN preprocessing and representation caches | Avoiding repeated extraction while preserving identities |
| Head-run checkpoints and run manifests | Exact checkpoint selection and external inference |
| Study lock, environment lock, and prediction inventory | Provenance and fail-closed resume |

Do not substitute a moving `latest` dataset release, a different REVE revision,
or an unpinned checkpoint. A replacement input requires a new study identity and
must not be mixed with the retained aggregate results.

## 6. Full prospective pipeline

After access-controlled inputs are mounted, follow the numbered commands in
[`article_ready_protocol.md`](article_ready_protocol.md) in order:

1. build the content-addressed MIPDB inventory;
2. run target-free pilot QC;
3. finalize the MIPDB cohort;
4. materialize the HBN representations;
5. train the exact four-head by ten-seed matrix;
6. seal the study from artifact-derived hashes;
7. run the sealed external holdout;
8. run confirmatory analysis only after the prediction inventory is complete.

The protocol fixes the encoder, channel layout, preprocessing, seeds 33--42,
checkpoint rule, optimizer, scheduler, 40-epoch budget, patience 7, and
subject-level estimand. Do not manually override a head, seed, checkpoint, or
external subject list during a rerun.

The primary decision rule requires all predeclared conditions: Holm-adjusted
one-sided significance, a bootstrap interval above zero, at least 8 of 10 seed
wins, and a worst-seed delta no lower than `-0.01`. Failure to meet the rule is
reported as failure to establish stable improvement; it is not converted into
an equivalence claim.

## 7. Evidence and privacy boundary

The following may be committed when they contain no participant-level data:

- protocol and training configurations;
- schemas and tests;
- aggregate analysis JSON;
- precision-gate reports;
- aggregate tables and figures;
- LaTeX macros and asset manifests;
- provenance hashes and access instructions.

The following must remain outside Git and outside public artifact bundles:

- raw EEG and BIDS recordings;
- participant IDs, exact ages, and target-bearing manifests;
- participant-level predictions and subject-level prediction rows;
- representation caches;
- model checkpoints and private run directories;
- credentials, tokens, SSH commands, and private server paths;
- logs that expose restricted paths or data values.

If a restricted artifact is archived for future exact reruns, store it in
access-controlled storage together with its content hash, source release, study
identity, and a short retention note. Do not copy it into this repository merely
to make a future clean-server launch easier.

## 8. Replacing or shutting down a server

Before deleting an instance or its local disk:

1. verify that the repository contains the current aggregate evidence and
   manuscript assets;
2. record the final repository revision and environment lock;
3. copy any still-needed private caches, checkpoints, manifests, and logs to
   persistent access-controlled storage;
4. verify the copied files against their recorded hashes;
5. confirm that a fresh server can access the required datasets and checkpoint;
6. stop or delete the old instance only after the backup verification succeeds.

Powering off an instance may preserve its attached disk, depending on the
provider. Terminating an instance commonly removes an ephemeral root disk. The
provider's storage semantics must therefore be checked before deletion.

## Reproduction limits

A clean checkout plus the retained aggregate bundle can reproduce the reported
analysis artifacts and manuscript, but not the original subject-level inference
without the restricted source artifacts. This is intentional: the paper's
public release is auditable without publishing neuroimaging data or participant
identifiers. Any exact rerun from private inputs must preserve the same protocol
identity, source hashes, environment lock, and evidence boundary.
