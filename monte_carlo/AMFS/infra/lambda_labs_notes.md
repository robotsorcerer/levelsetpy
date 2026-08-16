# Lambda Labs GPU onboarding notes (for AMFS Option C scaling)

Captured 2026-08-16 from the IRG ML Platform quickstart + bastion runbook.
Onboarding video: https://broadcast.amazon.com/videos/2029501
Quickstart:  https://irg-ml-platform.harmony.a2z.com/compute/lambda-labs/quick-start/index.html
Bastion:     https://irg-ml-platform.harmony.a2z.com/compute/lambda-labs/bastion-access/index.html
Slack: #irg-ml-ops-cont

## Why this matters for AMFS
Local CPU: the 30-seed harness (n_agents=14, 40 ticks) took ~7 min, dominated by
the O(n^2) per-micro-step BRT shield interpolations. GPU scaling unlocks:
  * Option C sweeps: many (n_agents x dist_sigma x window) cells x >=30 seeds.
  * Larger BRT precompute grids / higher-D relative dynamics via the JAX sampler
    (which already supports multi-GPU pmap in src/gpu_distribution.py).
  * The "robust throughput" benchmark axis (DESIGN §9.4) at RHCR agent counts.

## Recommended path: Metaflow via `imp` (one command)
Handles submission, retries/gang-restart, S3 checkpointing, W&B logging. Jobs are
auto-labeled team=IRG (avoids Fauna/River preemption).

First-time setup (once):
```
toolbox registry add s3://buildertoolbox-registry-ar-metaflow-us-west-2/tools.json
toolbox install ar-metaflow
ar-metaflow login --account 809338888231 --region us-east-2 --name irg-prod
./src/imp_cli/install.sh          # from an IRGMlPlatform workspace; needs brazil + uv
```
Each session:
```
ar-metaflow login irg-prod        # tokens ~1h; refresh: ar-metaflow auth refresh
export WANDB_API_KEY="<key>"; export WANDB_BASE_URL="https://apollo.wandb.io"; export WANDB_ENTITY="apollo"
```
Validate end-to-end before a real job:
```
imp run templates/ll_smoke_test.py         # 1 GPU, ~2-3 min; 10.43.x host = ran on LL
imp run templates/ll_smoke_test_gang.py    # 2-pod multi-node rendezvous check
```
Real training example (multi-node FSDP, 2x8=16 H100): `templates/ll_train_mnmg_fsdp.py`.
Keep `@pytorch_parallel` on the GPU step; Lambda Labs binding rides its JobSet.
Track: Metaflow/Argo UI (prod.irg-ml-metaflow.amazon.dev) + W&B (apollo.wandb.io).

## Advanced path: direct kubectl + Rancher via Bastion (debugging only)
SOCKS5 proxy on :6080 through the FIRE bastion. Add `Host ll-proxy` block to
~/.ssh/config (HostName firesp-basti-...elb.us-east-1.amazonaws.com), then:
```
mwinit && ssh -fN ll-proxy         # ssh -O check/stop ll-proxy to manage
kubectl config set-cluster local --proxy-url=socks5://localhost:6080
```
Rancher UI: rancher.amazon02.clusters.gpus.com (FoxyProxy SOCKS5 localhost:6080,
pattern amazon02.clusters.gpus.com; OIDC/Midway login).
**MUST label jobs `nodeSelector: {team: irg}`** (IRG owns 50 nodes) or Fauna/River
can preempt/kill. GPU jobs need the nvidia.com/gpu toleration; multi-node needs a
headless service + torchrun rendezvous (see runbook `multinode-train.yml`).

## How AMFS would use it (Option C plan)
1. Package the AMFS harness as a Metaflow flow: one step = one (config, seed) cell,
   fan-out over the grid; foreach seed for CRN pairing. Emit per-seed metrics to S3.
2. BRT precompute step uses the JAX sampler on GPU (bigger relative-state grids,
   optionally higher-D dynamics); cache npz to S3; downstream numpy steps read it.
3. Aggregate step runs the pure-numpy stats (bootstrap CIs, paired diffs,
   Holm-Bonferroni) and writes the results JSON + figures; log scalars to W&B.
4. Vectorize the shield's BRT interpolation (batch all pairs) before scaling —
   it is the current CPU bottleneck; a JAX/GPU batched interp removes it.
