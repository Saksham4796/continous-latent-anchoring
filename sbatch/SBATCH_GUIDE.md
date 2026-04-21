# Sapelo2 Slurm Pipeline

Sbatch scripts for running the CLA pipeline on the UGA GACRC Sapelo2 cluster.
All outputs go to `/scratch/$USER/` (no quota, 30-day purge — move keepers to `/project/<group>/` after).

## Core pipeline (Phase 1 -> 4)

| Script                       | Purpose                                          | GPU    | Walltime | Rough runtime |
|------------------------------|--------------------------------------------------|--------|----------|---------------|
| `run_phase1.sbatch`          | GSM8K anchor generation (1000 train + 250 test)  | A100   | 6h       | ~1.5-2h       |
| `run_phase2_cla.sbatch`      | CLA training (lambda=1.0)                        | A100   | 12h      | ~3-6h         |
| `run_phase2_baseline.sbatch` | Continuous-CoT baseline (lambda=0.0)             | A100   | 12h      | ~3-6h         |
| `run_phase4_cla.sbatch`      | CLA eval (accuracy + drift + hardware)           | A100   | 4h       | ~30-90min     |
| `run_phase4_baseline.sbatch` | Baseline eval                                    | A100   | 4h       | ~30-90min     |

All commands below are run from the repo root (`/home/hh89313/continous-latent-anchoring/`).
The sbatch files cd to the repo root internally, so they can be submitted from anywhere.

Submit the full chain (Phase 2 CLA and baseline run in parallel after Phase 1):

```bash
bash sbatch/submit_chain.sh
```

Or submit individually (useful for inspecting intermediate results):

```bash
sbatch sbatch/run_phase1.sbatch                                # note the job ID, then:
sbatch --dependency=afterok:<P1_ID> sbatch/run_phase2_cla.sbatch
sbatch --dependency=afterok:<P1_ID> sbatch/run_phase2_baseline.sbatch
# ...and so on
```

## Phase 5 ablations (expensive — review Phase 4 results first)

| Script                         | Runs | Est. GPU-hours |
|--------------------------------|------|----------------|
| `run_phase5_negatives.sbatch`  |   2  |     6-12       |
| `run_phase5_k_sweep.sbatch`    |  10  |    40-80       |
| `run_phase5_hyperparams.sbatch`|  12  |    50-100      |

Cross-domain ablation is not included — it requires generating StrategyQA train/test anchors first.

## Monitoring

```bash
squeue -u $USER                                          # queue status
tail -f /scratch/$USER/cla_runs/phase2_cla.log           # live training log
scontrol show job <jobid>                                # detailed job info
sacct -j <jobid> --format=JobID,JobName,State,Elapsed    # post-hoc
scancel <jobid>                                          # cancel
```

## Outputs

- Anchor shards: `/scratch/$USER/cla_anchors/gsm8k/{train,test}/gsm8k_*_anchors_*.pt`
- Checkpoints:   `/scratch/$USER/cla_checkpoints/{qwen15b_cla,qwen15b_cocot_baseline}/final_projection.pt`
- Ablations:     `/scratch/$USER/cla_ablations/`
- Slurm stdout:  `/scratch/$USER/cla_runs/*.out`
- Logs:          `/scratch/$USER/cla_runs/phase*.log`

Move final checkpoints and eval results to `/project/zxlab/` for persistence before the `/scratch` 30-day purge.
