#!/bin/bash
# Submit the GPT-4o teacher arm: Phase 1 -> Phase 2 (CLA + baseline parallel) -> Phase 4 (evals).
# Separate output paths from the Qwen-7B arm so both experiments coexist.

set -e

cd /home/hh89313/continous-latent-anchoring

mkdir -p /scratch/$USER/cla_anchors/gsm8k_gpt4o
mkdir -p /scratch/$USER/cla_runs
mkdir -p /scratch/$USER/cla_checkpoints

P1=$(sbatch --parsable sbatch/run_phase1_gpt4o.sbatch)
echo "Phase 1 GPT-4o (anchor gen):               $P1"

P2_CLA=$(sbatch --parsable --dependency=afterok:$P1 sbatch/run_phase2_cla_gpt4o.sbatch)
echo "Phase 2 GPT-4o (CLA training):             $P2_CLA  [waits on $P1]"

P2_BASE=$(sbatch --parsable --dependency=afterok:$P1 sbatch/run_phase2_baseline_gpt4o.sbatch)
echo "Phase 3 GPT-4o (CoCoT baseline):           $P2_BASE  [waits on $P1]"

P4_CLA=$(sbatch --parsable --dependency=afterok:$P2_CLA sbatch/run_phase4_cla_gpt4o.sbatch)
echo "Phase 4 GPT-4o (CLA evaluation):           $P4_CLA  [waits on $P2_CLA]"

P4_BASE=$(sbatch --parsable --dependency=afterok:$P2_BASE sbatch/run_phase4_baseline_gpt4o.sbatch)
echo "Phase 4 GPT-4o (baseline evaluation):      $P4_BASE  [waits on $P2_BASE]"

echo ""
echo "Queue:"
squeue -u $USER
echo ""
echo "Cancel the whole GPT-4o chain with:"
echo "  scancel $P1 $P2_CLA $P2_BASE $P4_CLA $P4_BASE"
