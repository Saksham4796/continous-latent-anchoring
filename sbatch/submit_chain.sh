#!/bin/bash
# Submit Phase 1 -> Phase 2 (CLA + baseline in parallel) -> Phase 4 (evals) as a Slurm dependency chain.
# After Phase 4 finishes, review results under /scratch/$USER/cla_runs/ before submitting Phase 5.

set -e

cd /home/hh89313/continous-latent-anchoring

mkdir -p /scratch/$USER/cla_anchors/gsm8k
mkdir -p /scratch/$USER/cla_runs
mkdir -p /scratch/$USER/cla_checkpoints

P1=$(sbatch --parsable sbatch/run_phase1.sbatch)
echo "Phase 1 (anchor gen, GSM8K train+test):    $P1"

P2_CLA=$(sbatch --parsable --dependency=afterok:$P1 sbatch/run_phase2_cla.sbatch)
echo "Phase 2 (CLA training, lambda=1.0):        $P2_CLA  [waits on $P1]"

P2_BASE=$(sbatch --parsable --dependency=afterok:$P1 sbatch/run_phase2_baseline.sbatch)
echo "Phase 3 (CoCoT baseline, lambda=0.0):      $P2_BASE  [waits on $P1]"

P4_CLA=$(sbatch --parsable --dependency=afterok:$P2_CLA sbatch/run_phase4_cla.sbatch)
echo "Phase 4 (CLA evaluation):                  $P4_CLA  [waits on $P2_CLA]"

P4_BASE=$(sbatch --parsable --dependency=afterok:$P2_BASE sbatch/run_phase4_baseline.sbatch)
echo "Phase 4 (baseline evaluation):             $P4_BASE  [waits on $P2_BASE]"

echo ""
echo "Queue:"
squeue -u $USER
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo "  tail -f /scratch/$USER/cla_runs/phase2_cla.log"
echo ""
echo "Cancel the whole chain with:"
echo "  scancel $P1 $P2_CLA $P2_BASE $P4_CLA $P4_BASE"
