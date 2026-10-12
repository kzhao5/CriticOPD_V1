#!/usr/bin/env bash
# Ablations and sensitivity runs of the paper (Qwen3-1.7B student, seed 42).
#   bash opd/scripts/criticopd/ablation.sh <variant> [extra Hydra overrides]
# Needs the same environment as train.sh (STUDENT_MODEL, TEACHER_MODEL, TRAIN_DATA, BENCH) and either
# OUTPUT_DIR or OUTPUT_ROOT (the run then goes to $OUTPUT_ROOT/<variant>_seed<SEED>).
# Each variant sets every CriticOPD switch explicitly, so nothing is inherited from the defaults of train.sh.
# The number of training steps is the checkpoint reported in the paper (the learning rate is constant).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
variant=${1:?usage: ablation.sh <variant> [hydra overrides]}; shift

# CriticOPD without the reference answer in the feedback: base of block (b) and of the sensitivity runs
base_noans() {
  export CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1
  export CRITIC_OPD_KERR=last CRITIC_OPD_CRITIC_TOKENS=2048 CRITIC_OPD_FB_ANSWER=0
  export CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=0 CRITIC_OPD_ANNEAL=
}
# earlier variants: single-error critic with a 512-token limit, no reference answer
base_early() {
  export CRITIC_OPD_ENABLE=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_GIVE_GT=0 CRITIC_OPD_KERR= CRITIC_OPD_CRITIC_TOKENS=512
  export CRITIC_OPD_FB_ANSWER=0 CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=0 CRITIC_OPD_ANNEAL=
}
early_fb() { base_early; export CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_LOSS=fb; }   # + teacher scores with feedback
early_gt() { early_fb; export CRITIC_OPD_GIVE_GT=1; }                             # + reference answer to the critic

steps=40
case "$variant" in
  # ---- (a) from OPD to CriticOPD ----
  opd)                      export CRITIC_OPD_ENABLE=0 ROLLOUT_N=1; steps=80 ;;
  opd_two_rollouts)         export CRITIC_OPD_ENABLE=0 ROLLOUT_N=2; steps=80 ;;
  rewrite_from_error)       base_early; export CRITIC_OPD_USE_FEEDBACK=0 CRITIC_OPD_LOSS=center; steps=60 ;;
  feedback_in_rewrite)      base_early; export CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_LOSS=center ;;
  teacher_scores_feedback)  early_fb ;;
  critic_reference_answer)  early_gt ;;
  all_errors)               early_gt; export CRITIC_OPD_KERR=last ;;   # the multi-error critic uses at least 1,024 tokens
  limit_2048)               base_noans ;;
  # ---- (b) changes to CriticOPD ----
  no_feedback_distillation) base_noans; export CRITIC_OPD_LOSS=clean ;;
  generic_feedback)         base_noans; export CRITIC_OPD_FB_MODE=generic ;;
  loss_before_first_error)  base_noans; export CRITIC_OPD_PREFIX_LOSS=1 ;;
  cut_at_end)               base_noans; export CRITIC_OPD_CUT=end ;;
  critique_every_rollout)   base_noans; export CRITIC_OPD_ALL=1 ;;
  # ---- (c) changes to earlier variants ----
  without_error_segment)    early_gt; export CRITIC_OPD_SPLICE=before ;;
  first_paragraph)          early_fb; export CRITIC_OPD_TRUNC=1 CRITIC_OPD_TRUNC_MAX=512 ;;
  correct_rewrites_only)    early_gt; export CRITIC_OPD_KEEP_CORRECT=1 ;;
  phase_out)                early_gt; export CRITIC_OPD_ANNEAL=20,40,0 ;;
  # ---- sensitivity: number k of listed errors, length limit of the critique ----
  k1|k2|k3)                 base_noans; export CRITIC_OPD_KERR=${variant#k} ;;
  limit512|limit1024|limit4096)
                            base_noans; export CRITIC_OPD_CRITIC_TOKENS=${variant#limit} CRITIC_OPD_CRITIC_TOKENS_EXACT=1 ;;
  *) echo "unknown variant: $variant" >&2; exit 1 ;;
esac

export SEED=${SEED:-42}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-$steps}
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  export OUTPUT_DIR=${OUTPUT_ROOT:?set OUTPUT_DIR or OUTPUT_ROOT}/${variant}_seed${SEED}
fi
exec bash "${SCRIPT_DIR}/train.sh" "$@"
