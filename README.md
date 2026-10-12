# CriticOPD

Code for **Recovering Reliable Supervision via Critique-Guided On-Policy Distillation**.

On-policy distillation (OPD) trains a student on its own rollouts with token-level supervision from a
teacher. After the student makes an error, the teacher usually cannot continue correctly from the
student's prefix, although it can recognize the error. CriticOPD uses this ability:

1. **Critique.** If a rollout is incorrect, the teacher receives the problem, the reference answer and the
   rollout split into numbered paragraphs, and lists at most five errors. Each error has a paragraph index,
   the quoted sentence and a fix. An incomplete last item is discarded. If the teacher finds no error, the
   rollout is trained with standard OPD.
2. **Cut and rewrite.** The student keeps its prefix up to the paragraph of the last listed error. It then
   writes a new suffix with the critique, which ends with the reference answer, in its context.
3. **Feedback distillation (L_fb).** The training sequence is the kept prefix followed by the rewrite,
   without the critique. Each rewrite token gets the advantage
   `log pi_T(y'_t | x, p, c, y'_<t) - log pi_theta(y'_t | x, p, y'_<t)`: the teacher scores with the
   critique `c` in its context, and the student does not see it. The prefix receives no loss.
   Correct rollouts are trained with standard OPD.

## Repository layout

The code builds on the open-source [Relay-OPD](https://github.com/zju-real/Relay-OPD) code base
(verl, Apache-2.0). This repository holds only the files that CriticOPD adds or changes. `setup.sh`
copies them onto the base code at a pinned commit.

```
setup.sh                                   fetch the base code and add the files below
overlay/                                   mirrors the relay-opd/ directory of the base code
  verl/experimental/agent_loop/
    critic_opd_agent_loop.py               new: grading, critique, cut point, rewrite (the method)
    agent_loop.py                          changed: teacher client in the agent loop; L_fb rescoring
    __init__.py                            changed: registers the CriticOPD agent loop
  verl/trainer/ppo/ray_trainer.py          changed: rewrite-token metrics; mask for the L_center ablation
  verl/trainer/distillation/losses.py      changed: L_center ablation (advantage centering on rewrite tokens)
  opd/scripts/criticopd/
    train.sh                               training; the defaults are the final CriticOPD configuration
    ablation.sh                            every ablation and sensitivity run of the paper
    eval.sh                                evaluation protocol of the paper
```

Where each step of the method is implemented:

| Step | Code |
| --- | --- |
| Grade the rollout against the reference answer | `CriticOpdAgentLoop._run` → `_grade_async` (math grader in subprocesses) |
| Split into paragraphs, critic prompt | `segment`, `CRITIC_SYS_MULTI` / `CRITIC_SYS_MULTI_CAPPED` |
| Parse the critique, drop the incomplete item | `parse_critic_multi`, `drop_incomplete_tail`, `kerr_select` |
| Cut point and rewrite | `_tok_prefix_for_text` (cut on the original tokens), `FEEDBACK_TMPL`, student generation |
| Training sequence and mask | kept prefix → mask 0, rewrite → mask 1; the critique is not in the sequence |
| L_fb | the agent loop passes `critic_fb_prompt_ids`; `AgentLoopWorker` in `agent_loop.py` rescores the rewrite with the teacher on prompt + prefix + critique + rewrite and replaces the teacher log-probs of the rewrite tokens |

The OPD loss itself, the PG-style k1 reverse-KL loss of the base code, is unchanged.

## Installation

```bash
bash setup.sh                       # creates ./Relay-OPD with CriticOPD added
cd Relay-OPD
conda create -n criticopd python==3.12 -y
conda activate criticopd
pip3 install -c environment/vllm-constraints.txt vllm==0.21.0
cd relay-opd
pip3 install -e .
pip3 install -r requirements-relay-opd.txt
python environment/verify_install.py
```

The environment is that of the base code (Python 3.12, PyTorch 2.11, vLLM 0.21.0). See its README for
the locked installer.

## Models and data

| | |
| --- | --- |
| Teacher | `Qwen/Qwen3-4B-Instruct-2507` |
| Students | `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-0.6B` (non-thinking mode) |
| Training data | DAPO-Math-17k: 17,917 problems |
| Evaluation | AIME24 (30), AIME25 (30), AMC23 (83), MATH500 (500), OlympiadBench (675), Minerva (272) |

All data are verl parquets with a `prompt` column (chat messages) and a `reward_model.ground_truth`
column. The evaluation directory (`DATA_DIR`, also `BENCH`) holds one parquet per benchmark under the
file names in `BENCH_PATHS` of `opd/eval/math_benchmarks.py`: `aime-24.parquet`, `aime-2025.parquet`,
`amc23.parquet`, `math500.parquet`, `olympiadBench.parquet` and `minerva.parquet`. Their user message
ends with "Please reason step by step, and put your final answer within \boxed{}.". verl also loads
`aime-24_verl.parquet` and `aime-2025_verl.parquet` from `BENCH` for in-training validation, which is
off by default.

## Training

Run all commands from `Relay-OPD/relay-opd`. On one node with 4 GPUs (2 for the student, 2 for the
teacher):

```bash
export STUDENT_MODEL=/path/to/Qwen3-1.7B
export TEACHER_MODEL=/path/to/Qwen3-4B-Instruct-2507
export TRAIN_DATA=/path/to/dapo-math-17k.parquet
export BENCH=/path/to/eval_parquets
export CUDA_VISIBLE_DEVICES=0,1,2,3

SEED=42 OUTPUT_DIR=/path/to/runs/criticopd_seed42 bash opd/scripts/criticopd/train.sh
```

The paper reports seeds 42, 43 and 44 at step 40, which is the default length. Checkpoints are saved
every 20 steps; `<OUTPUT_DIR>/global_step_40/actor/huggingface` is the model to evaluate. For the
0.6B student, set `STUDENT_MODEL` to Qwen3-0.6B.

Shared hyperparameters: 128 prompts per step, one rollout per prompt, one optimizer update per step,
learning rate 1e-6 (constant, no warmup), prompt length up to 2,048 tokens, response length up to 16,384
tokens, and rollout temperature and top-p of 1.0. The critic decodes greedily, with at most 2,048 tokens
and five errors.

### Switches

The CriticOPD agent loop is configured through environment variables. `train.sh` sets the final
configuration:

| Variable | Final | Meaning |
| --- | --- | --- |
| `CRITIC_OPD_ENABLE` | `1` | `0` gives plain OPD |
| `CRITIC_OPD_GIVE_GT` | `1` | the critic receives the reference answer |
| `CRITIC_OPD_KERR` | `last` | the critic lists all errors and the cut is after the `k`-th (`1`, `2`, `3`, … or `last`); empty: single-error critic |
| `CRITIC_OPD_CRITIC_TOKENS` | `2048` | length limit of the critique (at least 1,024 for the multi-error critic unless `CRITIC_OPD_CRITIC_TOKENS_EXACT=1`) |
| `CRITIC_OPD_FB_ANSWER` | `1` | the critique ends with "The correct final answer is …" |
| `CRITIC_OPD_SPLICE` | `after` | the kept prefix ends after the erroneous paragraph (`before`: it ends before it) |
| `CRITIC_OPD_USE_FEEDBACK` | `1` | the critique is in the context of the rewrite (`0`: rewrite from the error without it) |
| `CRITIC_OPD_LOSS` | `fb` | `fb`: L_fb; `clean`: the teacher scores the rewrite without the critique; `center`: like `clean`, with the advantages of rewrite tokens centered on the mean on-policy advantage |
| `CRITIC_OPD_PREFIX_LOSS` | `0` | `1`: OPD loss also on the prefix before the first error |
| `CRITIC_OPD_CUT` | `error` | `end`: keep the whole incorrect solution and rewrite after it |
| `CRITIC_OPD_FB_MODE` | `critique` | `generic`: replace the critique with a fixed generic hint |
| `CRITIC_OPD_ALL` | `0` | `1`: critique every rollout, including correct ones |
| `CRITIC_OPD_TRUNC` | `0` | `1`: keep only the first paragraph of the rewrite (at most `CRITIC_OPD_TRUNC_MAX` tokens) |
| `CRITIC_OPD_KEEP_CORRECT` | `0` | `1`: keep only rewrites that reach the reference answer |
| `CRITIC_OPD_ANNEAL` | empty | `start,end,min`: repair probability falls linearly from 1 to `min` between two steps |

During training, the log prints a running tally of the agent loop (`[critic_opd] tally`) and the metrics
`critic_opd/repair_seq_frac` (share of rollouts that are rewritten) and `critic_opd/repair_tok_frac`.

## Ablations

`ablation.sh <variant>` runs one row of the ablation table (Qwen3-1.7B, seed 42) or one point of the
sensitivity analysis. It sets every switch explicitly and trains for the number of steps reported in the paper:

```bash
OUTPUT_ROOT=/path/to/runs bash opd/scripts/criticopd/ablation.sh generic_feedback
```

| Paper row | Variant | Steps |
| --- | --- | --- |
| *(a) From OPD to CriticOPD* | | |
| OPD | `opd` | 80 |
| two rollouts per prompt | `opd_two_rollouts` | 80 |
| + rewrite from the error | `rewrite_from_error` | 60 |
| + feedback in the rewrite | `feedback_in_rewrite` | 40 |
| + teacher scores with feedback | `teacher_scores_feedback` | 40 |
| + reference answer to the critic | `critic_reference_answer` | 40 |
| + all errors, cut after the last | `all_errors` | 40 |
| + 2,048-token limit | `limit_2048` | 40 |
| *(b) Changes to CriticOPD* | | |
| without feedback distillation | `no_feedback_distillation` | 40 |
| generic feedback | `generic_feedback` | 40 |
| loss before the first error | `loss_before_first_error` | 40 |
| cut at the end of the solution | `cut_at_end` | 40 |
| critique of every rollout | `critique_every_rollout` | 40 |
| *(c) Changes to earlier variants* | | |
| without the erroneous paragraph | `without_error_segment` | 40 |
| first paragraph of each rewrite | `first_paragraph` | 40 |
| rewrites with a correct answer | `correct_rewrites_only` | 40 |
| rewrites phased out by step 40 | `phase_out` | 40 |
| *Sensitivity* | | |
| number of listed errors k = 1, 2, 3 | `k1`, `k2`, `k3` | 40 |
| critique limit 512, 1,024, 4,096 tokens | `limit512`, `limit1024`, `limit4096` | 40 |

The variants of block (a) up to `critic_reference_answer` and those of block (c) use a single-error
critic with a 512-token limit. `all_errors` switches to the multi-error critic, whose limit is 1,024
tokens, and `limit_2048` raises the limit to 2,048 tokens. The rows of block (b) and the sensitivity
runs change `limit_2048`, which does not append the reference answer to the critique
(`CRITIC_OPD_FB_ANSWER=0`). To apply a change to the final configuration instead, call `train.sh` with
the corresponding switch, for example `CRITIC_OPD_FB_MODE=generic bash opd/scripts/criticopd/train.sh`.

## Evaluation

```bash
MODEL=/path/to/runs/criticopd_seed42/global_step_40/actor/huggingface \
DATA_DIR=/path/to/eval_parquets OUT_ROOT=/path/to/eval RUN_NAME=criticopd_seed42 STEP=40 \
bash opd/scripts/criticopd/eval.sh
```

The protocol samples with temperature 1.0 and top-p 1.0 and allows up to 32,768 new tokens. It reports
avg@32 on AIME24, AIME25 and AMC23 and avg@8 on MATH500, OlympiadBench and Minerva, and prints the six
scores and their average. Main result of the paper (Qwen3-1.7B, step 40, mean over three seeds):

| | AIME24 | AIME25 | AMC23 | MATH500 | Olympiad | Minerva | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OPD (step 80) | 35.10 | 22.78 | 62.94 | 86.26 | 56.65 | 36.78 | 50.08 |
| CriticOPD | 41.11 | 30.14 | 69.88 | 88.87 | 60.45 | 37.44 | 54.65 |

## License

Apache-2.0, as for the base code. The changed files keep their original license headers.
