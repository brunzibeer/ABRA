# SLURM scripts (local only: `slurm/` is git-ignored)

Converted from the original `sb_*.sh` (conda + `main.py`) to `python -m abra` with uv. Hyper-parameters are the
original ones. Every script prints the full command it runs; logs go to `LOGS/<job>_<array job>_<task>.out`
(the original `%x_%A` made all tasks of an array write one file).

## Paths (slurm/common.sh)

| Variable | Default | Meaning |
|---|---|---|
| `REPO` | `/homes/mbernardi/git/abra-refactor` | this repo on the server (scripts are sourced from here) |
| `VENV` | `$REPO/.venv` | uv environment |
| `CKPT`, `CKPT_FILE` | `/work/horizon_store/abraout`, `best.pth` | checkpoints read (the original runs) |
| `RUNS` | `$ABRAOUT/refactor` | where every new run writes; the original runs are never overwritten |
| `CS`, `SD`, `WEIGHTS` | `$REPO/datasets/{cityscapes,sdgod}`, `$REPO/weights/groundingdino_swint_ogc.pth` | data, base model |
| `EXTRA` | empty | extra options appended to the command, e.g. `EXTRA="--fix_layer0_grad 1"` |
| `WANDB_PROJECT`, `WANDB_ENTITY` | unset | mirror metrics to wandb |

To evaluate or continue from checkpoints trained with the new code: `CKPT=$RUNS CKPT_FILE=best.safetensors`.

## One-time setup on the server

```bash
cd /homes/mbernardi/git/abra-refactor          # code of branch `refactor` + this slurm/ folder
module load cuda/11.8
uv sync                                         # creates .venv from uv.lock (Python 3.10, torch 2.1.0 cu118)
source .venv/bin/activate
cd groundingdino/models/GroundingDINO/ops && python setup.py build install && cd -   # deformable attention op
mkdir -p LOGS
sbatch slurm/data.sh                            # annotation files (fog train, few-shot, strict, top-k, multi) + stats
```
`python groundingdino/models/GroundingDINO/ops/test.py` checks the op on a GPU node.

## Old script -> new call

Submit from anywhere; positional arguments go after the script. `--array` selects classes or domains in the order
of `common.sh` (Cityscapes unavailable: bus motorcycle rider train truck; all: bicycle bus car motorcycle person rider
train truck; SDGOD unavailable: bike bus motorcycle rider; available: car person truck).

| Old | New |
|---|---|
| `sb_cs_train_dom` 0-1 DOM1 (2-3 DOM2, 4 DOM5, 5-6 DOM6, 7-8 DOM7, 9-10 DOM8) | `sbatch --array=0-1 slurm/train_domain.sh cityscapes dom1` (`dom2`, `--array=0 ... dom5`, `dom6`, `dom7`, `dom8`) |
| — (k ablation, strict target expert) | `slurm/train_domain.sh cityscapes top1` / `top2`; `--array=1 slurm/train_domain.sh cityscapes no_bus` / `no_unavailable` |
| `sb_sd_train_dom` 0-4 | `sbatch --array=0-4 --time=24:00:00 slurm/train_domain.sh sdgod dom1` |
| `sb_cs_train_exp_normal` EXP1 0-7 (EXP2/5/6/7/8) | `sbatch --array=0-7 slurm/train_expert.sh cityscapes exp1 normal` (`exp2`, `exp5` ... `exp8`) |
| `sb_cs_train_exp_fog` EXP1 0-7 | `sbatch --array=0-7 slurm/train_expert.sh cityscapes exp1 fog` |
| `sb_cs_train_exp_normal_zero` EXP9 0-7 | `sbatch --array=0-7 slurm/train_expert.sh cityscapes exp9 normal` |
| `sb_cs_train_exp_normal_full` 0-4 | `sbatch --array=0-4 slurm/train_full.sh cityscapes exp1full normal` (output `exp1full_<cls>`, no longer `exp5_<cls>`) |
| `sb_cs_train_ft_normal` FT2 0-7 / FT4 8-15 | `sbatch --array=0-7 slurm/train_expert.sh cityscapes ft2 normal` / `... ft4 normal` |
| `sb_cs_train_ft_fog` FT4 0-7 / FT5 8-15 | `sbatch --array=0-7 slurm/train_expert.sh cityscapes ft4 fog` / `sbatch --array=0-7 slurm/train_full.sh cityscapes ft5 fog` |
| `sb_sd_train_exp_ds` 0-3 | `sbatch --array=0-3 slurm/train_expert.sh sdgod exp1 daytime_sunny unavailable` |
| `sb_sd_train_exp_seen` 0-14 | per domain: `sbatch --array=0-2 slurm/train_expert.sh sdgod exp1 <domain> available` |
| `sb_sd_train_ft` 0-19 | per domain: `sbatch --array=0-3 slurm/train_expert.sh sdgod ft2 <domain> unavailable` |
| `sb_cs_eval_n2f` EVA1A/B/C/D/F/I/J | `sbatch slurm/eval.sh cityscapes fog <setting>`: A `source_multi`, B `task_add`, C `abra_multi`, D `abra`, F `abra_alpha`, I `abra_alpha_ap50`, J `abra_alpha_ap75` |
| `sb_cs_eval_f2n` EVA1A/B/C/D, EVA8A/B/C/D | `sbatch slurm/eval.sh cityscapes normal <setting> [exp8]`: A `source_multi`, B `source`, C `abra_multi`, D `abra` |
| `sb_sd_eval_ds2{df,dr,nr,ns}` | `sbatch --array=0-3 slurm/eval.sh sdgod <domain> <setting>`: A `source_multi`, B `source`, C `abra_multi`, D `abra`, α / α50 / α75 `abra_alpha[_ap50/_ap75]` |
| `sb_sd_eval_analogy_all` | `sbatch --array=0-3 slurm/eval.sh sdgod <domain> task_add` |
| `sb_sd_eval_exp8a` | `FT_DOMAIN=daytime_sunny sbatch --array=0-3 slurm/eval.sh sdgod <domain> ft_multi ft2` (the original pointed to `ft2/<cls>`, training wrote `ft2_<cls>`) |
| `sb_uda_fda` / `sb_uda_fda_abra` | `sbatch slurm/fda.sh fda` / `fda_abra` |
| `sb_uda_few_imgs` / `_abra` (N = 1 5 10 20 30) | `sbatch --time=00:30:00 slurm/train_expert.sh cityscapes few<N> fog` / `few<N>_abra` |
| — (new transported models instead of `transported/<cls>/merged.pth`) | `sbatch slurm/transport.sh cityscapes fog`, then `MERGED_TEMPLATE=$RUNS/transported/cityscapes/fog/%s/merged.safetensors` |
| — (paper rows not in the old scripts) | `slurm/eval.sh ... zero_shot`, `target`, `param_delta`; `exp9` residual on theta_T: `slurm/eval.sh cityscapes fog task_add exp9` |
| `sb_cs_eval_n2f` EVA1G (`abraplus`), `sb_bd_*` (BDD100K) | not converted (abraplus removed; BDD100K not used) |

Setting-compliant variants: `ALPHA_ANN=<coco json, %s = class>` selects α on another split; `EXTRA="--val_annotations <json>"`
selects checkpoints on another split; strict target experts with `TARGET_DOMAIN=$RUNS/domains/cityscapes/fog/no_%s/best.safetensors`.
