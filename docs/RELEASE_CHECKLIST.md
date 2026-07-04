# Release Checklist

Use this checklist before pushing to the public/anonymous repository.

## Include

- `README.md`
- `.gitignore`
- `config/`
- `model/pglp_seg.py`
- `model/model_lrab_v1.py`
- `model/gatedsap_v2.py` if required by ablation scripts
- `tools/test_lrab_v1.py`
- `tools/train_lrab_v1.py`
- `tools/visualize_dataset_compare.py`
- `tools/make_diagnostic_vis.py`
- `utils/`
- `text/*.json` pseudo-label files needed for training, if releasing training
- `docs/WEIGHTS.md`
- `docs/EXPERIMENTS.md`

## Exclude

- `experiments/`
- `outputs/`
- `weights/`
- `pretrain/`
- `figures/`
- `vis/`
- `.agents/`
- `.codex-plugins/`
- `.idea/`
- `debug_list.txt`
- hard-coded local paper figure scripts
- Full epoch checkpoint histories
- Per-image prediction `.pt` dumps
- Local dataset paths

## Naming

Public-facing model name:

```python
from model.pglp_seg import PGLP_Seg
```

Public commands should use `PGLP_Seg`.

## Remote

Push to:

```text
https://github.com/zozo5085/pglp-seg
```

If this repository is meant to be anonymous for review, verify that the GitHub
account name, repository owner name, commit author, and README do not reveal the
paper authors.
