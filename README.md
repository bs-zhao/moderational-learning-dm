# Moderational Learning

Code and materials accompanying:

> Zhao, B., Radev, S. T., Sokratous, K., & Kvam, P. D.
> *Moderational Learning: A Framework for Discovering Models and Individual Differences from Behavioral Data*

Preprint: https://osf.io/preprints/psyarxiv/b865r_v1
OSF project: https://osf.io/jyw5u/

## Repository layout

```
.
├── models/                 # Core Neural Process / moderational-learning modules
├── simulation_studies/
│   ├── sim1_linear/        # Simulation study 1 (linear generative model)
│   └── sim2_hete/          # Simulation study 2 (heterogeneous generative model)
└── application/
    ├── data_prep/          # Scripts that build task .pkl files from raw CSV
    └── Peter_v4_factor/    # Empirical analyses on value-based decision data
```

Scripts append `../../models` to `sys.path`. Run them from their own directories so that relative `save/` paths resolve correctly.

## Requirements

- Python >= 3.10
- Optional CUDA GPU (CPU works; training is slower)

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Cognitive baseline fits also use R (`application/Peter_v4_factor/hcog_s2_fit.r`). Install a recent R release and the packages used in that script (e.g. `rstan`) if you need to reproduce those fits.

## Quick start (simulation)

From `simulation_studies/sim1_linear`:

```bash
python s1_gen_train_batch.py
```

Paper checkpoints for Study 1 (`Flex6Lv2b`, latent dim 16, beta = 1, input dims 1-6) are under `simulation_studies/sim1_linear/save/train_batch/`.
Study 2 checkpoint (`Hete4bn4`) is under `simulation_studies/sim2_hete/save/train_batch/`.

Figure / recovery scripts in each simulation folder load those checkpoints.

## Empirical application

Preprocessed task data:

- `application/Peter_v4_factor/save/s1_data/*.pkl`

Final trained models (only `all.pkl`, discovery and prediction settings):

- `application/Peter_v4_factor/save/s2_train/MLP_mod/`

Selected discovery hyperparameters (beta, sparse lambda):

| Task | beta | sparse lambda |
|------|------|---------------|
| Risky choice (`RiskC`) | 0.5 | 0 |
| Intertemporal choice (`DelayC`) | 0.5 | 0 |
| Risky pricing (`RiskP`) | 0.25 | 0.01 |
| Intertemporal pricing (`DelayP`) | 1 | 0 |
| BBRS risky choice (`BB`) | 0.25 | 0 |

Prediction-optimized models use beta = 0 (same five tasks).

Typical workflow from `application/Peter_v4_factor`:

1. (Optional) Rebuild `.pkl` data with scripts in `../data_prep/`.
2. Train: `python s2_train_choice_MLP_mod.py`
3. Latent / visualization analyses: `fr1_plot_var1.py`, `s3_2_latent_analysis.py`, `s3_3_moderational_check.py`, `s3_4_cross_factors.py`
4. Cross-validated training script: `s7_cross_test_mod_v3.py`
5. Cognitive baselines: `hcog_s1_getdata.py`, then `hcog_s2_fit.r`, then `hcog_s3_eval_generation.py`

## Notes

- This repository is English-only and contains filtered copies of the analysis code used in the manuscript.
- Large binary checkpoints (`.pkl`) are included for reproducibility. Each file is below GitHub's 100 MB per-file limit (total repo ~270 MB).
- Random seeds are set in the scripts; GPU nondeterminism may still cause small numerical differences.

## Citation

If you use this code, please cite the manuscript (and this repository / OSF materials).

## License

MIT - see `LICENSE`.
