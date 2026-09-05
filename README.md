# Moderational Learning

Code and materials accompanying:

> Zhao, B., Radev, S. T., Sokratous, K., & Kvam, P. D.  
> *Moderational Learning: A Framework for Discovering Models and Individual Differences from Behavioral Data*

Preprint: https://osf.io/preprints/psyarxiv/b865r
OSF project: https://osf.io/jyw5u/

## Repository layout

```
.
├── models/                 # Core moderational-learning modules
├── simulation_studies/
│   ├── sim1_linear/        # Simulation Study 1
│   └── sim2_hete/          # Simulation Study 2
└── application/
    ├── data_prep/          # Data preprocessing
    └── Peter_v4_factor/    # Empirical analyses
```

Run scripts from their own directories so that relative imports and `save/` paths resolve correctly.

## Requirements

- Python >= 3.10
- CUDA GPU optional

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Cognitive-model baselines additionally require R and the packages used in `application/Peter_v4_factor/hcog_s2_fit.r`, including `rstan`.

## Simulation studies

### Simulation Study 1

From `simulation_studies/sim1_linear`:

```bash
python s1_gen_train_batch.py
```

Paper checkpoints (`Flex6Lv2b`, latent dimension 16, beta = 1, input dimensions 1–6) are in:

```
simulation_studies/sim1_linear/save/train_batch/
```

### Simulation Study 2

Paper checkpoint (`Hete4bn4`) is in:

```
simulation_studies/sim2_hete/save/train_batch/
```

Figure and recovery scripts in each simulation folder use the corresponding saved checkpoints.

## Empirical application

Preprocessed task data:

```
application/Peter_v4_factor/save/s1_data/*.pkl
```

Final trained models:

```
application/Peter_v4_factor/save/s2_train/MLP_mod/
```

### Discovery hyperparameters

| Task | beta | sparse lambda |
|------|------|---------------|
| Risky choice (`RiskC`) | 0.5 | 0 |
| Intertemporal choice (`DelayC`) | 0.5 | 0 |
| Risky pricing (`RiskP`) | 0.25 | 0.01 |
| Intertemporal pricing (`DelayP`) | 1 | 0 |
| BBRS risky choice (`BB`) | 0.25 | 0 |

Prediction models use beta = 0 for all five tasks.

### Main workflow

From `application/Peter_v4_factor`:

1. Optional data preprocessing: scripts in `../data_prep/`
2. Train:
   ```bash
   python s2_train_choice_MLP_mod.py
   ```
3. Latent and visualization analyses:
   ```
   fr1_plot_var1.py
   s3_2_latent_analysis.py
   s3_3_moderational_check.py
   s3_4_cross_factors.py
   ```
4. Cross-validation:
   ```bash
   python s7_cross_test_mod_v3.py
   ```
5. Cognitive-model baselines:
   ```
   hcog_s1_getdata.py
   hcog_s2_fit.r
   hcog_s3_eval_generation.py
   ```

## Citation

If you use this code, please cite the accompanying manuscript.

## License

MIT License. See `LICENSE`.
