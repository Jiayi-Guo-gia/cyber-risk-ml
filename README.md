# Cyber Claim Prediction: Multi-label Classification and Multi-output Regression

Code for **"Entity-Specific Cyber Risk Assessment Using Insurtech-Empowered Risk Factors"**
Jiayi Guo, Linfeng Zhang, Zhiyu Quan. *Variance*, Vol. 19, 2026. <https://doi.org/10.66573/001c.155715>

We predict, for each firm-year, which of five cyber-incident categories lead to insurance claims
(`Data Breach`, `Privacy Violation`, `Extortion/Fraud`, `IT Error`, `Other`) and how many claims occur,
using 528 firm-level features. Models are Random Forest and LightGBM under several multi-output strategies,
and we compare feature importance across models with five attribution methods.

> **Data are not included.** The underlying data are proprietary. This repository contains the code,
> the tuned hyper-parameters, and the aggregate result tables/figures (feature-importance scores, heatmaps).
> See [Data](#data) for the expected file layout.

## Task formulation

| Task | Target | Strategies (prefix) | Base learners |
|---|---|---|---|
| Classification (`classification/`) | whether each category occurs (binary, 5 labels) | `MOC` MultiOutputClassifier · `CC` ClassifierChain · `MLRF` label-powerset Random Forest | RF, LightGBM |
| Regression (`regression/`) | claim count per category (5 outputs) | `MOR` MultiOutputRegressor · `RC` RegressorChain · `MORF` native multi-output RF | RF, LightGBM |

Script names follow `<STRATEGY>_<LEARNER>_<METHOD>.py`, e.g. `CC_RF_shap.py` = classifier chain + random forest + SHAP.

## Repository structure

```
.
├── preprocessing/
│   ├── raw_cleaning/               # raw JSON -> CSV, cleaning and flattening of the firm-level records
│   ├── data_processing.py          # dedup, one-hot claim types, aggregate per firm-year, join with features
│   ├── split_classification.py     # binarise targets, 80/20 split (seed 42)
│   └── split_regression.py         # count targets, 80/20 split (seed 42)
├── classification/
│   ├── tuning/                     # Optuna hyper-parameter search + chain-order search, with SLURM scripts
│   ├── baselines/                  # untuned reference models
│   ├── feature_importance/         # one script per (strategy, learner, method)
│   ├── notebooks/                  # model performance, importance heatmaps, residual analysis
│   ├── results/
│   │   ├── CC/, MOC/               # tuned hyper-parameters, chain orders, decision thresholds (JSON)
│   │   ├── feature/                # top-20 importance tables (CSV) and plots (PNG)
│   │   └── figures/                # paper figures
│   └── data/                       # <- put data here (git-ignored)
└── regression/                     # same layout as classification/
    ├── baselines/                  # RF, kernel ridge, SVR and multi-output SVR reference models
    └── notebooks/rf_hyperparameter_tuning.ipynb
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place the CSV files described in [`classification/data/README.md`](classification/data/README.md) and
[`regression/data/README.md`](regression/data/README.md) in the corresponding folders. They can be rebuilt from the
raw claim-level and firm-level tables with the preprocessing scripts.

## Reproducing the results

**1. Preprocessing** (run from the repository root)

`data_processing.py` expects two input files in `classification/data/`, `label_w_year.csv` (claim labels) and
`cyber_cleaned.csv` (firm-level features). They are derived from the raw data with `preprocessing/raw_cleaning/`
plus manual cleaning steps, and are not included in this repository.

```bash
python preprocessing/data_processing.py      # -> classification/data/label_cyber_year.csv
python preprocessing/split_classification.py # -> classification/data/{X,Y}_{train,test}.csv
cp classification/data/label_cyber_year.csv regression/data/
python preprocessing/split_regression.py     # -> regression/data/{X,Y}_{train,test}.csv
```

**2. Hyper-parameter tuning** (run from the *task* directory; 100 Optuna TPE trials, 5-fold CV, seed 42)

```bash
cd classification
python tuning/MOC_LGBM_TUNE.py     # also: MOC_RF_TUNE, CC_LGBM_ORDER, CC_RF_ORDER (chain order is searched too), MLRF_TUNE
sbatch tuning/run_MOC_LGBM_TUNE.sh # SLURM wrapper (edit the partition / module / conda env for your cluster)

cd ../regression
python tuning/MOR_LGBM_optimized.py  # also: MOR_RF_optimized, MORF_optimized, RC_LGBM_ORDER, RC_RF_ORDER
```

Each script writes best parameters, metrics, predictions and the Optuna study database under `results/`.
The tuned parameters used in the paper are already provided in `results/{CC,MOC}` (classification) and `results/{MOR,RC}`
(regression); the regression tuning scripts write to `results/{LGBM,RF,ORDER}`, so copy their JSON files into
`results/{MOR,RC}` (names as in the provided files) before running the feature-importance scripts on a re-tuned model.

**3. Feature importance** (run from the *task* directory; paths in the scripts are relative to it)

```bash
cd classification
python feature_importance/CC_RF_shap.py            # writes results/feature/RF_CC_shap_importance.csv + plot
python feature_importance/MOC_LGBM_permutation.py
```

Available methods per learner — LightGBM: `feature_gain`, `feature_split`, `shap`, `permutation`;
Random Forest: `feature_gini`/`feature_entropy` (classification) or `feature_impurity` (regression), `shap`, `permutation`.
Hyper-parameters are read from `results/<STRATEGY>/*.json`.

**4. Figures and tables** — open the notebooks in `*/notebooks/` with Jupyter started from the task directory
(`cd classification && jupyter lab`), since they read `results/` and `data/` via relative paths.

## Notes and limitations

- Row-level outputs (training residuals, predictions) are not published because they derive from the private data.

## Citation

If you use this code, please cite:

```bibtex
@article{guo2026entity,
  title   = {Entity-Specific Cyber Risk Assessment Using Insurtech-Empowered Risk Factors},
  author  = {Guo, Jiayi and Zhang, Linfeng and Quan, Zhiyu},
  journal = {Risk Management},
  volume  = {19},
  year    = {2026},
  doi     = {10.66573/001c.155715}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE). The license covers the code only, not the data.
