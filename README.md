# Privileged-to-Deployable Wearable Stress Recognition

This repository contains the code for reproducing the paper experiments on
watch-only wearable stress recognition with training-time privileged
physiological supervision.

The core setting is:

- **Deployable signals at inference:** watch PPG/BVP and accelerometry.
- **Privileged signals during training:** richer physiological sensors such as
  E4 BVP/ACC, Polar-derived rhythm/IBI targets, or WESAD chest sensors.
- **Goal:** train a deployable watch model that can benefit from richer
  training-time sensors without requiring them at inference.

The final code supports two main research questions:

1. **How should accelerometry enter a pulse-based watch model?**  
   We compare pulse-only, direct/simple fusion, gated fusion, and a
   motion-aware encoder that uses ACC as context for pulse adaptation and
   artifact-aware pulse cleaning/refinement.

2. **How should a deployable watch model learn from privileged teachers?**  
   We compare Standard KD, teacher-gated KD, student-gated KD, and
   Cross-Gated KD under the same motion-aware deployable path.

## Repository Layout

Important files are summarized in:

- [`CODE_MAP.md`](CODE_MAP.md): final release file map.
- [`REPRODUCE.md`](REPRODUCE.md): command-level reproduction guide.

The most important entry points are:

- `run_watch_input_ablation_loso.py`  
  RQ1 watch-input ablations.
- `run_galaxy_loso_eval_elastic_scaled_motion.py`  
  Galaxy PPG KD-regime experiments.
- `run_wesad_loso_eval_elastic_scaled_motion.py`  
  WESAD KD-regime experiments.
- `export_window_predictions.py`  
  Export fold/window predictions for statistical analysis.
- `align_window_prediction_thresholds.py`  
  Align exported predictions with the fold-specific thresholds used in the
  reported LOSO tables.
- `analyze_paired_bootstrap.py`  
  Subject-clustered paired bootstrap confidence intervals and Wilcoxon tests.
- `plot_motion_internal_six_panel.py` and
  `plot_gated_kd_reliability_curve.py`  
  Paper mechanism figures.

## Installation

Python 3.10 or later is recommended.  A clean conda environment is usually the
least painful setup:

```bash
conda create -n stress-distill python=3.10
conda activate stress-distill
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA setup if the default pip wheel
is not appropriate for your machine.  The requirements file intentionally uses
`numpy<2` because some scientific Python wheels used by this project, notably
PyWavelets and matplotlib, can fail with NumPy 2.x ABI mismatches in older
environments.

## Data

This repository does not redistribute Galaxy PPG or WESAD data.  Prepare the
datasets separately and point the commands in `REPRODUCE.md` to their local
paths.

Expected high-level layouts:

```text
/root/code/Dataset                 # Galaxy PPG dataset root
/root/blockdata/WESAD              # WESAD dataset root
```

The paper experiments use LOSO manifests.  See `REPRODUCE.md` for manifest
generation commands and the exact session filters:

- Galaxy PPG: `baseline` vs `tsst-prep`
- WESAD: `baseline` vs `stress`

## Reproducing Results

Start with:

```bash
python -m stress_ssl_distill.summarize_dataset_manifests --help
python -m stress_ssl_distill.run_watch_input_ablation_loso --help
python -m stress_ssl_distill.run_galaxy_loso_eval_elastic_scaled_motion --help
python -m stress_ssl_distill.run_wesad_loso_eval_elastic_scaled_motion --help
```

Then follow [`REPRODUCE.md`](REPRODUCE.md), which documents:

1. Dataset/window preparation.
2. LOSO manifest generation.
3. RQ1 watch-input ablations.
4. RQ2 KD-regime comparisons.
5. Window prediction export.
6. Threshold alignment.
7. Paired bootstrap/Wilcoxon statistics.
8. Figure generation.
9. Deployment efficiency measurement.

## Final Protocol At A Glance

- Selection monitor: validation AUROC.
- Threshold metric: validation balanced accuracy.
- Epochs: 80.
- Early stopping patience: 10.
- Seed: 42.
- Learning rate: `3e-4`.
- Weight decay: `1e-4`.
- Focal gamma: `1.5`.
- Label smoothing: `0.05`.
- Watch-side wavelet weight: `0.05`.
- KD temperature: `4.0`.
- KD weight: `0.08`.

Galaxy teacher-guided training additionally uses E4 auxiliary classification
and Polar-derived rhythm/IBI auxiliary supervision.  These privileged auxiliary
signals are training-only and are removed at inference.  WESAD does not use E4
or Polar-derived auxiliary targets.

## Inference Constraint

The deployment path uses only the watch-side encoder and classifier.  Privileged
teachers, teacher logits, E4/Polar/WESAD privileged signals, and teacher-side
auxiliary heads are not required at inference.

## License

This code is released under the MIT License.  See [`LICENSE`](LICENSE).

## Citation

If you use this code, please cite the associated paper.  A BibTeX entry can be
added here after publication.
