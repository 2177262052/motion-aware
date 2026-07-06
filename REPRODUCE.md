# Reproducing The Final Paper Experiments

Commands assume the package directory is named `stress_ssl_distill` and are run
from the repository root:

```bash
cd /root/code
```

If this cleanup copy is still named `stress_ssl_distill_new`, rename it before
running the LOSO wrappers because several wrappers launch subprocesses with
`python -m stress_ssl_distill...`.

## Final Protocol

- Evaluation: leave-one-subject-out with subject-disjoint validation.
- Model selection: validation AUROC.
- Test threshold: selected on validation balanced accuracy.
- Epochs: `80`.
- Early stopping patience: `10`.
- Seed: `42`.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Focal gamma: `1.5`.
- Label smoothing: `0.05`.
- Watch wavelet loss: `0.05`.
- KD temperature: `4.0`.
- KD weight: `0.08`.
- Galaxy sessions: `baseline` vs `tsst-prep`.
- WESAD sessions: `baseline` vs `stress`.

Galaxy teacher-guided runs use E4 auxiliary classification
(`--e4-cls-weight 0.05`) and Polar-derived rhythm/IBI supervision
(`--rhythm-weight 0.15`).  WESAD does not use E4 or Polar auxiliary targets.

## 1. Build Manifests

### Galaxy PPG

```bash
python -m stress_ssl_distill.galaxy_manifest_loso_val \
  --dataset-root /root/code/Dataset \
  --output-dir artifacts/manifests_loso_val_20s \
  --window-s 20 \
  --stride-s 20 \
  --calm-sessions baseline \
  --stress-sessions tsst-prep
```

### WESAD

```bash
python -m stress_ssl_distill.prepare_wesad_windows \
  --wesad-root /root/blockdata/WESAD \
  --output-dir artifacts/wesad/windows_20s \
  --window-s 20 \
  --stride-s 10

python -m stress_ssl_distill.wesad_manifest_loso_val \
  --dataset-root /root/blockdata/WESAD \
  --output-dir artifacts/wesad/manifests_loso_val_20s \
  --window-s 20 \
  --stride-s 10 \
  --calm-sessions baseline \
  --stress-sessions stress
```

## 2. RQ1 Input-Role Ablations

These runs test how ACC should enter the pulse/BVP model.  The refine variants
keep wavelet-guided physiology refinement available across input baselines.

### Galaxy PPG

```bash
python -m stress_ssl_distill.run_watch_input_ablation_loso \
  --dataset-kind galaxy \
  --ablation ppg_only_refine \
  --manifests-dir artifacts/manifests_loso_val_20s \
  --dataset-root /root/code/Dataset \
  --output-dir artifacts/galaxy/ppg_only_refine_wavelet005_p1 \
  --device cuda --epochs 80 --monitor auroc --threshold-metric balanced_acc \
  --early-stop-patience 10 --seed 42 --watch-batch-size 32 --lr 3e-4 \
  --num-workers 4 --pin-memory --watch-wavelet-weight 0.05 \
  --calm-sessions baseline --stress-sessions tsst-prep

python -m stress_ssl_distill.run_watch_input_ablation_loso \
  --dataset-kind galaxy \
  --ablation simple_concat_refine \
  --manifests-dir artifacts/manifests_loso_val_20s \
  --dataset-root /root/code/Dataset \
  --output-dir artifacts/galaxy/simple_concat_refine_wavelet005_p1 \
  --device cuda --epochs 80 --monitor auroc --threshold-metric balanced_acc \
  --early-stop-patience 10 --seed 42 --watch-batch-size 32 --lr 3e-4 \
  --num-workers 4 --pin-memory --watch-wavelet-weight 0.05 \
  --calm-sessions baseline --stress-sessions tsst-prep

python -m stress_ssl_distill.run_watch_input_ablation_loso \
  --dataset-kind galaxy \
  --ablation gated_fusion_refine \
  --manifests-dir artifacts/manifests_loso_val_20s \
  --dataset-root /root/code/Dataset \
  --output-dir artifacts/galaxy/gated_fusion_refine_wavelet005_p1 \
  --device cuda --epochs 80 --monitor auroc --threshold-metric balanced_acc \
  --early-stop-patience 10 --seed 42 --watch-batch-size 32 --lr 3e-4 \
  --num-workers 4 --pin-memory --watch-wavelet-weight 0.05 \
  --calm-sessions baseline --stress-sessions tsst-prep

python -m stress_ssl_distill.run_galaxy_watch_loso_eval_scaled_motion \
  --manifests-dir artifacts/manifests_loso_val_20s \
  --dataset-root /root/code/Dataset \
  --output-dir artifacts/galaxy/watch_only_scaled_motion_wavelet005_p1 \
  --device cuda --epochs 80 --monitor auroc --threshold-metric balanced_acc \
  --early-stop-patience 10 --seed 42 --batch-size 32 --lr 3e-4 \
  --num-workers 4 --pin-memory --watch-enhancement motion_disentangled \
  --wavelet-weight 0.05 \
  --calm-sessions baseline --stress-sessions tsst-prep
```

### WESAD

```bash
python -m stress_ssl_distill.run_watch_input_ablation_loso \
  --dataset-kind wesad \
  --ablation ppg_only_refine \
  --manifests-dir artifacts/wesad/manifests_loso_val_20s \
  --dataset-root /root/blockdata/WESAD \
  --output-dir artifacts/wesad/ppg_only_refine_wavelet005_p1 \
  --device cuda --epochs 80 --monitor auroc --threshold-metric balanced_acc \
  --early-stop-patience 10 --seed 42 --watch-batch-size 32 --lr 3e-4 \
  --num-workers 4 --pin-memory --watch-wavelet-weight 0.05 \
  --calm-sessions baseline --stress-sessions stress

python -m stress_ssl_distill.run_watch_input_ablation_loso \
  --dataset-kind wesad \
  --ablation simple_concat_refine \
  --manifests-dir artifacts/wesad/manifests_loso_val_20s \
  --dataset-root /root/blockdata/WESAD \
  --output-dir artifacts/wesad/simple_concat_refine_wavelet005_p1 \
  --device cuda --epochs 80 --monitor auroc --threshold-metric balanced_acc \
  --early-stop-patience 10 --seed 42 --watch-batch-size 32 --lr 3e-4 \
  --num-workers 4 --pin-memory --watch-wavelet-weight 0.05 \
  --calm-sessions baseline --stress-sessions stress

python -m stress_ssl_distill.run_watch_input_ablation_loso \
  --dataset-kind wesad \
  --ablation gated_fusion_refine \
  --manifests-dir artifacts/wesad/manifests_loso_val_20s \
  --dataset-root /root/blockdata/WESAD \
  --output-dir artifacts/wesad/gated_fusion_refine_wavelet005_p1 \
  --device cuda --epochs 80 --monitor auroc --threshold-metric balanced_acc \
  --early-stop-patience 10 --seed 42 --watch-batch-size 32 --lr 3e-4 \
  --num-workers 4 --pin-memory --watch-wavelet-weight 0.05 \
  --calm-sessions baseline --stress-sessions stress

python -m stress_ssl_distill.run_wesad_watch_loso_eval_scaled_motion \
  --manifests-dir artifacts/wesad/manifests_loso_val_20s \
  --dataset-root /root/blockdata/WESAD \
  --output-dir artifacts/wesad/watch_only_scaled_motion_wavelet005_p1 \
  --device cuda --epochs 80 --monitor auroc --threshold-metric balanced_acc \
  --early-stop-patience 10 --seed 42 --batch-size 32 --lr 3e-4 \
  --num-workers 4 --pin-memory --watch-enhancement motion_disentangled \
  --wavelet-weight 0.05 --cache-subjects 15 \
  --calm-sessions baseline --stress-sessions stress
```

## 3. RQ2 KD Regimes

All KD regimes use the same scaled motion-aware deployable watch path.

### Galaxy PPG

Standard KD:

```bash
python -m stress_ssl_distill.run_galaxy_loso_eval_elastic_scaled_motion \
  --manifests-dir artifacts/manifests_loso_val_20s \
  --dataset-root /root/code/Dataset \
  --output-dir artifacts/galaxy/standard_kd_detach_scaled_motion_distill008_p1 \
  --skip-watch-only \
  --device cuda --epochs 80 --monitor auroc --threshold-metric balanced_acc \
  --early-stop-patience 10 --seed 42 --watch-batch-size 32 --deploy-batch-size 16 \
  --lr 3e-4 --num-workers 4 --pin-memory \
  --deploy-watch-enhancement motion_disentangled \
  --distill-weight 0.08 --distill-temp 4.0 --detach-standard-kd-teacher \
  --teacher-cls-weight 0.80 --e4-cls-weight 0.05 --rhythm-weight 0.15 \
  --wavelet-weight 0.05 \
  --calm-sessions baseline --stress-sessions tsst-prep
```

Cross-Gated KD:

```bash
python -m stress_ssl_distill.run_galaxy_loso_eval_elastic_scaled_motion \
  --manifests-dir artifacts/manifests_loso_val_20s \
  --dataset-root /root/code/Dataset \
  --output-dir artifacts/galaxy/cross_gated_kd_scaled_motion_distill008_p1 \
  --skip-watch-only \
  --device cuda --epochs 80 --monitor auroc --threshold-metric balanced_acc \
  --early-stop-patience 10 --seed 42 --watch-batch-size 32 --deploy-batch-size 16 \
  --lr 3e-4 --num-workers 4 --pin-memory \
  --deploy-watch-enhancement motion_disentangled \
  --distill-weight 0.08 --distill-temp 4.0 \
  --cross-confidence-distill --cross-confidence-min-weight 0.0 \
  --teacher-cls-weight 0.80 --e4-cls-weight 0.05 --rhythm-weight 0.15 \
  --wavelet-weight 0.05 \
  --calm-sessions baseline --stress-sessions tsst-prep
```

Teacher-gated and student-gated KD use the Standard KD command with one of:

```bash
--kd-gate-mode teacher_true_confidence --kd-gate-min-weight 0.0
--kd-gate-mode student_true_confidence --kd-gate-min-weight 0.0
```

### WESAD

Standard KD:

```bash
python -m stress_ssl_distill.run_wesad_loso_eval_elastic_scaled_motion \
  --manifests-dir artifacts/wesad/manifests_loso_val_20s \
  --dataset-root /root/blockdata/WESAD \
  --output-dir artifacts/wesad/standard_kd_scaled_motion_distill008_p1 \
  --skip-watch-only \
  --device cuda --epochs 80 --monitor auroc --threshold-metric balanced_acc \
  --early-stop-patience 10 --seed 42 --watch-batch-size 32 --deploy-batch-size 16 \
  --lr 3e-4 --num-workers 4 --pin-memory --cache-subjects 15 \
  --deploy-watch-enhancement motion_disentangled \
  --distill-weight 0.08 --distill-temp 4.0 --detach-standard-kd-teacher \
  --teacher-cls-weight 0.80 --privileged-cls-weight 0.05 --priv-wavelet-weight 0.05 \
  --calm-sessions baseline --stress-sessions stress
```

Cross-Gated KD:

```bash
python -m stress_ssl_distill.run_wesad_loso_eval_elastic_scaled_motion \
  --manifests-dir artifacts/wesad/manifests_loso_val_20s \
  --dataset-root /root/blockdata/WESAD \
  --output-dir artifacts/wesad/cross_gated_kd_scaled_motion_distill008_p1 \
  --skip-watch-only \
  --device cuda --epochs 80 --monitor auroc --threshold-metric balanced_acc \
  --early-stop-patience 10 --seed 42 --watch-batch-size 32 --deploy-batch-size 16 \
  --lr 3e-4 --num-workers 4 --pin-memory --cache-subjects 15 \
  --deploy-watch-enhancement motion_disentangled \
  --distill-weight 0.08 --distill-temp 4.0 \
  --cross-confidence-distill --cross-confidence-min-weight 0.0 \
  --teacher-cls-weight 0.80 --privileged-cls-weight 0.05 --priv-wavelet-weight 0.05 \
  --calm-sessions baseline --stress-sessions stress
```

Teacher-gated and student-gated KD use the Standard KD command with one of:

```bash
--kd-gate-mode teacher_true_confidence --kd-gate-min-weight 0.0
--kd-gate-mode student_true_confidence --kd-gate-min-weight 0.0
```

## 4. Window Predictions, Threshold Alignment, And Statistics

Export per-window predictions:

```bash
python -m stress_ssl_distill.export_window_predictions --help
```

Align exported predictions to the reported fold thresholds:

```bash
python -m stress_ssl_distill.align_window_prediction_thresholds --help
```

Compute paired subject-clustered bootstrap CIs and Wilcoxon tests:

```bash
python -m stress_ssl_distill.analyze_paired_bootstrap \
  --registry artifacts/paper_stats/window_prediction_registry_aligned.csv \
  --output-dir artifacts/paper_stats/paired_bootstrap
```

## 5. Figures

```bash
python -m stress_ssl_distill.plot_combined_subject_deltas --help
python -m stress_ssl_distill.plot_motion_internal_six_panel --help
python -m stress_ssl_distill.plot_gated_kd_reliability_curve --help
python -m stress_ssl_distill.plot_threshold_robustness_curves --help
```

## 6. Deployment Efficiency

```bash
python -m stress_ssl_distill.measure_deployment_efficiency \
  --dataset-kind galaxy \
  --checkpoint-dir ppg_only=artifacts/galaxy/ppg_only_refine_wavelet005_p1/checkpoints \
  --checkpoint-dir motion_aware=artifacts/galaxy/watch_only_scaled_motion_wavelet005_p1/checkpoints \
  --checkpoint-dir cross_gated_kd=artifacts/galaxy/cross_gated_kd_scaled_motion_distill008_p1/checkpoints \
  --output-dir artifacts/efficiency/galaxy_final \
  --device cpu --batch-size 1 --watch-channels 5 --watch-length 500 \
  --warmup 50 --repeats 1000 --max-checkpoints-per-set 1
```
