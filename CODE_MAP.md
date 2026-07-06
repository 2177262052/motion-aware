# Code Map

This file maps the release-facing code for the final paper experiments.  It
intentionally excludes earlier exploratory losses and model variants.

> Package-name note: several LOSO wrappers launch subprocesses with
> `python -m stress_ssl_distill...`.  Before running those wrappers, rename
> this cleanup directory to `stress_ssl_distill` or update the module strings.

## Dataset Preparation

- `prepare_wesad_windows.py`  
  Window WESAD signals.
- `galaxy_manifest.py`, `galaxy_manifest_loso_val.py`  
  Galaxy manifest utilities and LOSO/validation split generation.
- `wesad_manifest_loso_val.py`  
  WESAD LOSO/validation manifest generation.
- `summarize_dataset_manifests.py`  
  Dataset/window summary tables.

## Models

- `galaxy_models.py`  
  Galaxy watch encoders, motion-aware pulse encoder, E4 privileged teacher, and
  final Galaxy privileged-to-deployable model.
- `wesad_models_safe_sgpc.py`  
  WESAD wrist watch encoder wrapper, chest privileged encoder, and final WESAD
  privileged-to-deployable model.
- `reliability.py`  
  True-class confidence, cross-confidence trust, and trust-weighted KD.
- `metrics.py`, `early_stopping.py`, `samplers.py`  
  Evaluation, early stopping, and optional subject-aware batching utilities.

## Training And LOSO Evaluation

- `train_galaxy_watch.py`, `train_galaxy_watch_scaled_motion.py`  
  Single-fold Galaxy watch-only training.
- `train_wesad_watch.py`, `train_wesad_watch_scaled_motion.py`  
  Single-fold WESAD watch-only training.
- `train_galaxy_privileged_elastic.py`  
  Final Galaxy privileged training for Standard KD, teacher-gated KD,
  student-gated KD, and Cross-Gated KD.
- `train_wesad_privileged_elastic.py`  
  Final WESAD privileged training for the same KD regimes.
- `run_galaxy_watch_loso_eval.py`,
  `run_galaxy_watch_loso_eval_scaled_motion.py`  
  Galaxy watch-only LOSO wrappers.
- `run_wesad_watch_loso_eval.py`,
  `run_wesad_watch_loso_eval_scaled_motion.py`  
  WESAD watch-only LOSO wrappers.
- `run_galaxy_loso_eval_elastic.py`,
  `run_galaxy_loso_eval_elastic_scaled_motion.py`  
  Galaxy privileged/KD LOSO utilities and final scaled-motion wrapper.
- `run_wesad_loso_eval_elastic.py`,
  `run_wesad_loso_eval_elastic_scaled_motion.py`  
  WESAD privileged/KD LOSO utilities and final scaled-motion wrapper.
- `run_watch_input_ablation_loso.py`  
  RQ1 input-role ablations: pulse/BVP-only, ACC-only, direct concat, gated
  fusion, and motion-aware variants.

Historical filenames containing `elastic` are kept for command compatibility;
the release code no longer trains elastic residual correction modules.

## Prediction Export, Statistics, And Figures

- `export_window_predictions.py`  
  Export fold/window probabilities.
- `align_window_prediction_thresholds.py`  
  Align exported probabilities to the reported fold thresholds.
- `analyze_paired_bootstrap.py`,
  `compute_paired_ci_wilcoxon.py`,
  `summarize_named_paired_statistics.py`  
  Paired subject-clustered bootstrap CIs and Wilcoxon summaries.
- `analyze_threshold_robustness.py`,
  `plot_threshold_robustness_curves.py`  
  Threshold-offset robustness analysis and plots.
- `analyze_motion_aware_mechanism.py`,
  `analyze_motion_internal_response.py`,
  `plot_motion_internal_six_panel.py`  
  Motion-aware mechanism analysis and figure generation.
- `plot_gated_kd_reliability_curve.py`  
  Cross-Gated KD reliability mechanism figure.
- `plot_combined_subject_deltas.py`  
  Paired subject-delta figure for Galaxy and WESAD.
- `benchmark_watch_variant_latency.py`,
  `measure_deployment_efficiency.py`  
  CPU latency and deployment-path parameter counts.

## Dataset Loaders And Protocol Helpers

- `dataset.py`, `galaxy_dataset.py`, `wesad_dataset.py`  
  Dataset classes used by the training and analysis pipelines.
- `galaxy_protocols.py`, `protocols.py`  
  Session/protocol helpers.

## Standard Project Files

- `README.md`
- `REPRODUCE.md`
- `requirements.txt`
- `LICENSE`
