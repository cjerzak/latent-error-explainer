# Latent Predictor Scaling Dynamics — Manim Animation

This folder contains a Manim Community Edition script for animating the scaling dynamics discussed in **Correcting Bias when Using Latent Regressors**.

## Files

- `latent_predictor_scaling_manim.py` — the Manim scene code.

## Install

```bash
python -m pip install manim numpy
```

The script uses `MathTex`, so a working LaTeX installation is recommended for formula rendering.

## Render

Preview / fast render:

```bash
manim -pql latent_predictor_scaling_manim.py LatentPredictorScalingDynamics
```

High-quality MP4:

```bash
manim -pqh latent_predictor_scaling_manim.py LatentPredictorScalingDynamics
```

GIF:

```bash
manim -pqm --format gif latent_predictor_scaling_manim.py LatentPredictorScalingDynamics
```

## What the animation shows

1. `X` is the true latent trait with unit variance.
2. `\tilde X = X + U` spreads out as measurement-error variance increases.
3. `\hat X = \tilde X / sqrt(1 + sigma_U^2)` is rescaled for identification.
4. Classical attenuation is `(1 + sigma_U^2)^-1`, while latent-predictor attenuation is `(1 + sigma_U^2)^-1/2`.
5. Split-indicator reliability `rho = Cor(\hat X_1, \hat X_2)` yields the correction `beta* = beta_hat / sqrt(rho)`.
