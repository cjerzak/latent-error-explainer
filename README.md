# Latent Error Explainer

This repository contains a Manim animation that explains the core scaling idea in
[Attenuation Bias with Latent Predictors](https://arxiv.org/abs/2507.22218) by
Connor T. Jerzak and Stephen A. Jessee.

The paper studies what happens when a latent quantity, such as political
knowledge, ideology, democracy, sentiment, or another constructed trait, is
estimated from indicators and then used as a predictor in a regression. Because
the latent predictor is measured with error, the regression slope is generally
attenuated toward zero. The paper's central point is that standard
measurement-error fixes do not transfer mechanically to latent predictors
because latent scores are usually identified by a scale normalization, such as
mean zero and unit variance.

## What the Animation Shows

The animation focuses on the simplest version of the problem:

1. A true latent predictor `X` is standardized so that `Var(X) = 1`.
2. A noisy latent estimate is formed as `tilde X = X + U`.
3. The estimated latent score is identified by rescaling:

   ```text
   hat X = tilde X / sqrt(1 + sigma_U^2)
   ```

4. Classical measurement error attenuates a slope by
   `(1 + sigma_U^2)^-1`, but the identified latent-predictor case attenuates it
   by `(1 + sigma_U^2)^-1/2`.
5. If the indicators can be split into two independent halves, the split-score
   correlation estimates reliability:

   ```text
   rho = Cor(hat X_1, hat X_2) = 1 / (1 + sigma_U^2)
   ```

6. For the bivariate OLS case visualized here, the latent-scale slope can be
   corrected with:

   ```text
   beta_star = beta_hat / sqrt(rho)
   ```

The visual goal is to make the paper's identification point concrete: the noisy
estimate first spreads out, then latent-variable identification rescales it back
to unit variance. That rescaling changes the attenuation factor.

## Files

- `latent_predictor_scaling_manim.py`: Manim scene for the full animation.
- `latent_predictor_scaling_README.md`: Short rendering notes for the animation.
- `README.md`: This explainer and citation entry.

## Install

Install Manim Community Edition and NumPy:

```bash
python -m pip install manim numpy
```

The scene uses `MathTex`, so a working LaTeX installation is recommended.

## Render

Preview render:

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

Manim writes rendered outputs under `media/videos/`.

## Paper

Read the paper on arXiv:

- [Attenuation Bias with Latent Predictors](https://arxiv.org/abs/2507.22218)
- arXiv:2507.22218
- DOI: [10.48550/arXiv.2507.22218](https://doi.org/10.48550/arXiv.2507.22218)

The paper shows that common strategies such as instrumental variables and the
method of composition can themselves be biased when applied to latent regressors.
It proposes a modular split-sample correlation correction that can be used with
many latent-trait estimators, including additive scores, factor models, and
machine-learning models, without requiring joint estimation of the latent trait
and outcome model.

## Citation

```bibtex
@misc{jerzak2026attenuationbiaslatentpredictors,
      title={Attenuation Bias with Latent Predictors}, 
      author={Connor T. Jerzak and Stephen A. Jessee},
      year={2026},
      eprint={2507.22218},
      archivePrefix={arXiv},
      primaryClass={stat.AP},
      url={https://arxiv.org/abs/2507.22218}, 
}
```
