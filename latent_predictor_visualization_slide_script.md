# Slide-by-slide script: Latent predictor scaling dynamics

**Use case:** Voiceover/storyboard script for the Manim visualization exported as `LatentPredictorScalingDynamics`.

**Recommended pacing:** about 60-90 seconds total. Slides 2-4 can run longer if the animation is rendered as a full explainer video.

## Slide 1 - The setup: latent predictors need two steps

**Purpose:** Introduce the core problem: the predictor is latent, so we first estimate it with error and then impose an identifying scale.

**Visual direction:**
- Dark background with title: "Latent predictors: measurement error plus identification."
- Equations appear one at a time: tilde X = X + U; hat X = (tilde X - mean(tilde X)) / sd(tilde X); Var(X)=1 and Var(hat X)=1.
- Subtle note at bottom: "The noisy estimate spreads out, then identification rescales it back to unit variance."

**On-screen text/equations:**
- `Latent predictors: measurement error plus identification`
- `tilde X = X + U`
- `hat X = (tilde X - mean(tilde X)) / sd(tilde X)`
- `Var(X)=1 and Var(hat X)=1`

**Voiceover:**
Many quantities we care about in political science are latent: ideology, knowledge, democracy, sentiment, and many others. We do not observe X directly. Instead, we estimate it from indicators, so the first object we get is a noisy estimate, tilde X equals X plus U. But latent variables also require identification. In practice, we usually put both the true latent trait and the estimated score on a mean-zero, unit-variance scale. That second step matters. The estimate is noisy, but after standardization it has the same variance as the target latent trait.

**Transition:** Move from equations to a visual point cloud showing what noise and standardization do to the distribution.

## Slide 2 - Scaling dynamics: noise spreads, identification compresses

**Purpose:** Show the mechanics: X is measured with noise; tilde X has too much spread; hat X is rescaled to unit variance.

**Visual direction:**
- Three horizontal number lines stacked vertically: X, tilde X = X + U, and hat X = tilde X / sqrt(1 + sigma_U^2).
- Dots on the first line show the true latent trait. Dots on the second line spread outward as sigma_U^2 increases. Dots on the third line are pulled back by the identifying rescale.
- Readouts update dynamically: sigma_U^2, sd(tilde X)=sqrt(1+sigma_U^2), and rescale by 1/sqrt(1+sigma_U^2).

**On-screen text/equations:**
- `1. Noisy latent estimates are re-scaled for identification`
- `X`
- `tilde X = X + U`
- `hat X = tilde X / sqrt(1 + sigma_U^2)`
- `add noise -> identify scale`

**Voiceover:**
Here is the intuition. The top row is the true latent variable, X, identified to have variance one. When measurement error is added, the noisy estimate tilde X spreads out. Its standard deviation is sqrt one plus sigma_U squared. But the score we use in the regression is not usually left on that wider scale. We standardize it, producing hat X. So as the noise grows, the middle row gets wider, but the bottom row is compressed back to the identified latent scale. This is why the latent-predictor case is not the same as ordinary classical measurement error.

**Transition:** Use this scaling intuition to compare the classical and latent attenuation factors.

## Slide 3 - Why the attenuation factor changes

**Purpose:** Contrast the classical measurement-error attenuation factor with the latent-predictor attenuation factor.

**Visual direction:**
- Plot with sigma_U^2 on the x-axis and slope multiplier on the y-axis.
- Red curve: classical attenuation, (1 + sigma_U^2)^(-1).
- Blue curve: latent-predictor attenuation, (1 + sigma_U^2)^(-1/2).
- Animated vertical line moves as sigma_U^2 increases, with dots tracking both curves.

**On-screen text/equations:**
- `2. Identification changes the attenuation factor`
- `classical: (1 + sigma_U^2)^(-1)`
- `latent: (1 + sigma_U^2)^(-1/2)`
- `With identified latent scores, variance is forced back to 1; only the covariance loss remains.`

**Voiceover:**
In the ordinary classical-error case, the observed predictor has inflated variance, so the slope is multiplied by one over one plus sigma_U squared. For an identified latent predictor, the variance inflation has already been removed by standardization. That changes the slope multiplier to one over the square root of one plus sigma_U squared. The two curves are not the same. The latent-predictor slope is still attenuated, but it is attenuated by the square-root factor because the scale has been reset.

**Transition:** Next, show how split indicators let us estimate the amount of attenuation directly.

## Slide 4 - Split indicators estimate reliability and correct the slope

**Purpose:** Explain the split-indicator correction: two independent estimates reveal measurement reliability, which gives the correction factor.

**Visual direction:**
- Left side shows four equations: tilde X_1 = X + U_1 and tilde X_2 = X + U_2; rho = Cor(hat X_1, hat X_2) = 1/(1+sigma_U^2); plim beta_hat_hatX = sqrt(rho) beta_X; beta_star = beta_hat_hatX / sqrt(rho).
- Right side shows three bars: target slope, unadjusted slope, and corrected slope.
- As measurement error grows, the unadjusted bar shrinks while the corrected bar stays aligned with the target slope.

**On-screen text/equations:**
- `3. Split indicators estimate the correction factor`
- `tilde X_1 = X + U_1, tilde X_2 = X + U_2`
- `rho = Cor(hat X_1, hat X_2) = 1 / (1 + sigma_U^2)`
- `plim beta_hat_hatX = sqrt(rho) beta_X`
- `beta_star = beta_hat_hatX / sqrt(rho)`

**Voiceover:**
The correction uses the fact that the same latent trait can often be estimated from two disjoint sets of indicators. Each split gives a noisy measurement of X. If the split-specific errors are independent, then the correlation between the two identified scores estimates reliability: rho equals one over one plus sigma_U squared. In the latent-predictor case, the unadjusted regression slope converges to sqrt rho times the true latent-scale slope. So the correction is simple: divide the unadjusted slope by sqrt rho. Visually, the blue bar shows attenuation. The corrected green bar restores the slope to the latent scale.

**Transition:** Close by summarizing the lesson for applied work with latent predictors.

## Slide 5 - Takeaway

**Purpose:** Summarize the key message and leave viewers with the correction formula.

**Visual direction:**
- Takeaway title with three bullets.
- Final equation displayed prominently: rho = Cor(hat X_1, hat X_2), beta_star = beta_hat_hatX / sqrt(rho).

**On-screen text/equations:**
- `Takeaway`
- `1. Latent estimates are noisy measurements of an identified trait.`
- `2. Standardization changes the measurement-error attenuation factor.`
- `3. Split indicators estimate reliability, so slopes can be corrected by 1/sqrt(rho).`
- `rho = Cor(hat X_1, hat X_2), beta_star = beta_hat_hatX / sqrt(rho)`

**Voiceover:**
The takeaway is that latent predictors combine two issues: measurement error and scale identification. Because the estimated latent score is standardized, standard errors-in-variables intuition does not carry over directly. But split indicators give a modular way to estimate reliability and correct attenuation. Compute rho as the correlation between the two split-based scores, then divide the unadjusted latent-predictor slope by sqrt rho.

**Transition:** End on the formula so the viewer remembers the operational correction.

## Optional short closing line

"Latent predictors are not just noisy variables; they are noisy variables on an identified scale. The correction has to respect both pieces."
