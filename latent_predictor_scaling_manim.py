"""
Manim animation: scaling dynamics for latent predictors
=======================================================

This scene visualizes the central scaling point in the paper
"Correcting Bias when Using Latent Regressors":

    tilde X = X + U
    hat X   = (tilde X - mean(tilde X)) / sd(tilde X)
    Cor(hat X_1, hat X_2) = 1 / (1 + sigma_U^2)
    plim beta_hat_hatX = sqrt(Cor(hat X_1, hat X_2)) * beta_X
    beta_star = beta_hat_hatX / sqrt(Cor(hat X_1, hat X_2))

Target renderer: Manim Community Edition.

Install:
    python -m pip install manim numpy

Preview render:
    manim -pql latent_predictor_scaling_manim.py LatentPredictorScalingDynamics

High-quality MP4:
    manim -pqh latent_predictor_scaling_manim.py LatentPredictorScalingDynamics

GIF:
    manim -pqm --format gif latent_predictor_scaling_manim.py LatentPredictorScalingDynamics

Notes:
    * This script uses MathTex, so a working LaTeX installation is recommended.
    * The simulated dots are purely illustrative; the equations drive the animation.
"""

from __future__ import annotations

from math import sqrt
from typing import Callable

import numpy as np
from manim import *


# 16:9 canvas. Resolution/FPS are still controlled by Manim CLI quality flags.
config.frame_width = 16
config.frame_height = 9
config.background_color = "#101218"


class LatentPredictorScalingDynamics(Scene):
    """One continuous animation explaining latent-predictor rescaling."""

    def construct(self) -> None:
        self.camera.background_color = "#101218"

        # A single tracker drives all moving quantities.
        sigma2 = ValueTracker(0.15)

        self.opening_frame()
        self.point_cloud_scaling(sigma2)
        self.attenuation_curves(sigma2)
        self.split_indicator_correction(sigma2)
        self.closing_frame()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def section_title(self, text: str) -> Text:
        return Text(text, font_size=34, weight="BOLD", color=WHITE).to_edge(UP, buff=0.35)

    def make_equation(self, tex: str, font_size: int = 36, color=WHITE) -> MathTex:
        return MathTex(tex, font_size=font_size, color=color)

    def labeled_decimal(
        self,
        label_tex: str,
        value_func: Callable[[], float],
        color=WHITE,
        decimals: int = 2,
        font_size: int = 30,
    ) -> VGroup:
        label = MathTex(label_tex, font_size=font_size, color=color)
        number = DecimalNumber(
            value_func(),
            num_decimal_places=decimals,
            font_size=font_size,
            color=color,
        )
        number.add_updater(lambda m: m.set_value(value_func()))
        return VGroup(label, number).arrange(RIGHT, buff=0.12)

    def fade_out_all_visible(self) -> None:
        visible = [m for m in self.mobjects if not isinstance(m, ValueTracker)]
        if visible:
            self.play(*[FadeOut(m) for m in visible], run_time=0.8)
        self.clear()

    # ------------------------------------------------------------------
    # 1. Opening frame
    # ------------------------------------------------------------------
    def opening_frame(self) -> None:
        title = Text(
            "Latent predictors: measurement error plus identification",
            font_size=40,
            weight="BOLD",
            color=WHITE,
        ).to_edge(UP, buff=0.6)

        equations = VGroup(
            self.make_equation(r"\tilde X = X + U", 44, ORANGE),
            self.make_equation(
                r"\hat X = \frac{\tilde X - \overline{\tilde X}}{\operatorname{sd}(\tilde X)}",
                44,
                BLUE_B,
            ),
            self.make_equation(r"\operatorname{Var}(X)=1 \quad \text{and} \quad \operatorname{Var}(\hat X)=1", 36),
        ).arrange(DOWN, buff=0.45)
        equations.next_to(title, DOWN, buff=0.85)

        note = Text(
            "The noisy estimate spreads out, then identification rescales it back to unit variance.",
            font_size=26,
            color=GRAY_A,
        ).next_to(equations, DOWN, buff=0.7)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.8)
        self.play(Write(equations[0]), run_time=0.9)
        self.play(Write(equations[1]), run_time=1.1)
        self.play(FadeIn(equations[2], shift=UP * 0.1), FadeIn(note), run_time=0.9)
        self.wait(0.8)
        self.play(FadeOut(VGroup(title, equations, note)), run_time=0.8)

    # ------------------------------------------------------------------
    # 2. Animate X -> tilde X -> hat X using point clouds
    # ------------------------------------------------------------------
    def point_cloud_scaling(self, sigma2: ValueTracker) -> None:
        sigma2.set_value(0.15)
        title = self.section_title("1. Noisy latent estimates are re-scaled for identification")

        rng = np.random.default_rng(12)
        n = 85
        # Sort for a calmer visual; clip prevents extreme points from leaving the frame.
        x_true = np.sort(np.clip(rng.normal(0, 1, n), -3.6, 3.6))
        u_unit = rng.normal(0, 1, n)

        row_y = {"X": 1.35, "tilde": -0.15, "hat": -1.65}

        def make_line(y: float) -> NumberLine:
            return NumberLine(
                x_range=[-4, 4, 1],
                length=10.3,
                color=GRAY_B,
                include_tip=False,
            ).move_to([0.6, y, 0])

        lines = {
            "X": make_line(row_y["X"]),
            "tilde": make_line(row_y["tilde"]),
            "hat": make_line(row_y["hat"]),
        }

        row_labels = VGroup(
            MathTex(r"X", font_size=42, color=GREEN_B).next_to(lines["X"], LEFT, buff=0.55),
            MathTex(r"\tilde X = X + U", font_size=38, color=ORANGE).next_to(lines["tilde"], LEFT, buff=0.55),
            MathTex(r"\hat X = \tilde X / \sqrt{1+\sigma_U^2}", font_size=36, color=BLUE_B).next_to(
                lines["hat"], LEFT, buff=0.55
            ),
        )

        def dots_for(line: NumberLine, values: np.ndarray, color) -> VGroup:
            return VGroup(
                *[
                    Dot(
                        line.n2p(float(v)),
                        radius=0.032,
                        color=color,
                        fill_opacity=0.72,
                        stroke_width=0,
                    )
                    for v in np.clip(values, -4, 4)
                ]
            )

        true_cloud = dots_for(lines["X"], x_true, GREEN_B)

        noisy_cloud = always_redraw(
            lambda: dots_for(
                lines["tilde"],
                x_true + u_unit * sqrt(max(sigma2.get_value(), 0.0)),
                ORANGE,
            )
        )

        identified_cloud = always_redraw(
            lambda: dots_for(
                lines["hat"],
                (x_true + u_unit * sqrt(max(sigma2.get_value(), 0.0)))
                / sqrt(1.0 + max(sigma2.get_value(), 0.0)),
                BLUE_B,
            )
        )

        # Dynamic readouts.
        sigma_readout = self.labeled_decimal(r"\sigma_U^2 =", lambda: sigma2.get_value(), YELLOW, 2, 32)
        sd_readout = self.labeled_decimal(
            r"\operatorname{sd}(\tilde X)=\sqrt{1+\sigma_U^2}=",
            lambda: sqrt(1.0 + sigma2.get_value()),
            ORANGE,
            2,
            30,
        )
        factor_readout = self.labeled_decimal(
            r"\text{rescale by }1/\sqrt{1+\sigma_U^2}=",
            lambda: 1.0 / sqrt(1.0 + sigma2.get_value()),
            BLUE_B,
            2,
            30,
        )
        readouts = VGroup(sigma_readout, sd_readout, factor_readout).arrange(
            DOWN, aligned_edge=LEFT, buff=0.25
        )
        readouts.to_corner(UR, buff=0.45).shift(DOWN * 0.65)

        arrows = VGroup(
            Arrow(lines["X"].get_bottom(), lines["tilde"].get_top(), buff=0.12, color=GRAY_A, max_tip_length_to_length_ratio=0.16),
            Arrow(lines["tilde"].get_bottom(), lines["hat"].get_top(), buff=0.12, color=GRAY_A, max_tip_length_to_length_ratio=0.16),
        ).shift(LEFT * 5.1)

        captions = VGroup(
            Text("add noise", font_size=22, color=GRAY_A).next_to(arrows[0], LEFT, buff=0.1),
            Text("identify scale", font_size=22, color=GRAY_A).next_to(arrows[1], LEFT, buff=0.1),
        )

        scene_group = VGroup(title, *lines.values(), row_labels, true_cloud, noisy_cloud, identified_cloud, readouts, arrows, captions)

        self.play(FadeIn(title), run_time=0.6)
        self.play(
            LaggedStart(
                *[Create(lines[k]) for k in ["X", "tilde", "hat"]],
                FadeIn(row_labels),
                lag_ratio=0.15,
            ),
            run_time=1.2,
        )
        self.play(FadeIn(true_cloud), FadeIn(noisy_cloud), FadeIn(identified_cloud), run_time=0.9)
        self.play(FadeIn(readouts), Create(arrows), FadeIn(captions), run_time=0.9)
        self.wait(0.3)
        self.play(sigma2.animate.set_value(2.8), run_time=5.0, rate_func=smooth)
        self.play(sigma2.animate.set_value(0.55), run_time=2.6, rate_func=smooth)
        self.wait(0.5)
        self.play(FadeOut(scene_group), run_time=0.9)
        self.clear()

    # ------------------------------------------------------------------
    # 3. Attenuation factors: classical vs latent predictor
    # ------------------------------------------------------------------
    def attenuation_curves(self, sigma2: ValueTracker) -> None:
        sigma2.set_value(0.05)
        title = self.section_title("2. Identification changes the attenuation factor")

        ax = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1.05, 0.25],
            x_length=8.9,
            y_length=5.15,
            tips=False,
            axis_config={"color": GRAY_B, "include_numbers": True, "font_size": 22},
        ).to_edge(LEFT, buff=0.8).shift(DOWN * 0.35)

        x_label = MathTex(r"\sigma_U^2", font_size=32, color=GRAY_A).next_to(ax.x_axis.get_end(), RIGHT, buff=0.25)
        y_label = Text("slope multiplier", font_size=24, color=GRAY_A).rotate(PI / 2).next_to(
            ax.y_axis, LEFT, buff=0.35
        )

        classical_curve = ax.plot(lambda t: 1.0 / (1.0 + t), x_range=[0, 4], color=RED_B, stroke_width=6)
        latent_curve = ax.plot(lambda t: 1.0 / np.sqrt(1.0 + t), x_range=[0, 4], color=BLUE_B, stroke_width=6)

        classical_label = MathTex(
            r"\mathrm{classical:}\ (1+\sigma_U^2)^{-1}",
            font_size=31,
            color=RED_B,
        ).next_to(ax, RIGHT, buff=0.7).shift(UP * 1.35)
        latent_label = MathTex(
            r"\mathrm{latent:}\ (1+\sigma_U^2)^{-1/2}",
            font_size=31,
            color=BLUE_B,
        ).next_to(classical_label, DOWN, aligned_edge=LEFT, buff=0.35)

        explainer = Text(
            "With identified latent scores, variance is forced back to 1;\nonly the covariance loss remains.",
            font_size=23,
            line_spacing=0.95,
            color=GRAY_A,
        ).next_to(latent_label, DOWN, aligned_edge=LEFT, buff=0.6)

        vertical_line = always_redraw(
            lambda: DashedLine(
                ax.c2p(sigma2.get_value(), 0),
                ax.c2p(sigma2.get_value(), 1.03),
                dash_length=0.08,
                color=GRAY_C,
            )
        )
        classical_dot = always_redraw(
            lambda: Dot(
                ax.c2p(sigma2.get_value(), 1.0 / (1.0 + sigma2.get_value())),
                radius=0.07,
                color=RED_B,
            )
        )
        latent_dot = always_redraw(
            lambda: Dot(
                ax.c2p(sigma2.get_value(), 1.0 / sqrt(1.0 + sigma2.get_value())),
                radius=0.07,
                color=BLUE_B,
            )
        )

        sigma_readout = self.labeled_decimal(r"\sigma_U^2=", lambda: sigma2.get_value(), YELLOW, 2, 30)
        classical_readout = self.labeled_decimal(
            r"\lambda_{\mathrm{classical}}=",
            lambda: 1.0 / (1.0 + sigma2.get_value()),
            RED_B,
            2,
            30,
        )
        latent_readout = self.labeled_decimal(
            r"\lambda_{\mathrm{latent}}=",
            lambda: 1.0 / sqrt(1.0 + sigma2.get_value()),
            BLUE_B,
            2,
            30,
        )
        readouts = VGroup(sigma_readout, classical_readout, latent_readout).arrange(
            DOWN, aligned_edge=LEFT, buff=0.26
        )
        readouts.next_to(explainer, DOWN, aligned_edge=LEFT, buff=0.55)

        scene_group = VGroup(
            title,
            ax,
            x_label,
            y_label,
            classical_curve,
            latent_curve,
            classical_label,
            latent_label,
            explainer,
            vertical_line,
            classical_dot,
            latent_dot,
            readouts,
        )

        self.play(FadeIn(title), Create(ax), FadeIn(x_label), FadeIn(y_label), run_time=1.1)
        self.play(Create(classical_curve), FadeIn(classical_label), run_time=1.0)
        self.play(Create(latent_curve), FadeIn(latent_label), run_time=1.0)
        self.play(FadeIn(explainer), FadeIn(readouts), FadeIn(vertical_line), FadeIn(classical_dot), FadeIn(latent_dot), run_time=0.9)
        self.play(sigma2.animate.set_value(4.0), run_time=5.0, rate_func=smooth)
        self.play(sigma2.animate.set_value(1.0), run_time=2.0, rate_func=smooth)
        self.wait(0.5)
        self.play(FadeOut(scene_group), run_time=0.9)
        self.clear()

    # ------------------------------------------------------------------
    # 4. Split indicators estimate rho and correct beta
    # ------------------------------------------------------------------
    def split_indicator_correction(self, sigma2: ValueTracker) -> None:
        sigma2.set_value(0.2)
        title = self.section_title("3. Split indicators estimate the correction factor")

        equations = VGroup(
            self.make_equation(r"\tilde X_1=X+U_1,\qquad \tilde X_2=X+U_2", 34, WHITE),
            self.make_equation(
                r"\rho=\operatorname{Cor}(\hat X_1,\hat X_2)=\frac{1}{1+\sigma_U^2}",
                36,
                YELLOW,
            ),
            self.make_equation(r"\operatorname{plim}\,\hat\beta_{\hat X}=\sqrt{\rho}\,\beta_X", 36, BLUE_B),
            self.make_equation(r"\hat\beta^{*}=\frac{\hat\beta_{\hat X}}{\sqrt{\rho}}", 38, GREEN_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        equations.to_edge(LEFT, buff=0.75).shift(UP * 0.25)

        left_x = 0.7
        scale = 4.65
        bar_h = 0.42
        y_true, y_unadj, y_corr = 1.2, 0.25, -0.7

        baseline = VGroup(
            Line([left_x, y_true - 0.5, 0], [left_x + scale, y_true - 0.5, 0], color=GRAY_D),
            Line([left_x, y_unadj - 0.5, 0], [left_x + scale, y_unadj - 0.5, 0], color=GRAY_D),
            Line([left_x, y_corr - 0.5, 0], [left_x + scale, y_corr - 0.5, 0], color=GRAY_D),
        )

        def bar(value_func: Callable[[], float], y: float, color) -> Mobject:
            return always_redraw(
                lambda: Rectangle(
                    width=max(0.03, scale * value_func()),
                    height=bar_h,
                    stroke_color=color,
                    fill_color=color,
                    fill_opacity=0.78,
                    stroke_width=1.5,
                ).move_to([left_x + max(0.03, scale * value_func()) / 2, y, 0])
            )

        true_bar = bar(lambda: 1.0, y_true, GREEN_B)
        unadj_bar = bar(lambda: 1.0 / sqrt(1.0 + sigma2.get_value()), y_unadj, BLUE_B)
        corrected_bar = bar(lambda: 1.0, y_corr, GREEN_C)

        bar_labels = VGroup(
            Text("target slope", font_size=24, color=GREEN_B).next_to([left_x, y_true, 0], LEFT, buff=0.35),
            Text("unadjusted", font_size=24, color=BLUE_B).next_to([left_x, y_unadj, 0], LEFT, buff=0.35),
            Text("corrected", font_size=24, color=GREEN_C).next_to([left_x, y_corr, 0], LEFT, buff=0.35),
        )

        tick_labels = VGroup(
            Text("0", font_size=20, color=GRAY_A).next_to([left_x, y_corr - 1.05, 0], DOWN, buff=0.05),
            Text("1", font_size=20, color=GRAY_A).next_to([left_x + scale, y_corr - 1.05, 0], DOWN, buff=0.05),
        )
        tick_axis = NumberLine(x_range=[0, 1, 0.25], length=scale, include_tip=False, color=GRAY_C).move_to(
            [left_x + scale / 2, y_corr - 1.05, 0]
        )

        beta_values = VGroup(
            self.labeled_decimal(r"\rho=", lambda: 1.0 / (1.0 + sigma2.get_value()), YELLOW, 2, 30),
            self.labeled_decimal(r"\sqrt{\rho}=", lambda: 1.0 / sqrt(1.0 + sigma2.get_value()), BLUE_B, 2, 30),
            self.labeled_decimal(r"\hat\beta^{*}/\beta_X=", lambda: 1.0, GREEN_B, 2, 30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        beta_values.next_to(equations, DOWN, aligned_edge=LEFT, buff=0.65)

        bar_title = Text("Effect on slope magnitude", font_size=28, weight="BOLD", color=WHITE)
        bar_title.next_to([left_x + scale / 2, y_true + 0.55, 0], UP, buff=0.2)

        note = Text(
            "As measurement error grows, the unadjusted slope shrinks.\nThe split-correlation correction restores the latent-scale slope.",
            font_size=22,
            color=GRAY_A,
            line_spacing=0.95,
        ).next_to(tick_axis, DOWN, buff=0.35)

        scene_group = VGroup(
            title,
            equations,
            beta_values,
            baseline,
            true_bar,
            unadj_bar,
            corrected_bar,
            bar_labels,
            tick_axis,
            tick_labels,
            bar_title,
            note,
        )

        self.play(FadeIn(title), FadeIn(equations, shift=UP * 0.15), run_time=1.0)
        self.play(FadeIn(beta_values), run_time=0.7)
        self.play(
            FadeIn(bar_title),
            Create(tick_axis),
            FadeIn(tick_labels),
            FadeIn(baseline),
            FadeIn(bar_labels),
            run_time=0.9,
        )
        self.play(FadeIn(true_bar), FadeIn(unadj_bar), FadeIn(corrected_bar), FadeIn(note), run_time=0.9)
        self.play(sigma2.animate.set_value(3.6), run_time=5.0, rate_func=smooth)
        self.play(sigma2.animate.set_value(0.35), run_time=2.2, rate_func=smooth)
        self.wait(0.5)
        self.play(FadeOut(scene_group), run_time=0.9)
        self.clear()

    # ------------------------------------------------------------------
    # 5. Closing frame
    # ------------------------------------------------------------------
    def closing_frame(self) -> None:
        title = Text("Takeaway", font_size=44, weight="BOLD", color=WHITE).to_edge(UP, buff=0.65)

        bullets = VGroup(
            Text("1. Latent estimates are noisy measurements of an identified trait.", font_size=29, color=GRAY_A),
            Text("2. Standardization changes the measurement-error attenuation factor.", font_size=29, color=GRAY_A),
            Text("3. Split indicators estimate reliability, so slopes can be corrected by 1/sqrt(rho).", font_size=29, color=GRAY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.38)
        bullets.next_to(title, DOWN, buff=0.8).to_edge(LEFT, buff=1.25)

        final_eq = MathTex(
            r"\rho=\operatorname{Cor}(\hat X_1,\hat X_2),\qquad "
            r"\hat\beta^{*}=\frac{\hat\beta_{\hat X}}{\sqrt{\rho}}",
            font_size=42,
            color=GREEN_B,
        ).next_to(bullets, DOWN, buff=0.75)

        self.play(FadeIn(title), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.1) for b in bullets], lag_ratio=0.2), run_time=1.4)
        self.play(Write(final_eq), run_time=1.2)
        self.wait(1.5)
