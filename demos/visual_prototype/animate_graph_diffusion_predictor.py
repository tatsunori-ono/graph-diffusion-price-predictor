from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
from manim import (
    Scene,
    VGroup,
    Circle,
    Line,
    Text,
    Rectangle,
    DecimalNumber,
    ValueTracker,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    WHITE,
    BLUE,
    RED,
    YELLOW,
    GREEN,
    GREY_B,
    GREY_D,
    FadeIn,
    Transform,
    AnimationGroup,
    rate_functions,
    interpolate_color,
)


def load_npz() -> dict:
    path = os.environ.get("DATA", "gd_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Set DATA=... or run export_gd_data.py --out gd_data.npz"
        )
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def val_to_color(v: float, vmin: float = -2.5, vmax: float = 2.5):
    """Map a scalar to a diverging BLUE->WHITE->RED color."""
    v = clamp(v, vmin, vmax)
    if v < 0:
        a = (v - vmin) / (0 - vmin + 1e-12)  # 0..1
        return interpolate_color(BLUE, WHITE, a)
    else:
        a = (v - 0) / (vmax - 0 + 1e-12)
        return interpolate_color(WHITE, RED, a)


@dataclass
class NodeMobs:
    circle: Circle
    label: Text


@dataclass
class EdgeMobs:
    line: Line
    i: int
    j: int


class GraphDiffusionPricePredictor(Scene):
    """
    Graph diffusion (message passing) + next-day price prediction.

    The diffusion update is computed *inside* the animation:
        x_{s+1} = (1-μ)x_s + μ A x_s

    Environment variables:
      DATA        : path to gd_data.npz (default: gd_data.npz)
      MAX_FRAMES  : limit number of time frames (default: 25)
      EDGE_THR    : threshold for drawing edges based on average adjacency (default: 0.25)
    """

    def construct(self):
        data = load_npz()
        dates = list(data["dates"])
        tickers = list(data["tickers"])
        target = str(data["target"].item()) if hasattr(data["target"], "item") else str(data["target"])
        tgt_idx = int(data["target_idx"])
        layout = data["layout"]        # (N,2)

        W = data["W"]                  # (S,N,N) for edge opacity updates
        A = data["A"]                  # (S,N,N) row-stochastic diffusion operator
        x0_all = data["x0"]            # (S,N) initial cross-sectional signal

        prices = data["prices"]        # (S,N)
        prices_next = data["prices_next"]  # (S,N)
        y_pred = data["y_pred"]        # (S,N) precomputed walk-forward ridge predictions
        y_true = data["y_true"]        # (S,N)
        p_hat_next = data["p_hat_next"]  # (S,) target-only predicted next-day price (based on y_pred)

        anim_idx = data.get("anim_idx", np.arange(len(dates), dtype=int))
        mu = float(data.get("mu", 0.55))
        steps = int(data.get("steps", 6))

        max_frames = int(os.environ.get("MAX_FRAMES", "25"))
        edge_thr = float(os.environ.get("EDGE_THR", "0.02"))

        # Use only exported animation indices, possibly truncated
        frame_indices = [int(i) for i in anim_idx.tolist()]
        if max_frames > 0:
            frame_indices = frame_indices[: max_frames]

        N = len(tickers)

        # --- Layout constants ---
        graph_center = LEFT * 3.0 + DOWN * 1.3
        graph_scale = 2.0
        node_r = 0.15
        label_fs = 14
        edge_base_opacity = 0.45

        # --- Title / HUD ---
        title = Text("Graph diffusion → next-day price prediction", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.2)

        date_text = Text("", font_size=24, color=WHITE)
        date_text.next_to(title, DOWN, buff=0.15)

        self.add(title, date_text)

        # --- Pseudocode block (highlight line-by-line) ---
        code_lines = [
            "1  corr_t  <- corr(returns[t-W:t])",
            "2  W_t     <- threshold(|corr_t|)",
            "3  A_t     <- row_normalize(W_t)",
            "4  x0      <- zscore(r_t)",
            "5  diffuse <- x[s+1]=(1-u)x[s]+u*A*x[s]",
            "6  feats   <- [x0, x_S, x0-x_S]",
            "7  y_hat   <- Ridge(feats)",
            "8  P_hat   <- P(t)*exp(y_hat)",
        ]
        code_mobs = VGroup(*[Text(s, font_size=17, color=GREY_B) for s in code_lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        code_mobs.to_edge(RIGHT, buff=0.4).to_edge(UP, buff=1.0)
        code_box = Rectangle(width=code_mobs.width + 0.4, height=code_mobs.height + 0.3).set_stroke(GREY_D, width=2)
        code_box.move_to(code_mobs)
        self.add(code_box, code_mobs)

        hl = Rectangle(width=code_box.width - 0.1, height=0.32).set_stroke(YELLOW, width=3)
        hl.set_fill(opacity=0.0)
        hl.move_to(code_mobs[0])
        self.add(hl)

        # --- Build node mobjects ---
        node_mobs: List[NodeMobs] = []
        node_group = VGroup()
        for i in range(N):
            pos = layout[i]
            x = graph_center[0] + graph_scale * float(pos[0])
            y = graph_center[1] + graph_scale * float(pos[1])
            c = Circle(radius=node_r).set_stroke(WHITE, width=2).set_fill(WHITE, opacity=0.85)
            c.move_to([x, y, 0])
            lab = Text(tickers[i], font_size=label_fs, color=WHITE)
            lab.next_to(c, DOWN, buff=0.08)
            node_mobs.append(NodeMobs(circle=c, label=lab))
            node_group.add(c, lab)

        # --- Build edges from average adjacency (stable topology) ---
        W_avg = np.mean(W, axis=0)
        edges: List[EdgeMobs] = []
        edge_group = VGroup()
        for i in range(N):
            for j in range(i + 1, N):
                w = float(W_avg[i, j])
                if w <= edge_thr:
                    continue
                li = node_mobs[i].circle.get_center()
                lj = node_mobs[j].circle.get_center()
                ln = Line(li, lj).set_stroke(BLUE, width=3, opacity=edge_base_opacity)
                edge_group.add(ln)
                edges.append(EdgeMobs(line=ln, i=i, j=j))

        halo = Circle(radius=node_r * 1.5).set_stroke(GREEN, width=4)
        halo.move_to(node_mobs[tgt_idx].circle.get_center())

        self.play(FadeIn(edge_group, shift=0.1 * DOWN, run_time=0.8), FadeIn(node_group, shift=0.1 * DOWN, run_time=0.8))
        self.play(FadeIn(halo, run_time=0.4))

        # --- Prediction panel (upper-left, below date) ---
        panel_title = Text(f"Target: {target}", font_size=20, color=WHITE)

        p_now_tracker = ValueTracker(float(prices[frame_indices[0], tgt_idx]))
        p_pred_tracker = ValueTracker(float(p_hat_next[frame_indices[0]]))
        p_act_tracker = ValueTracker(float(prices_next[frame_indices[0], tgt_idx]))

        def make_row(label: str, tracker: ValueTracker, color=WHITE):
            lbl = Text(label, font_size=16, color=GREY_B)
            num = DecimalNumber(tracker.get_value(), num_decimal_places=2, color=color, font_size=16)
            num.add_updater(lambda m: m.set_value(tracker.get_value()))
            return VGroup(lbl, num).arrange(RIGHT, buff=0.2)

        row_now = make_row("P(t):", p_now_tracker, color=WHITE)
        row_pred = make_row("P̂(t+1):", p_pred_tracker, color=YELLOW)
        row_act = make_row("P(t+1):", p_act_tracker, color=WHITE)

        row_group = VGroup(row_now, row_pred, row_act).arrange(DOWN, aligned_edge=LEFT, buff=0.14)

        r_pred_tracker = ValueTracker(float(y_pred[frame_indices[0], tgt_idx]))
        r_act_tracker = ValueTracker(float(y_true[frame_indices[0], tgt_idx]))
        row_rpred = make_row("ŷ(t+1) logret:", r_pred_tracker, color=YELLOW)
        row_ract = make_row("r(t+1) logret:", r_act_tracker, color=WHITE)
        r_group = VGroup(row_rpred, row_ract).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        pred_panel = VGroup(panel_title, row_group, r_group).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        pred_panel.to_edge(LEFT, buff=0.5).shift(UP * 1.6)

        self.add(pred_panel)

        # --- Mini bar chart for x signal (lower right, below pseudocode) ---
        bars_w = 3.6
        bars_h = 1.8
        bars_box = Rectangle(width=bars_w, height=bars_h).set_stroke(GREY_D, width=2)
        bars_box.next_to(code_box, DOWN, buff=0.5).align_to(code_box, LEFT)
        # Clamp to stay within frame
        if bars_box.get_bottom()[1] < -3.8:
            bars_box.shift(UP * ((-3.8) - bars_box.get_bottom()[1]))
        bars_title = Text("Node signal (x)", font_size=18, color=WHITE)
        bars_title.next_to(bars_box, UP, buff=0.12)

        bar_max = 2.5
        bar_left = bars_box.get_left()[0] + 0.15
        bar_top = bars_box.get_top()[1] - 0.15
        vis_n = min(N, 12)
        bar_row_h = (bars_h - 0.3) / vis_n

        bar_labels: List[Text] = []
        bar_rects: List[Rectangle] = []
        for k in range(vis_n):
            lab = Text(tickers[k], font_size=14, color=GREY_B)
            lab.move_to([bar_left + 0.35, bar_top - k * bar_row_h, 0]).align_to(bars_box, LEFT)
            rect = Rectangle(width=0.01, height=bar_row_h * 0.55).set_fill(WHITE, opacity=0.9).set_stroke(width=0)
            rect.move_to([bar_left + 1.05, bar_top - k * bar_row_h, 0]).align_to(lab, DOWN)
            rect.align_to(lab, LEFT).shift(RIGHT * 0.75)
            bar_labels.append(lab)
            bar_rects.append(rect)

        self.add(bars_box, bars_title, *bar_labels, *bar_rects)

        def update_bars(values: np.ndarray):
            vals = np.array(values[:vis_n], dtype=float)
            vals = np.clip(vals, -bar_max, bar_max)
            for k in range(vis_n):
                v = float(vals[k])
                w = (abs(v) / bar_max) * (bars_w - 1.6)
                w = max(0.02, w)
                rect = bar_rects[k]
                rect.stretch_to_fit_width(w)
                rect.set_fill(val_to_color(v), opacity=0.95)
                rect.align_to(bar_labels[k], LEFT).shift(RIGHT * 0.75)

        # --- Helpers to apply visual style ---
        def apply_edge_opacity(frame_i: int):
            Wt = W[frame_i]
            for e in edges:
                w = float(Wt[e.i, e.j])
                strength = clamp(w, 0.0, 1.0)
                op = edge_base_opacity + 0.55 * strength
                ew = 2.0 + 3.0 * strength
                col = interpolate_color(BLUE, YELLOW, strength)
                e.line.set_stroke(color=col, width=ew, opacity=op)

        def apply_node_colors(sig: np.ndarray):
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            for i in range(N):
                node_mobs[i].circle.set_fill(val_to_color(float(sig[i])), opacity=0.95)
            update_bars(sig)

        # initial style
        apply_edge_opacity(frame_indices[0])
        apply_node_colors(x0_all[frame_indices[0]])

        # --- Main loop ---
        for t_count, idx in enumerate(frame_indices):
            # Update date
            new_date = Text(str(dates[idx]), font_size=24, color=WHITE).next_to(title, DOWN, buff=0.15)
            self.play(Transform(date_text, new_date), run_time=0.25)

            # highlight 1-4 (corr, W, A, x0)
            self.play(hl.animate.move_to(code_mobs[0]), run_time=0.16)
            self.play(hl.animate.move_to(code_mobs[1]), run_time=0.16)
            self.play(hl.animate.move_to(code_mobs[2]), run_time=0.16)
            self.play(hl.animate.move_to(code_mobs[3]), run_time=0.16)

            # Update edges for this time
            apply_edge_opacity(idx)

            # Diffusion steps computed live
            self.play(hl.animate.move_to(code_mobs[4]), run_time=0.22)

            x = np.array(x0_all[idx], dtype=float)
            apply_node_colors(x)
            self.wait(0.05)

            A_t = np.array(A[idx], dtype=float)
            for _s in range(steps):
                # pulse target node outline
                tgt = node_mobs[tgt_idx].circle
                self.play(
                    tgt.animate.set_stroke(YELLOW, width=4),
                    run_time=0.08,
                    rate_func=rate_functions.ease_in_out_sine,
                )

                # compute diffusion update
                x = (1.0 - mu) * x + mu * (A_t @ x)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                x = np.clip(x, -6.0, 6.0)

                apply_node_colors(x)

                self.play(
                    tgt.animate.set_stroke(WHITE, width=2),
                    run_time=0.08,
                    rate_func=rate_functions.ease_in_out_sine,
                )
                self.wait(0.03)

            # highlight 6-8 (features, ridge, price)
            self.play(hl.animate.move_to(code_mobs[5]), run_time=0.18)
            self.play(hl.animate.move_to(code_mobs[6]), run_time=0.18)
            self.play(hl.animate.move_to(code_mobs[7]), run_time=0.18)

            # Update displayed prediction numbers (precomputed ridge output)
            p_now_tracker.set_value(float(prices[idx, tgt_idx]))
            p_pred_tracker.set_value(float(p_hat_next[idx]))
            p_act_tracker.set_value(float(prices_next[idx, tgt_idx]))
            r_pred_tracker.set_value(float(y_pred[idx, tgt_idx]))
            r_act_tracker.set_value(float(y_true[idx, tgt_idx]))

            self.play(halo.animate.set_stroke(YELLOW, width=6), run_time=0.12)
            self.play(halo.animate.set_stroke(GREEN, width=4), run_time=0.12)

            self.wait(0.14)

        end_text = Text("End of clip (adjust MAX_FRAMES / date range to render more)", font_size=22, color=GREY_B)
        end_text.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(end_text, run_time=0.5))
        self.wait(1.0)
