from __future__ import annotations

import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go


def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def diffusion_signal(x0: np.ndarray, A: np.ndarray, mu: float, steps: int) -> np.ndarray:
    x = np.array(x0, dtype=float)
    A = np.array(A, dtype=float)
    for _ in range(int(steps)):
        x = (1.0 - mu) * x + mu * (A @ x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -6.0, 6.0)
    return x


def main():
    st.set_page_config(page_title="Graph Diffusion Price Predictor", layout="wide")
    st.title("Graph diffusion → next-day price prediction")

    path = os.environ.get("DATA", "gd_data.npz")
    if not os.path.exists(path):
        st.error(f"Could not find {path}. Run export_gd_data.py first or set DATA=...")
        st.stop()

    data = load_npz(path)
    dates = [str(d) for d in data["dates"]]
    tickers = [str(t) for t in data["tickers"]]

    layout = data["layout"]     # (N,2)
    W = data["W"]               # (S,N,N)
    A = data["A"]               # (S,N,N)
    x0_all = data["x0"]         # (S,N)

    prices = data["prices"]
    prices_next = data["prices_next"]
    y_pred = data["y_pred"]
    y_true = data["y_true"]
    p_hat_next = data["p_hat_next"]

    mu0 = float(data.get("mu", 0.55))
    steps_max = int(data.get("steps", 6))

    N = len(tickers)
    S = len(dates)

    col_left, col_right = st.columns([1.1, 0.9])

    with col_right:
        st.subheader("Controls")
        target = st.selectbox("Target ticker", tickers, index=int(data.get("target_idx", 0)))
        tgt_idx = tickers.index(target)

        t = st.slider("Time index", min_value=0, max_value=S - 1, value=min(0, S - 1), step=1)
        s = st.slider("Diffusion steps", min_value=0, max_value=steps_max, value=0, step=1)

        mu = st.slider("μ (mixing)", min_value=0.0, max_value=1.0, value=float(mu0), step=0.01)

        edge_thr = st.slider("Edge threshold (draw)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        st.caption("Tip: drag time/step/μ to see diffusion update live.")

        if abs(mu - mu0) > 1e-6:
            st.info("Note: predictions shown below were exported with the original μ; changing μ only changes the diffusion visualization.")

        st.subheader("Prediction snapshot (exported model)")
        st.write(f"Date: **{dates[t]}**")
        st.metric("P(t)", f"{prices[t, tgt_idx]:.2f}")
        st.metric("P̂(t+1)", f"{p_hat_next[t]:.2f}")
        st.metric("P(t+1) actual", f"{prices_next[t, tgt_idx]:.2f}")
        st.metric("ŷ(t+1) logret", f"{y_pred[t, tgt_idx]:+.5f}")
        st.metric("r(t+1) logret", f"{y_true[t, tgt_idx]:+.5f}")

    with col_left:
        st.subheader("Graph + diffused signal")
        sig = diffusion_signal(x0_all[t], A[t], mu=mu, steps=s)

        # Build edge traces (threshold on current W)
        Wt = W[t]
        xs = []
        ys = []
        for i in range(N):
            for j in range(i + 1, N):
                if float(Wt[i, j]) <= edge_thr:
                    continue
                xs += [layout[i, 0], layout[j, 0], None]
                ys += [layout[i, 1], layout[j, 1], None]

        edge_trace = go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(width=1),
            opacity=0.25,
            hoverinfo="skip",
        )

        node_trace = go.Scatter(
            x=layout[:, 0],
            y=layout[:, 1],
            mode="markers+text",
            text=tickers,
            textposition="bottom center",
            marker=dict(
                size=16,
                color=sig,
                colorscale="RdBu",
                reversescale=True,
                cmin=-2.5,
                cmax=2.5,
                line=dict(width=2, color="white"),
            ),
            hovertemplate="<b>%{text}</b><br>x=%{marker.color:.3f}<extra></extra>",
        )

        halo_trace = go.Scatter(
            x=[layout[tgt_idx, 0]],
            y=[layout[tgt_idx, 1]],
            mode="markers",
            marker=dict(size=26, color="rgba(0,0,0,0)", line=dict(width=3, color="lime")),
            hoverinfo="skip",
        )

        fig = go.Figure(data=[edge_trace, node_trace, halo_trace])
        fig.update_layout(
            height=550,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Target price over sample window")
        p = prices[:, tgt_idx]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(range(S)), y=p, mode="lines", name="P(t)"))
        fig2.add_trace(go.Scatter(x=[t], y=[p[t]], mode="markers", name="current"))
        fig2.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="time index",
            yaxis_title="price",
        )
        st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
