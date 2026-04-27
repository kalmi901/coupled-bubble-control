from __future__ import annotations
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from trajectory_reader import TrajectoryZarrReader

st.set_page_config(page_title="Zarr Explorer for Coupled Bubble Position Control", layout="wide")

# --- CSS ---
st.markdown(
    """
<style>
html, body, [class*="css"]  { font-size: 16px; }

.stButton button {
    font-size: 12px;
    width: 100%;
    padding: 0.25rem 0.4rem;
    margin-top: 0.2rem;
    margin-bottom: 0.2rem;
}

div[data-testid="stRadio"] > div[role="radiogroup"] {
    max-height: 220px;
    overflow-y: auto;
    padding: 0.25rem 0.5rem;
    border: 1px solid rgba(49,51,63,0.2);
    border-radius: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def meta_get(meta, key, default=None):
    """Works for dict-like or object-like metadata."""
    if meta is None:
        return default
    if isinstance(meta, dict):
        return meta.get(key, default)
    return getattr(meta, key, default)


def get_palette(name: str):
    palettes = {
        "Plotly": px.colors.qualitative.Plotly,
        "D3": px.colors.qualitative.D3,
        "Dark24": px.colors.qualitative.Dark24,
        "Set1": px.colors.qualitative.Set1,
    }
    return palettes.get(name, px.colors.qualitative.Plotly)


def apply_axis_style(fig: go.Figure, axis_title_size: int, tick_size: int):
    fig.update_layout(
        xaxis=dict(title_font=dict(size=axis_title_size), tickfont=dict(size=tick_size)),
        yaxis=dict(title_font=dict(size=axis_title_size), tickfont=dict(size=tick_size)),
        legend=dict(
            font=dict(size=tick_size),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=20, t=40, b=40),
    )


# -----------------------------
# UI Components
# -----------------------------
def folder_browser(label: str, start_path: str):
    start = Path(start_path).resolve()

    cur_key = f"{label}_current_dir"
    sel_key = f"{label}_selected_name"

    if cur_key not in st.session_state:
        st.session_state[cur_key] = str(start)
        st.session_state[sel_key] = None

    current_dir = Path(st.session_state[cur_key]).resolve()

    st.markdown(f"**Current folder:** `{current_dir}`")

    try:
        subdirs = [p for p in current_dir.iterdir() if p.is_dir()]
    except PermissionError:
        subdirs = []

    subdirs_sorted = sorted(subdirs, key=lambda p: p.name.lower())
    options = [p.name for p in subdirs_sorted]

    prev_selected = st.session_state.get(sel_key, None)

    if options:
        default_idx = options.index(prev_selected) if prev_selected in options else 0
        selected_name = st.radio("Folders", options=options, index=default_idx, key=f"{label}_radio")
        st.session_state[sel_key] = selected_name
    else:
        st.info("Nincs almappa ebben a mappában.")
        selected_name = None
        st.session_state[sel_key] = None

    col_up, col_open, col_load = st.columns(3)

    with col_up:
        if st.button("Up", key=f"{label}_up"):
            parent = current_dir.parent
            if parent.exists() and parent != current_dir:
                st.session_state[cur_key] = str(parent.resolve())
                st.session_state[sel_key] = None
                st.rerun()

    with col_open:
        if st.button("Open", key=f"{label}_open"):
            if selected_name is not None:
                new_dir = current_dir / selected_name
                if new_dir.exists() and new_dir.is_dir():
                    st.session_state[cur_key] = str(new_dir.resolve())
                    st.session_state[sel_key] = None
                    st.rerun()

    with col_load:
        if st.button("Load", key=f"{label}_load"):
            target = (current_dir / selected_name).resolve() if selected_name else current_dir
            try:
                reader = TrajectoryZarrReader(str(target))
                st.session_state["zarr_reader"] = reader
                st.session_state["zarr_path"] = str(target)
                st.session_state["zarr_meta"] = getattr(reader, "global_metadata", None)

                # Reset episode cache on new load
                st.session_state["episode_id_loaded"] = None
                st.session_state["episode_data"] = None

            except Exception as e:
                st.error(f"Hiba a TrajectoryZarrReader inicializálásakor: {e}")

    return current_dir, selected_name


def show_static_metadata(meta: dict):
    # meta lehet { "attributes": {...} } vagy maga az attrib dict
    attrs = meta.get("attributes", meta) if isinstance(meta, dict) else meta

    ac_type = meta_get(attrs, "ac_type", "N/A")
    k = int(meta_get(attrs, "k", 0) or 0)
    ref_freq = meta_get(attrs, "ref_freq", None)
    num_bubbles = int(meta_get(attrs, "num_bubbles", 0) or 0)
    dt = meta_get(attrs, "time_step_length", "N/A")
    pos_tol = meta_get(attrs, "position_tolerance", "N/A")

    ref_freq_txt = "N/A"
    if isinstance(ref_freq, (int, float)):
        ref_freq_txt = f"{ref_freq/1000:.2f} kHz"

    st.subheader("Constant Parameters")
    st.markdown(f"**Number of bubbles:** `{num_bubbles}`")
    st.markdown(f"**Acoustic field type:** `{ac_type}`")
    st.markdown(f"**Number of acoustic components (k):** `{k}`")
    st.markdown(f"**Time step length:** `{dt}`")
    st.markdown(f"**Reference frequency:** `{ref_freq_txt}`")
    st.markdown(f"**Position tolerance:** `{pos_tol}`")

    st.markdown("---")
    st.subheader("Acoustic Control Parameters")

    freqs = meta_get(attrs, "freq", []) or []
    PA_max = meta_get(attrs, "PA_max", []) or []
    PA_min = meta_get(attrs, "PA_min", []) or []
    PS_max = meta_get(attrs, "PS_max", []) or []
    PS_min = meta_get(attrs, "PS_min", []) or []

    if k == 0:
        st.info("No component metadata found.")
        return

    for i in range(k):
        f = freqs[i] if i < len(freqs) else None
        f_khz = f / 1000.0 if isinstance(f, (int, float)) else None

        pmax = PA_max[i] if i < len(PA_max) else None
        pmin = PA_min[i] if i < len(PA_min) else None
        psmax = PS_max[i] if i < len(PS_max) else None
        psmin = PS_min[i] if i < len(PS_min) else None

        st.markdown(rf"**Component {i}**")
        if f_khz is not None:
            st.markdown(rf"- $f_{i} = {f_khz:.2f}\,\text{{kHz}}$")
        if pmax is not None or pmin is not None:
            st.markdown(rf"- $P_{{A\min,{i}}} = {pmin}$ [bar],  $P_{{A\max,{i}}} = {pmax}$ [bar]")
        if psmax is not None or psmin is not None:
            st.markdown(rf"- $\theta_{{\min,{i}}} = {psmin}$ [rad],  $\theta_{{\max,{i}}} = {psmax}$ [rad]")


# -----------------------------
# Plotting
# -----------------------------
def plot_radius_time_curves(dense_time, dense_state_0, num_units, colors, axis_title_size, tick_size):
    st.markdown("### Radius–time curves")
    fig = go.Figure()
    for i in range(num_units):
        fig.add_trace(
            go.Scatter(
                x=dense_time.flatten(),
                y=dense_state_0[:, i],
                name=f"Bubble {i}",
                line=dict(color=colors[i]),
            )
        )
    fig.update_layout(xaxis_title="τ", yaxis_title="R/R₀", height=300 + 20 * num_units)
    apply_axis_style(fig, axis_title_size, tick_size)
    st.plotly_chart(fig, width='stretch')


def plot_position_time_curves(dense_time, dense_state_1, observations, num_units, colors, global_meta, axis_title_size, tick_size):
    st.markdown("### Position–time curves")
    fig = go.Figure()

    tol = meta_get(global_meta, "position_tolerance", 0.01) or 0.0

    # Defaults if not present
    xt_min = meta_get(global_meta, "XT_min", -0.5)
    xt_max = meta_get(global_meta, "XT_max", 0.5)

    x_min = meta_get(global_meta, "X_min", -0.5)
    x_max = meta_get(global_meta, "X_max", 0.5)

    # Draw tolerance bands + target lines
    for i in range(num_units):
        y0 = 0.5 * (observations[0, i] * (xt_max - xt_min) + (xt_max + xt_min)) #type: ignore

        fig.add_hrect(y0=y0 - tol, y1=y0 + tol, fillcolor=colors[i], opacity=0.12, line_width=0)
        fig.add_hline(y=y0, line_dash="dash", line_color=colors[i], opacity=0.6)

    # Draw Observations
    time_edges = [i * dt for i in range(timesteps + 1)]
    stacked_pos = (observations.shape[1] - num_units) // num_units
    for i in range(num_units):
        y = 0.5 * (observations[:, num_units + i * stacked_pos] * (x_max - x_min) + (x_max + x_min))    #type: ignore
        fig.add_trace(
            go.Scatter(
                x=time_edges,
                y=y,
                name=f"Observation {i}",
                mode="markers",
                marker=dict(size=10, color=colors[i]),
            )
        )

    # Trajectories
    for i in range(num_units):
        fig.add_trace(
            go.Scatter(
                x=dense_time.flatten(),
                y=dense_state_1[:, i],
                name=f"Bubble {i}",
                line=dict(color=colors[i]),
            )
        )

    fig.update_layout(xaxis_title="τ", yaxis_title="x/λ₀", height=300 + 20 * num_units)
    apply_axis_style(fig, axis_title_size, tick_size)
    st.plotly_chart(fig, width='stretch')


def plot_step_action_curves(title, y_label, actions_block, dt, timesteps, colors, axis_title_size, tick_size):
    st.markdown(f"### {title}")
    time_edges = [i * dt for i in range(timesteps + 1)]
    fig = go.Figure()

    k = actions_block.shape[1]
    for i in range(k):
        y = actions_block[:timesteps, i]
        y_edges = list(y) + [y[-1]]
        fig.add_trace(
            go.Scatter(
                x=time_edges,
                y=y_edges,
                name=f"Component {i}",
                mode="lines",
                line=dict(color=colors[i]),
                line_shape="hv",
            )
        )

    fig.update_layout(xaxis_title="τ", yaxis_title=y_label, height=300 + 20 * k)
    apply_axis_style(fig, axis_title_size, tick_size)
    st.plotly_chart(fig, width='stretch')


def plot_reward_time_curves(rewards, dt, timesteps, axis_title_size, tick_size):
    st.markdown("### Reward curves")
    r = np.asarray(rewards).flatten()[:timesteps]
    t = np.arange(len(r)) * dt
    rcum = np.cumsum(r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=r, name="reward", mode="markers", marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=t, y=rcum, name="cumulative reward", mode="lines"))

    fig.update_layout(xaxis_title="τ", yaxis_title="reward", height=320)
    apply_axis_style(fig, axis_title_size, tick_size)
    st.plotly_chart(fig, width='stretch')


# -----------------------------
# Session state init
# -----------------------------
for k in ["zarr_reader", "zarr_path", "zarr_meta", "episode_id_loaded", "episode_data"]:
    if k not in st.session_state:
        st.session_state[k] = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Zarr browser")
    current_dir, _ = folder_browser("zarr_browser", "./")

    if st.session_state["zarr_reader"] is not None:
        st.markdown("---")
        st.success(f"Trajectory loading completed from: `{st.session_state['zarr_path']}`")
        st.info(f"The storage contains {len(st.session_state['zarr_reader'])} trajectories")

    if st.session_state["zarr_meta"] is not None:
        st.markdown("---")
        show_static_metadata(st.session_state["zarr_meta"])


# -----------------------------
# Main
# -----------------------------
st.title("Coupled Bubble Position Control")

reader = st.session_state.get("zarr_reader", None)
if reader is None:
    st.info("Select a folder containing a Zarr trajectory storage, then click on the Load button.")
    st.stop()

# Plot style
st.subheader("Plot style")
style_cols = st.columns(3)
with style_cols[0]:
    palette_name = st.selectbox("Color palette", ["Plotly", "D3", "Dark24", "Set1"], index=0)
with style_cols[1]:
    axis_title_size = st.slider("Axis title font size", 10, 24, 14)
with style_cols[2]:
    tick_size = st.slider("Tick label font size", 8, 20, 12)

# Episode selection
st.subheader("Trajectory selection")
meta_cols = st.columns(3, vertical_alignment="bottom")
with meta_cols[0]:
    episode_id = st.number_input(
        "Trajectory_id",
        min_value=0,
        max_value=len(reader) - 1 if len(reader) > 0 else 0,
        step=1,
        help="Select the trajectory id",
    )

# Load episode only when ID changes
episode_id = int(episode_id)
if st.session_state["episode_id_loaded"] != episode_id:
    st.session_state["episode_data"] = reader.load_episode(episode_id)
    st.session_state["episode_id_loaded"] = episode_id

data = st.session_state["episode_data"] or {}
attrs = data.get("attrs", {})

episode_length = attrs.get("episode_length", None)
episode_reward = attrs.get("episode_reward", None)

with meta_cols[1]:
    st.metric("Episode length", episode_length if episode_length is not None else "N/A")
with meta_cols[2]:
    st.metric("Episode reward", f"{episode_reward:.2f}" if isinstance(episode_reward, (int, float)) else "N/A")

st.markdown("---")

# Pull arrays safely
dense_time = np.asanyarray(data.get("dense_time", []))
dense_state = data.get("dense_state", None)
observations = np.asanyarray(data.get("observations", []))
actions = np.asanyarray(data.get("actions", []))
rewards = np.asanyarray(data.get("rewards", []))

if dense_state is None or len(dense_time) == 0:
    st.error("Episode data is missing required fields (dense_time / dense_state).")
    st.stop()

dense_state_0 = np.asanyarray(dense_state[0])
dense_state_1 = np.asanyarray(dense_state[1])

# Determine num_units robustly
num_units = attrs.get("num_units", None)
if num_units is None:
    # fallback: second dim of state_1
    num_units = dense_state_1.shape[1] if dense_state_1.ndim == 2 else 1
num_units = int(num_units)

# Colors
palette = get_palette(palette_name)
colors = [palette[i % len(palette)] for i in range(num_units)]

global_meta = st.session_state.get("zarr_meta", None)
dt = float(meta_get(global_meta, "time_step_length", 5.0) or 5.0)
timesteps = int(attrs.get("episode_length", actions.shape[0] if actions.size else 0) or 0)

# Layout: two columns of plots
plot_cols = st.columns(2)

with plot_cols[0]:
    plot_position_time_curves(dense_time, dense_state_1, observations, num_units, colors, global_meta, axis_title_size, tick_size)

    # actions: amplitude and phase assumed packed as [PA(0..k-1), PS(0..k-1)]
    k_components = int(meta_get(global_meta, "k", 2) or 2)
    if actions.ndim == 2 and actions.shape[1] >= 2 * k_components:
        plot_step_action_curves(
            "Amplitude time curves", "Pₐ [bar]",
            actions[:, :k_components],
            dt, timesteps,
            colors[:k_components], axis_title_size, tick_size
        )
        plot_step_action_curves(
            "Phase-shift time curves", "θ [rad]",
            actions[:, k_components:2 * k_components],
            dt, timesteps,
            colors[:k_components], axis_title_size, tick_size
        )
    else:
        st.warning("Actions array shape does not match expected [T, 2*k]. Skipping action plots.")

with plot_cols[1]:
    plot_radius_time_curves(dense_time, dense_state_0, num_units, colors, axis_title_size, tick_size)
    if rewards.size:
        plot_reward_time_curves(rewards, dt, timesteps, axis_title_size, tick_size)
    else:
        st.info("No rewards found for this episode.")