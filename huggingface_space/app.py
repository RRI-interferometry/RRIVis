"""RRIVis Streamlit App for Hugging Face Spaces.

Interactive radio interferometer visibility simulator using the RIME
(Radio Interferometer Measurement Equation).
"""

import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="RRIVis - Visibility Simulator",
    page_icon="📡",
    layout="wide",
)

# --- Constants ---
CONFIG_PATH = Path(__file__).parent / "config.yaml"


# --- Plotting Functions ---
def plot_antenna_layout(antennas: dict) -> go.Figure:
    """Plot antenna positions using Plotly."""
    east = [ant["Position"][0] for ant in antennas.values()]
    north = [ant["Position"][1] for ant in antennas.values()]
    names = [ant["Name"] for ant in antennas.values()]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=east,
            y=north,
            mode="markers+text",
            text=names,
            textposition="top center",
            textfont={"size": 9},
            marker={"size": 10, "color": "#1f77b4", "symbol": "circle"},
            name="Antennas",
        )
    )
    fig.update_layout(
        title="Antenna Layout (ENU Coordinates)",
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        xaxis={"scaleanchor": "y", "scaleratio": 1},
        height=500,
        template="plotly_white",
    )
    return fig


def plot_uv_coverage(baselines: dict, wavelengths) -> go.Figure:
    """Plot UV coverage."""
    u_all, v_all = [], []
    for _bl_key, bl_data in baselines.items():
        bl_vec = np.asarray(bl_data["BaselineVector"])
        for wl in wavelengths:
            wl_val = wl.value if hasattr(wl, "value") else float(wl)
            u = bl_vec[0] / wl_val
            v = bl_vec[1] / wl_val
            u_all.extend([u, -u])
            v_all.extend([v, -v])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=u_all,
            y=v_all,
            mode="markers",
            marker={"size": 2, "color": "#1f77b4", "opacity": 0.5},
            name="UV points",
        )
    )
    fig.update_layout(
        title="UV Coverage",
        xaxis_title="u (wavelengths)",
        yaxis_title="v (wavelengths)",
        xaxis={"scaleanchor": "y", "scaleratio": 1},
        height=500,
        template="plotly_white",
    )
    return fig


def plot_visibility_spectrum(results: dict) -> go.Figure:
    """Plot visibility amplitude vs frequency for all baselines."""
    visibilities = results["visibilities"]
    freqs_mhz = results["frequencies"] / 1e6

    fig = go.Figure()
    for bl_key, bl_vis in visibilities.items():
        vis_I = bl_vis.get("I", bl_vis.get("XX", None))
        if vis_I is None:
            continue

        if vis_I.ndim == 2:
            amp = np.abs(vis_I[0, :])
        else:
            amp = np.abs(vis_I)

        fig.add_trace(
            go.Scatter(
                x=freqs_mhz,
                y=amp,
                mode="lines",
                name=f"Baseline {bl_key}",
                opacity=0.7,
                line={"width": 1.5},
            )
        )

    fig.update_layout(
        title="Visibility Amplitude vs Frequency (Stokes I, t=0)",
        xaxis_title="Frequency (MHz)",
        yaxis_title="Visibility Amplitude (Jy)",
        height=500,
        template="plotly_white",
        showlegend=True,
        legend={"font": {"size": 8}},
    )
    return fig


def plot_visibility_phase(results: dict) -> go.Figure:
    """Plot visibility phase vs frequency for all baselines."""
    visibilities = results["visibilities"]
    freqs_mhz = results["frequencies"] / 1e6

    fig = go.Figure()
    for bl_key, bl_vis in visibilities.items():
        vis_I = bl_vis.get("I", bl_vis.get("XX", None))
        if vis_I is None:
            continue

        if vis_I.ndim == 2:
            phase = np.angle(vis_I[0, :], deg=True)
        else:
            phase = np.angle(vis_I, deg=True)

        fig.add_trace(
            go.Scatter(
                x=freqs_mhz,
                y=phase,
                mode="lines",
                name=f"Baseline {bl_key}",
                opacity=0.7,
                line={"width": 1.5},
            )
        )

    fig.update_layout(
        title="Visibility Phase vs Frequency (Stokes I, t=0)",
        xaxis_title="Frequency (MHz)",
        yaxis_title="Phase (degrees)",
        height=500,
        template="plotly_white",
        showlegend=True,
        legend={"font": {"size": 8}},
    )
    return fig


def plot_visibility_matrix(results: dict) -> go.Figure:
    """Plot visibility amplitude as a baseline-frequency heatmap."""
    visibilities = results["visibilities"]
    freqs_mhz = results["frequencies"] / 1e6
    bl_keys = list(visibilities.keys())

    n_bl = len(bl_keys)
    n_freq = len(freqs_mhz)
    matrix = np.zeros((n_bl, n_freq))

    for i, bl_key in enumerate(bl_keys):
        vis_I = visibilities[bl_key].get("I", visibilities[bl_key].get("XX", None))
        if vis_I is None:
            continue
        if vis_I.ndim == 2:
            matrix[i, :] = np.abs(vis_I[0, :])
        else:
            matrix[i, :] = np.abs(vis_I)

    bl_labels = [str(k) for k in bl_keys]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=freqs_mhz,
            y=bl_labels,
            colorscale="Viridis",
            colorbar={"title": "Amplitude (Jy)"},
        )
    )
    fig.update_layout(
        title="Visibility Amplitude Heatmap (Stokes I, t=0)",
        xaxis_title="Frequency (MHz)",
        yaxis_title="Baseline",
        height=max(400, n_bl * 25),
        template="plotly_white",
    )
    return fig


# --- Main Content ---
st.title("RRIVis - Radio Interferometer Visibility Simulator")

# --- Display Config YAML ---
st.subheader("Simulation Configuration")
config_text = CONFIG_PATH.read_text()
st.code(config_text, language="yaml")

st.download_button(
    label="Download config.yaml",
    data=config_text,
    file_name="rrivis_config.yaml",
    mime="text/yaml",
)

# --- Run Simulation ---
run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Running visibility simulation..."):
        try:
            from rrivis import Simulator

            t0 = time.perf_counter()
            sim = Simulator.from_config(str(CONFIG_PATH))
            sim.setup()
            results = sim.run(progress=False)
            elapsed = time.perf_counter() - t0

            st.session_state["results"] = results
            st.session_state["sim"] = sim
            st.session_state["elapsed"] = elapsed

        except Exception as e:
            st.error(f"Simulation failed: {e}")
            import traceback

            st.code(traceback.format_exc())

# --- Display Results ---
if "results" in st.session_state:
    results = st.session_state["results"]
    sim = st.session_state["sim"]
    elapsed = st.session_state["elapsed"]

    st.success(f"Simulation completed in {elapsed:.2f}s")

    col1, col2, col3, col4 = st.columns(4)
    meta = results["metadata"]
    col1.metric("Antennas", meta["n_antennas"])
    col2.metric("Baselines", meta["n_baselines"])
    col3.metric("Sources", meta["n_sky_elements"])
    col4.metric("Freq Channels", meta["n_frequencies"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Antenna Layout",
            "UV Coverage",
            "Visibility Amplitude",
            "Visibility Phase",
            "Amplitude Heatmap",
        ]
    )

    with tab1:
        fig = plot_antenna_layout(results["antennas"])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = plot_uv_coverage(results["baselines"], results["wavelengths"])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = plot_visibility_spectrum(results)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = plot_visibility_phase(results)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        fig = plot_visibility_matrix(results)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw Visibility Data"):
        bl_keys = list(results["visibilities"].keys())
        selected_bl = st.selectbox("Select Baseline", bl_keys)
        if selected_bl:
            bl_vis = results["visibilities"][selected_bl]
            vis_I = bl_vis.get("I", bl_vis.get("XX"))
            if vis_I is not None:
                st.write(f"**Shape:** {vis_I.shape} (time steps x frequency channels)")
                st.write(
                    f"**Amplitude range:** {np.abs(vis_I).min():.4f} – "
                    f"{np.abs(vis_I).max():.4f} Jy"
                )
                st.write(
                    f"**Phase range:** {np.degrees(np.angle(vis_I)).min():.1f} – "
                    f"{np.degrees(np.angle(vis_I)).max():.1f} deg"
                )

    with st.expander("Timing Details"):
        timing = results["timing"]
        st.write(f"**Total time:** {timing['total']:.3f}s")
        st.write(f"**Setup time:** {timing['setup']:.3f}s")
        st.write(f"**Computation time:** {timing['total'] - timing['setup']:.3f}s")
        st.write(f"**Backend:** {meta['backend']}")
        st.write(f"**Simulator:** {meta['simulator']}")

else:
    st.info(
        "Click **Run Simulation** to simulate visibilities from the configuration above."
    )
