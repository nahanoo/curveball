"""
Consumer–resource model with Baranyi lag and Monod uptake.

Workflow:
1) Fit growth curve with curveball (logistic-lag) to get r and q0 (and Kcarry).
2) Convert Kcarry + resource concentration to a yield Y.
3) Fit ONLY Km (Monod half-saturation constant) by matching the ODE model to OD(t).

Units:
- t in hours
- resource R in mM
- Km in mM (bounds are set in µM->mM)
- N is OD (or biomass proxy)
- Y is OD per mM resource
"""

import numpy as np
from scipy.integrate import solve_ivp
import lmfit
import plotly.graph_objects as go


def sort_unique_xy(t, y, how="mean"):
    """Sort (t,y) by time and deduplicate t (needed for solve_ivp t_eval)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    order = np.argsort(t)
    t, y = t[order], y[order]

    if how == "first":
        t_u, idx = np.unique(t, return_index=True)
        return t_u, y[idx]

    if how == "mean":
        t_u = np.unique(t)
        y_u = np.array([y[t == tu].mean() for tu in t_u], dtype=float)
        return t_u, y_u

    raise ValueError("how must be 'first' or 'mean'")


def alpha_baranyi(t, q0, v):
    """Baranyi adjustment function alpha(t) in [0,1]."""
    return q0 / (q0 + np.exp(-v * t))


def cr_lag_rhs(t, y, r, Km, q0, Y, v):
    """ODE RHS for consumer (N) and resource (R)."""
    N, R = y
    R = max(R, 0.0)
    N = max(N, 0.0)

    a = alpha_baranyi(t, q0=q0, v=v)
    mu = r * R / (R + Km)

    dNdt = a * mu * N
    dRdt = -(1.0 / Y) * a * mu * N
    return [dNdt, dRdt]


def simulate_cr_lag(t, N0, R0, r, Km, q0, Y, v=None, method="LSODA"):
    """
    Simulate model at times t. If v is not provided, we set v=r (common Baty/Baranyi constraint).
    """
    if v is None:
        v = r

    t = np.asarray(t, dtype=float)
    if t.size < 2:
        raise ValueError("Need at least 2 timepoints.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing (use sort_unique_xy).")

    sol = solve_ivp(
        fun=lambda tt, yy: cr_lag_rhs(tt, yy, r=r, Km=Km, q0=q0, Y=Y, v=v),
        t_span=(float(t[0]), float(t[-1])),
        y0=[float(N0), float(R0)],
        t_eval=t,
        method=method,
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    N_hat, R_hat = sol.y
    return N_hat, R_hat


def fit_Km_lmfit(
    t,
    N_data,
    *,
    N0,
    R0,
    r,
    q0,
    Y,
    v=None,
    Km_guess=0.1,  # mM
    Km_min_uM=0.001,  # µM
    Km_max_mM=20.0,  # mM
    dedup="mean",
    method="least_squares",
):
    """
    Fit ONLY Km (Monod half-saturation constant).
    Bounds:
      Km ∈ [Km_min_uM, Km_max_mM] converted to mM for the model.
    """
    t, N_data = sort_unique_xy(t, N_data, how=dedup)
    Km_min_mM = Km_min_uM * 1e-3  # µM -> mM

    params = lmfit.Parameters()
    params.add("Km", value=Km_guess, min=Km_min_mM, max=Km_max_mM)

    def residual(p):
        Km = p["Km"].value
        N_hat, _ = simulate_cr_lag(t, N0=N0, R0=R0, r=r, Km=Km, q0=q0, Y=Y, v=v)
        return N_hat - N_data

    return lmfit.minimize(residual, params, method=method)


def plot_fit(t_raw, y_raw, t_fit, y_fit, label, r, Km, out="tmp.svg"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_raw, y=y_raw, mode="markers", name="Data"))
    fig.add_trace(go.Scatter(x=t_fit, y=y_fit, mode="lines", name="Model"))
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.01,
        y=0.99,
        text=f"{label}<br>r: {r:.3f} h⁻¹<br>Km: {Km:.3f} mM",
        showarrow=False,
    )
    fig.write_image(out)


# -------------------------
# How to use (minimal)
# -------------------------
# Inputs expected from a curveball logistic-lag fit:
#   fit.userkws["t"]  -> timepoints (may be unsorted/duplicated)
#   fit.data          -> OD data
#   fit.params        -> includes y0, K (carrying capacity), r, q0
#
# You must provide:
#   conc = initial limiting resource concentration (mM)
#   label = string for plot annotation

# Example:
# models = fit_curveball(df)           # df: curveball-format tidy dataframe
# fit = models[2]                      # logistic-lag model
# p = fit.params
#
# conc = 15.0                          # mM, arbitrary scaling (must match your Y definition)
# q0 = p["q0"].value
# r  = p["r"].value
# N0 = p["y0"].value
# Y  = (p["K"].value - p["y0"].value) / conc   # OD per mM (assumption)
#
# t_raw = fit.userkws["t"]
# y_raw = fit.data
#
# res = fit_Km_lmfit(t_raw, y_raw, N0=N0, R0=conc, r=r, q0=q0, Y=Y, Km_guess=0.05)
# lmfit.report_fit(res)
#
# t_fit, y_fit_data = sort_unique_xy(t_raw, y_raw, how="mean")
# Km_hat = res.params["Km"].value
# y_fit = simulate_cr_lag(t_fit, N0=N0, R0=conc, r=r, Km=Km_hat, q0=q0, Y=Y)[0]
#
# plot_fit(t_raw, y_raw, t_fit, y_fit, label="At on glucose", r=r, Km=Km_hat, out="tmp.svg")
