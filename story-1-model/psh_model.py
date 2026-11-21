"""
Tonstad pumped-storage operating and sizing model
J4 Energy Policy & Energy Storage - mechanical story (Norway)

What it does
------------
1. Loads hourly day-ahead prices (e.g. ENTSO-E Transparency export for bidding zone NO2).
2. Runs a hypothetical pumped-storage plant at Tonstad against those prices with a
   linear program (pump when cheap, generate when expensive, respect power and
   reservoir limits, end the period with the same water level it started with).
3. Reports the resulting charge/discharge profile, cycles, efficiency, flows,
   waterway (tunnel) diameter and, if the lake area is given, the water-level swing
   in the lower reservoir (Sirdalsvatnet).
4. Repeats the run for a grid of plant sizes (power x storage hours) to justify the
   chosen size.

Important limitations (state these in the report)
-------------------------------------------------
* Perfect foresight of prices -> results are an UPPER BOUND on what a real operator
  achieves. Use --mode weekly for a more conservative run (one week of foresight).
* Price-taker: the plant does not change market prices. A 500 MW plant would in
  reality narrow the spreads it trades on.
* Closed loop: natural inflow to the reservoirs and Tonstad's existing generation are
  not modelled; the pumped-storage plant is treated as an addition on top.
* Arbitrage revenue is an OPERATING indicator only. Per the brief, the cost of the
  energy supplied to storage is outside the economic scope, so the LCOS must be
  calculated separately from CAPEX/O&M/replacement/end-of-life costs.

All default parameter values are starting assumptions and must be checked against a
cited source before use (see README).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

RHO = 1000.0  # kg/m3, fresh water
G = 9.81      # m/s2


# --------------------------------------------------------------------------- plant
@dataclass
class Plant:
    head_m: float = 450.0            # gross head, Tonstad (SINTEF/CEDREN Tonstad slides) (assumption)
    turbine_mw: float = 500.0        # generating power (design choice, Step 1)
    pump_mw: float = 500.0           # pumping power (design choice, Step 1)
    storage_hours: float = 8.0       # hours at full turbine power (design choice, Step 1)
    eta_turbine: float = 0.92        # hydraulic turbine efficiency (assumption)
    eta_pump: float = 0.91           # hydraulic pump efficiency (assumption)
    eta_machine: float = 0.98        # motor-generator efficiency (assumption)
    eta_transformer: float = 0.995   # step-up transformer efficiency (assumption)
    head_loss_frac: float = 0.03     # waterway friction loss as fraction of gross head, each way (assumption)
    var_om_eur_mwh: float = 1.0      # variable O&M per MWh generated (also stops pointless cycling) (assumption)
    lake_area_km2: float | None = None  # Sirdalsvatnet surface area (NVE)
    tunnel_velocity_ms: float = 4.0  # design flow velocity in headrace tunnel (assumption)

    # ---- derived quantities
    @property
    def eta_gen_total(self) -> float:
        """Electricity delivered to grid per unit of potential energy at GROSS head."""
        return self.eta_turbine * self.eta_machine * self.eta_transformer * (1 - self.head_loss_frac)

    @property
    def eta_pump_total(self) -> float:
        """Potential energy stored at GROSS head per unit of electricity drawn from grid."""
        return self.eta_pump * self.eta_machine * self.eta_transformer / (1 + self.head_loss_frac)

    @property
    def round_trip(self) -> float:
        return self.eta_gen_total * self.eta_pump_total

    @property
    def kwh_per_m3_gross(self) -> float:
        return RHO * G * self.head_m / 3.6e6

    @property
    def mwh_out_per_m3(self) -> float:
        return self.kwh_per_m3_gross * self.eta_gen_total / 1000.0

    @property
    def storage_mwh(self) -> float:
        """Storage capacity expressed as electricity deliverable to the grid."""
        return self.turbine_mw * self.storage_hours

    @property
    def storage_volume_m3(self) -> float:
        return self.storage_mwh / self.mwh_out_per_m3

    @property
    def q_turbine_max(self) -> float:
        return self.turbine_mw * 1e6 / (RHO * G * self.head_m * self.eta_gen_total)

    @property
    def q_pump_max(self) -> float:
        return self.pump_mw * 1e6 * self.eta_pump_total / (RHO * G * self.head_m)

    @property
    def tunnel_diameter_m(self) -> float:
        q = max(self.q_turbine_max, self.q_pump_max)
        return math.sqrt(4 * q / (math.pi * self.tunnel_velocity_ms))

    def lake_level_swing_m(self, volume_m3: float) -> float | None:
        if not self.lake_area_km2:
            return None
        return volume_m3 / (self.lake_area_km2 * 1e6)


# --------------------------------------------------------------------------- prices
def load_prices(path: str) -> pd.Series:
    """Load an hourly (or 15-min) price file. Accepts the ENTSO-E Transparency CSV export
    ('MTU (CET/CEST)', 'Day-ahead Price [EUR/MWh]'), or any two-column time,price file.
    Sub-hourly data are averaged to hourly."""
    df = pd.read_csv(path, sep=None, engine="python")
    cols = list(df.columns)
    pcols = [c for c in cols if "price" in c.lower() or "eur" in c.lower()]
    pc = pcols[0] if pcols else cols[1]
    tcols = [c for c in cols if c != pc and any(k in c.lower() for k in ("mtu", "time", "date", "hour", "start"))]
    tc = tcols[0] if tcols else cols[0]

    t = df[tc].astype(str).str.split(" - ").str[0].str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    iso = t.str.match(r"^\d{4}-\d{2}-\d{2}")
    ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if iso.any():
        ts[iso] = pd.to_datetime(t[iso], errors="coerce", utc=True).dt.tz_localize(None)
    if (~iso).any():
        ts[~iso] = pd.to_datetime(t[~iso], dayfirst=True, errors="coerce")
    p = pd.to_numeric(df[pc].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    s = pd.Series(p.values, index=pd.DatetimeIndex(ts.values)).dropna()
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="first")]  # DST repeated hour
    if len(s) < 48:
        raise ValueError(f"Only {len(s)} usable price rows found in {path}; check the file format.")
    hourly = s.resample("h").mean().interpolate(limit=3).dropna()
    hourly.name = "price_eur_mwh"
    return hourly


# --------------------------------------------------------------------------- dispatch LP
def _dispatch_window(prices: np.ndarray, plant: Plant, s0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal dispatch for one window, cyclic (ends at s0). Energy state in MWh-out."""
    T = len(prices)
    rt = plant.round_trip
    smax = plant.storage_mwh
    # variables: [Pp_0..Pp_T-1, Pg_0..Pg_T-1, S_1..S_T]
    c = np.concatenate([prices, -prices + plant.var_om_eur_mwh, np.zeros(T)])
    rows = np.arange(T)
    A = sparse.lil_matrix((T, 3 * T))
    A[rows, rows] = -rt              # pumping adds rt * Pp
    A[rows, T + rows] = 1.0          # generating removes Pg
    A[rows, 2 * T + rows] = 1.0      # S_t
    A[rows[1:], 2 * T + rows[:-1]] = -1.0  # - S_{t-1}
    b = np.zeros(T)
    b[0] = s0
    bounds = [(0, plant.pump_mw)] * T + [(0, plant.turbine_mw)] * T + [(0, smax)] * (T - 1) + [(s0, s0)]
    res = linprog(c, A_eq=A.tocsr(), b_eq=b, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x
    return x[:T], x[T:2 * T], x[2 * T:]


def dispatch(prices: pd.Series, plant: Plant, mode: str = "perfect", soc_start: float = 0.5) -> pd.DataFrame:
    p = prices.values.astype(float)
    s0 = soc_start * plant.storage_mwh
    if mode == "perfect":
        windows = [(0, len(p))]
    elif mode == "weekly":
        windows = [(i, min(i + 168, len(p))) for i in range(0, len(p), 168)]
    else:
        raise ValueError("mode must be 'perfect' or 'weekly'")
    pp, pg, s = [], [], []
    for a, b in windows:
        x1, x2, x3 = _dispatch_window(p[a:b], plant, s0)
        pp.append(x1); pg.append(x2); s.append(x3)
    out = pd.DataFrame({
        "price_eur_mwh": p,
        "pump_mw": np.concatenate(pp),
        "gen_mw": np.concatenate(pg),
        "stored_mwh_end": np.concatenate(s),
    }, index=prices.index)
    out.loc[out["pump_mw"].abs() < 1e-6, "pump_mw"] = 0.0
    out.loc[out["gen_mw"].abs() < 1e-6, "gen_mw"] = 0.0
    out["net_mw_to_grid"] = out["gen_mw"] - out["pump_mw"]
    out["storage_level_pct"] = 100 * out["stored_mwh_end"] / plant.storage_mwh
    out["upper_volume_m3"] = out["stored_mwh_end"] / plant.mwh_out_per_m3
    out["flow_m3s"] = (out["gen_mw"] / plant.turbine_mw) * plant.q_turbine_max - \
                      (out["pump_mw"] / max(plant.pump_mw, 1e-9)) * plant.q_pump_max
    return out


# --------------------------------------------------------------------------- summary
def summarise(res: pd.DataFrame, plant: Plant) -> dict:
    pumped = res["pump_mw"].sum()
    gen = res["gen_mw"].sum()
    rev = float((res["price_eur_mwh"] * res["net_mw_to_grid"]).sum())
    buy = float((res["price_eur_mwh"] * res["pump_mw"]).sum() / pumped) if pumped > 0 else float("nan")
    sell = float((res["price_eur_mwh"] * res["gen_mw"]).sum() / gen) if gen > 0 else float("nan")
    vol_swing = float(res["upper_volume_m3"].max() - res["upper_volume_m3"].min())
    hours = len(res)
    out = {
        "hours_modelled": hours,
        "energy_pumped_GWh": pumped / 1e3,
        "energy_generated_GWh": gen / 1e3,
        "realised_round_trip": gen / pumped if pumped > 0 else float("nan"),
        "design_round_trip": plant.round_trip,
        "equivalent_full_cycles": gen / plant.storage_mwh,
        "equivalent_full_cycles_per_year": gen / plant.storage_mwh * 8760 / hours,
        "hours_pumping": int((res["pump_mw"] > 1).sum()),
        "hours_generating": int((res["gen_mw"] > 1).sum()),
        "turbine_capacity_factor": gen / (plant.turbine_mw * hours),
        "avg_buy_price_eur_mwh": buy,
        "avg_sell_price_eur_mwh": sell,
        "arbitrage_margin_eur_per_annum": rev * 8760 / hours,
        "storage_capacity_MWh": plant.storage_mwh,
        "storage_volume_Mm3": plant.storage_volume_m3 / 1e6,
        "energy_per_m3_gross_kWh": plant.kwh_per_m3_gross,
        "max_turbine_flow_m3s": plant.q_turbine_max,
        "max_pump_flow_m3s": plant.q_pump_max,
        "tunnel_diameter_m": plant.tunnel_diameter_m,
        "max_volume_swing_Mm3": vol_swing / 1e6,
        "lower_lake_level_swing_m": plant.lake_level_swing_m(vol_swing),
    }
    return out


def size_sweep(prices: pd.Series, base: Plant, powers, hours, mode: str) -> pd.DataFrame:
    rows = []
    for pw in powers:
        for h in hours:
            pl = replace(base, turbine_mw=pw, pump_mw=pw, storage_hours=h)
            sm = summarise(dispatch(prices, pl, mode), pl)
            rows.append({"power_MW": pw, "storage_hours": h, "storage_MWh": pl.storage_mwh,
                         "margin_MEUR_per_year": sm["arbitrage_margin_eur_per_annum"] / 1e6,
                         "margin_EUR_per_kW_year": sm["arbitrage_margin_eur_per_annum"] / (pw * 1e3),
                         "cycles_per_year": sm["equivalent_full_cycles_per_year"],
                         "capacity_factor": sm["turbine_capacity_factor"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- charts
def make_charts(res: pd.DataFrame, sweep: pd.DataFrame, plant: Plant, outdir: str, label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})

    # 1. sample week with the largest price swing
    wk = res["price_eur_mwh"].rolling(168).apply(lambda x: x.max() - x.min(), raw=True)
    end = wk.idxmax() if wk.notna().any() else res.index[min(167, len(res) - 1)]
    w = res.loc[:end].tail(168)
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.2, 1]})
    ax[0].plot(w.index, w["price_eur_mwh"], color=INK, lw=1.6)
    ax[0].set_ylabel("Price (€/MWh)")
    ax[1].bar(w.index, w["gen_mw"], width=1 / 24 * 0.9, color=BLUE, label="Generating")
    ax[1].bar(w.index, -w["pump_mw"], width=1 / 24 * 0.9, color=ORANGE, label="Pumping")
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].set_ylabel("Power (MW)\n+ to grid / − from grid")
    ax[1].legend(loc="upper right", frameon=False, ncol=2)
    ax[2].plot(w.index, w["storage_level_pct"], color=BLUE, lw=2)
    ax[2].set_ylabel("Upper reservoir\nstorage (%)")
    ax[2].set_ylim(0, 105)
    for a in ax:
        a.grid(axis="y", color=GRID, lw=0.8)
    fig.suptitle(f"Tonstad pumped storage: one week of operation ({plant.turbine_mw:.0f} MW, "
                 f"{plant.storage_hours:g} h){label}", x=0.01, ha="left", fontsize=12, color=INK)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "week_operation.png"), dpi=200)
    plt.close(fig)

    # 2. storage level over the whole period (daily mean)
    d = res["storage_level_pct"].resample("D").mean()
    fig, a = plt.subplots(figsize=(10, 3.2))
    a.plot(d.index, d.values, color=BLUE, lw=1.6)
    a.set_ylabel("Daily mean storage (%)")
    a.set_ylim(0, 105)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_title(f"Upper reservoir storage level across the period{label}", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "storage_level_year.png"), dpi=200)
    plt.close(fig)

    # 3. size sweep heatmap (sequential single hue)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", ["#eaf2fc", "#1d5aa6"])
    piv = sweep.pivot(index="power_MW", columns="storage_hours", values="margin_EUR_per_kW_year")
    fig, a = plt.subplots(figsize=(7, 4))
    im = a.imshow(piv.values, cmap=cmap, aspect="auto", origin="lower")
    a.set_xticks(range(len(piv.columns)), [f"{c:g} h" for c in piv.columns])
    a.set_yticks(range(len(piv.index)), [f"{r:.0f} MW" for r in piv.index])
    vmax = np.nanmax(piv.values)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                   color="white" if v > 0.6 * vmax else INK)
    a.set_xlabel("Storage duration at full power")
    a.set_ylabel("Pump/turbine power")
    cb = fig.colorbar(im, ax=a)
    cb.set_label("Arbitrage margin (€/kW/yr)")
    a.set_title(f"Plant size comparison{label}", loc="left", fontsize=12, color=INK)
    for s in a.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "size_sweep.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Tonstad pumped-storage operating & sizing model")
    ap.add_argument("prices", help="CSV of hourly day-ahead prices (ENTSO-E export or time,price)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--mode", choices=["perfect", "weekly"], default="perfect")
    ap.add_argument("--power", type=float, default=500.0, help="pump and turbine power, MW")
    ap.add_argument("--hours", type=float, default=8.0, help="storage hours at full power")
    ap.add_argument("--head", type=float, default=450.0, help="gross head, m")
    ap.add_argument("--lake-area-km2", type=float, default=None, help="Sirdalsvatnet area from NVE")
    ap.add_argument("--no-sweep", action="store_true")
    a = ap.parse_args(argv)

    os.makedirs(a.outdir, exist_ok=True)
    synthetic = "synthetic" in os.path.basename(a.prices).lower()
    label = "  [SYNTHETIC TEST DATA - not real prices]" if synthetic else ""

    prices = load_prices(a.prices)
    plant = Plant(head_m=a.head, turbine_mw=a.power, pump_mw=a.power, storage_hours=a.hours,
                  lake_area_km2=a.lake_area_km2)
    res = dispatch(prices, plant, a.mode)
    summ = summarise(res, plant)
    summ.update({"price_file": os.path.basename(a.prices), "synthetic_data": synthetic,
                 "mode": a.mode, "period_start": str(prices.index[0]), "period_end": str(prices.index[-1])})

    res.to_csv(os.path.join(a.outdir, "hourly_operation.csv"), float_format="%.3f")
    with open(os.path.join(a.outdir, "summary.json"), "w") as f:
        json.dump({"plant": asdict(plant), "results": summ}, f, indent=2, default=float)

    if a.no_sweep:
        sweep = pd.DataFrame()
    else:
        sweep = size_sweep(prices, plant, [250, 500, 750, 1000], [2, 4, 6, 8, 12, 24, 48], a.mode)
        sweep.to_csv(os.path.join(a.outdir, "size_sweep.csv"), index=False, float_format="%.3f")
        make_charts(res, sweep, plant, a.outdir, label)

    print(f"\nTonstad PSH - {a.power:.0f} MW / {a.hours:g} h - mode={a.mode}{label}")
    for k, v in summ.items():
        if isinstance(v, float):
            print(f"  {k:34s} {v:,.3f}")
        else:
            print(f"  {k:34s} {v}")
    print(f"\nOutputs written to {os.path.abspath(a.outdir)}")


if __name__ == "__main__":
    main()
