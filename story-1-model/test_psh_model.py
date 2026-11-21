"""Verification tests: each model result is checked against an independent hand calculation.
Run:  python test_psh_model.py"""
import math

import numpy as np
import pandas as pd

from psh_model import Plant, dispatch, summarise, load_prices

TOL = 1e-6


def test_energy_per_m3():
    # Hand calc: rho*g*h = 1000*9.81*450 = 4,414,500 J/m3 = 1.22625 kWh/m3
    p = Plant(head_m=450)
    assert abs(p.kwh_per_m3_gross - 1.22625) < 1e-9, p.kwh_per_m3_gross


def test_round_trip_in_published_range():
    # Hand calc with default efficiencies:
    # gen  = 0.92*0.98*0.995*0.97 ; pump = 0.91*0.98*0.995/1.03
    gen = 0.92 * 0.98 * 0.995 * 0.97
    pump = 0.91 * 0.98 * 0.995 / 1.03
    p = Plant()
    assert abs(p.round_trip - gen * pump) < 1e-12
    # Pitorac et al. (2020): existing Norwegian PSPs achieve 65-80 % round trip
    assert 0.65 <= p.round_trip <= 0.80, p.round_trip


def test_flow_rate():
    # Q = P / (rho g H eta) ; independent calc
    p = Plant(turbine_mw=500, head_m=450)
    q = 500e6 / (1000 * 9.81 * 450 * p.eta_gen_total)
    assert abs(p.q_turbine_max - q) < 1e-9
    assert 120 < q < 145  # sanity: ~126 m3/s at eta 0.9 (Step 1 hand estimate)


def test_volume_matches_energy():
    p = Plant(turbine_mw=500, storage_hours=8)
    # 4000 MWh / (1.22625 kWh/m3 * eta_gen) ; hand estimate ~3.6 Mm3 at eta 0.9
    v = 4000e3 / (1.22625 * p.eta_gen_total)
    assert abs(p.storage_volume_m3 - v) < 1e-3
    assert 3.3e6 < v < 4.0e6


def test_toy_two_hour_arbitrage():
    # Prices 10 then 100 EUR/MWh, 100 MW, 1 h storage, start and end empty.
    # Analytic optimum: pump 100 MWh at 10, store rt*100, generate rt*100 at 100.
    p = Plant(turbine_mw=100, pump_mw=100, storage_hours=1)
    prices = pd.Series([10.0, 100.0], index=pd.date_range("2025-01-01", periods=2, freq="h"))
    r = dispatch(prices, p, soc_start=0.0)
    rt = p.round_trip
    assert abs(r["pump_mw"].iloc[0] - 100) < 1e-4
    assert abs(r["gen_mw"].iloc[1] - rt * 100) < 1e-4
    margin = float((r["price_eur_mwh"] * r["net_mw_to_grid"]).sum())
    assert abs(margin - (100 * rt * 100 - 10 * 100)) < 1e-3


def test_flat_prices_no_operation():
    # With flat prices, losses make every cycle loss-making -> plant should stay idle.
    p = Plant()
    prices = pd.Series(np.full(168, 80.0), index=pd.date_range("2025-01-01", periods=168, freq="h"))
    r = dispatch(prices, p)
    assert r["pump_mw"].sum() < 1e-3 and r["gen_mw"].sum() < 1e-3


def test_energy_balance_and_limits():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01", periods=24 * 28, freq="h")
    prices = pd.Series(60 + 30 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 10, len(idx)),
                       index=idx)
    p = Plant()
    for mode in ("perfect", "weekly"):
        r = dispatch(prices, p, mode=mode)
        s = summarise(r, p)
        # cyclic: everything pumped (times round trip) comes back out
        assert abs(r["gen_mw"].sum() - p.round_trip * r["pump_mw"].sum()) < 1e-3
        assert abs(s["realised_round_trip"] - p.round_trip) < 1e-6
        assert r["pump_mw"].max() <= p.pump_mw + TOL
        assert r["gen_mw"].max() <= p.turbine_mw + TOL
        assert r["stored_mwh_end"].min() >= -TOL
        assert r["stored_mwh_end"].max() <= p.storage_mwh + TOL
        # perfect foresight is never worse than weekly foresight
    perf = summarise(dispatch(prices, p, "perfect"), p)["arbitrage_margin_eur_per_annum"]
    week = summarise(dispatch(prices, p, "weekly"), p)["arbitrage_margin_eur_per_annum"]
    assert perf >= week - 1e-3


def test_tunnel_diameter():
    p = Plant()
    q = max(p.q_turbine_max, p.q_pump_max)
    assert abs(p.tunnel_diameter_m - math.sqrt(4 * q / (math.pi * 4.0))) < 1e-9


def test_entsoe_format_loader():
    """ENTSO-E export: 15-minute rows, quoted fields, an empty intraday column and a missing row.
    Each hour must be the mean of its four quarter-hours; the gap must not break the hourly series."""
    import os, tempfile
    start = pd.Timestamp("2025-01-01 00:00")
    lines = ['"MTU (CET/CEST)","Area","Sequence","Day-ahead Price (EUR/MWh)","Intraday Period (CET/CEST)","Intraday Price (EUR/MWh)"']
    for q in range(3 * 24 * 4):                      # three days of quarter-hours
        if q == 37:                                  # one missing quarter-hour (hour 9)
            continue
        t0 = start + pd.Timedelta(minutes=15 * q)
        t1 = t0 + pd.Timedelta(minutes=15)
        price = (q // 4) + (q % 4) * 0.5             # hour h -> h, h+0.5, h+1, h+1.5 (mean h+0.75)
        lines.append(f'"{t0:%d/%m/%Y %H:%M:%S} - {t1:%d/%m/%Y %H:%M:%S}","BZN|NO2","Without Sequence","{price}","",""')
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("\n".join(lines))
        path = f.name
    try:
        hourly = load_prices(path)
    finally:
        os.remove(path)
    assert len(hourly) == 72
    assert abs(hourly.iloc[0] - 0.75) < TOL
    assert abs(hourly.iloc[30] - 30.75) < TOL
    assert abs(hourly.iloc[9] - (9 + 10 + 10.5) / 3) < TOL   # hour 9 averaged over the 3 rows present


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
