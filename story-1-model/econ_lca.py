"""
Economic (LCOS / NPV) and life-cycle GWP assessment for the Tonstad pumped-storage design.
Reads the operating results of psh_model.py (summary.json) so the assessment uses the
same throughput as the technical analysis. All inputs are listed in INPUTS with a source
or an explicit 'ASSUMPTION' flag.

Run:  python econ_lca.py results/summary.json [results_weekly/summary.json]
"""
import json
import sys

# ----------------------------------------------------------------------------- inputs
P_MW = 500
INPUTS = {
    # --- economics (2025 EUR, real)
    "capex_eur_per_kw": 1200,   # Limberg III analogue: EUR 572M / 480 MW = EUR 1,192/kW (existing reservoirs, underground). Power Technology 2025
    "capex_low": 900, "capex_high": 1800,  # range
    "fixed_om_eur_per_kw_yr": 17.0,  # PNNL 2022: PSH fixed O&M incl. refurbishment $15.6 (1000 MW) - $28.2 (100 MW)/kW-yr
    "var_om_eur_per_mwh": 1.0,       # ASSUMPTION (same as dispatch model)
    "discount_rate": 0.07,           # group basis, matching the electrochemical story (7 %, 20 y)
    "study_years": 20,               # group basis, matching the electrochemical story
    "usd_to_eur": 0.885,             # ECB 2025 annual average 1.130 USD/EUR (Bank of Finland table)
    # CAPEX split (ASSUMPTION - typical PSH structure) and service lives
    "split": {
        "Civil & underground works (tunnels, shafts, caverns, intakes)": (0.45, 80),
        "Electromechanical (pump-turbines, motor-generators, valves)":   (0.27, 40),
        "Frequency converters (variable speed)":                         (0.03, 20),
        "Electrical & grid (transformers, GIS, 420 kV cables)":          (0.10, 40),
        "Control, protection & SCADA":                                   (0.02, 15),
        "Engineering, owner's costs & contingency":                      (0.13, 80),
    },
    "decom_frac_of_capex": 0.02,     # ASSUMPTION - end-of-life cost (E&M removal, plugging), net of scrap value
}


def pv(cash, year, r):
    return cash / (1 + r) ** year


def economics(gen_mwh_yr, margin_eur_yr, capex_kw, r, years, inp=INPUTS):
    capex = capex_kw * P_MW * 1e3
    om = inp["fixed_om_eur_per_kw_yr"] * P_MW * 1e3 + inp["var_om_eur_per_mwh"] * gen_mwh_yr
    pv_om = sum(pv(om, t, r) for t in range(1, years + 1))
    pv_repl, pv_resid = 0.0, 0.0
    for name, (share, life) in inp["split"].items():
        c = share * capex
        k = 1
        while k * life < years:            # like-for-like replacement at end of life
            pv_repl += pv(c, k * life, r)
            k += 1
        last = (k - 1) * life               # year the current item was installed
        remaining = last + life - years
        pv_resid += pv(c * remaining / life, years, r)  # straight-line residual value
    # end-of-life cost: only falls inside the study period if the study period reaches plant life (80 y)
    pv_decom = pv(inp["decom_frac_of_capex"] * capex, years, r) if years >= 80 else 0.0
    pv_energy = sum(pv(gen_mwh_yr, t, r) for t in range(1, years + 1))
    pv_cost = capex + pv_om + pv_repl - pv_resid + pv_decom
    pv_margin = sum(pv(margin_eur_yr, t, r) for t in range(1, years + 1))
    return {
        "capex_MEUR": capex / 1e6, "annual_OM_MEUR": om / 1e6,
        "PV_OM_MEUR": pv_om / 1e6, "PV_replacements_MEUR": pv_repl / 1e6,
        "PV_residual_value_MEUR": pv_resid / 1e6, "PV_decommissioning_MEUR": pv_decom / 1e6,
        "PV_total_cost_MEUR": pv_cost / 1e6,
        "LCOS_EUR_per_MWh": pv_cost / pv_energy,
        "arbitrage_margin_per_MWh_discharged": margin_eur_yr / gen_mwh_yr,
        "NPV_with_arbitrage_MEUR": (pv_margin - pv_cost) / 1e6,
    }


# ----------------------------------------------------------------------------- LCA
# Material/activity inventory for the design (quantities = ASSUMPTIONS from the design
# dimensions). Emission factors: ICE database v3.0 (Circular Ecology) cradle-to-gate unless noted.
LCA_INV = [
    # name, quantity, unit, kgCO2e per unit, stage, note
    ("Concrete (caverns, intakes, plugs, shotcrete)", 115_000, "t", 130.0, "A1-A3", "~48,000 m3 x 2.4 t/m3 (no new headrace or reservoir intake); ICE v3 general concrete ~0.13 kgCO2e/kg"),
    ("Reinforcing steel & rock bolts", 3_900, "t", 1990.0, "A1-A3", "ICE v3 rebar, world avg ~1.99 kgCO2e/kg"),
    ("Steel lining, pressure shaft & manifold", 2_000, "t", 2460.0, "A1-A3", "300 m x D5.5 m x 40 mm + manifold; ICE v3 steel plate ~2.46"),
    ("Electromechanical equipment (steel)", 3_000, "t", 2460.0, "A1-A3", "2 units x ~1,500 t; fabricated steel, plate factor used"),
    ("Copper (generator windings, transformers, cables)", 300, "t", 2710.0, "A1-A3", "ICE v3 copper ~2.71 kgCO2e/kg"),
    ("Transformer oil", 140, "t", 1000.0, "A1-A3", "ASSUMPTION ~1.0 kgCO2e/kg"),
    ("Explosives for rock excavation", 460, "t", 2000.0, "A5", "230,000 m3 x 2 kg/m3; ASSUMPTION ~2 kgCO2e/kg (ammonium-nitrate based)"),
    ("Diesel for excavation, mucking, haulage", 690_000, "L", 3.25, "A5", "3 L per m3 rock; 2.68 kg combustion + ~0.57 upstream"),
    ("Transport of steel & E&M (sea 1,500 km + road 300 km)", 9_200 * 1_800, "t.km", 0.03, "A4", "ASSUMPTION blended ~0.03 kgCO2e/t.km"),
    ("Transport of concrete & aggregates (road 30 km)", 115_000 * 30, "t.km", 0.1065, "A4", "HGV average laden 0.1065 kgCO2e/t.km (BEIS 2020 via IStructE 2020)"),
]
REPL = {  # B-stage replacements within the 80-year life
    # PNNL PSH methodology: every ~20 years the rotor is replaced and the stator rewired -> ~30 % of E&M mass at 20, 40, 60 y
    "Electromechanical refurbishment every 20 y (rotors, windings, ~30 % of E&M mass)": (0.30, 20),
    # Transformers and frequency converters: full replacement at 40 y (ASSUMPTION)
    "Transformers & converters at 40 y (100 %)": (1.0, 40),
}


def lca(gen_mwh_yr, life_years):
    rows, tot = [], 0.0
    for name, q, u, ef, stage, note in LCA_INV:
        t = q * ef / 1000  # tCO2e
        rows.append((name, stage, t))
        tot += t
    em_steel = 3000 * 2.46
    copper_oil = 300 * 2.71 + 140 * 1.0

    def n_events(interval):  # replacements strictly inside the assessment period
        return len([k for k in range(interval, 200, interval) if k < life_years])
    (f_em, i_em), (f_tr, i_tr) = REPL.values()
    b = n_events(i_em) * f_em * em_steel + n_events(i_tr) * f_tr * copper_oil
    c = 0.005 * tot  # C-stage: removal/dismantling of E&M at end of life (ASSUMPTION 0.5 % of A-stage)
    # Module D (reported separately, not netted): 90 % of E&M steel and copper recovered and recycled;
    # avoided primary production ASSUMED 1.5 kgCO2e/kg steel and 2.0 kgCO2e/kg copper. Steel lining,
    # rebar and concrete stay sealed in the rock and are not credited.
    d_credit = -(0.9 * 3000 * 1.5 + 0.9 * 300 * 2.0)
    total = tot + b + c
    return {"A_tCO2e": tot, "B_tCO2e": b, "C_tCO2e": c, "D_credit_tCO2e": d_credit, "total_ABC_tCO2e": total,
            "gCO2e_per_kWh_delivered": total * 1e6 / (gen_mwh_yr * life_years * 1e3), "rows": rows}


def main():
    s_perf = json.load(open(sys.argv[1]))["results"]
    s_week = json.load(open(sys.argv[2]))["results"] if len(sys.argv) > 2 else s_perf
    gen = s_perf["energy_generated_GWh"] * 1e3
    margin = s_perf["arbitrage_margin_eur_per_annum"]
    r, n = INPUTS["discount_rate"], INPUTS["study_years"]
    out = {"base": economics(gen, margin, INPUTS["capex_eur_per_kw"], r, n)}
    sens = {
        "CAPEX low (900 EUR/kW)": economics(gen, margin, INPUTS["capex_low"], r, n),
        "CAPEX high (1,800 EUR/kW)": economics(gen, margin, INPUTS["capex_high"], r, n),
        "Discount rate 4 %": economics(gen, margin, INPUTS["capex_eur_per_kw"], 0.04, n),
        "Discount rate 10 %": economics(gen, margin, INPUTS["capex_eur_per_kw"], 0.10, n),
        "Study period 60 y": economics(gen, margin, INPUTS["capex_eur_per_kw"], r, 60),
        "Study period 80 y (incl. end-of-life)": economics(gen, margin, INPUTS["capex_eur_per_kw"], r, 80),
        "Weekly foresight (realistic dispatch)": economics(s_week["energy_generated_GWh"] * 1e3,
                                                          s_week["arbitrage_margin_eur_per_annum"],
                                                          INPUTS["capex_eur_per_kw"], r, n),
        "Throughput -25 % (price-taker correction)": economics(0.75 * gen, 0.6 * margin, INPUTS["capex_eur_per_kw"], r, n),
    }
    out["sensitivity"] = sens
    out["lca_study_period"] = lca(gen, n)
    out["lca_80y"] = lca(gen, 80)
    json.dump(out, open("econ_lca_results.json", "w"), indent=2, default=float)

    b = out["base"]
    print(f"BASE CASE ({n} y, {r:.0%}, EUR 1,200/kW)")
    for k, v in b.items():
        print(f"  {k:40s} {v:10.2f}")
    print("\nSENSITIVITY            LCOS EUR/MWh   NPV w/ arbitrage MEUR")
    for k, v in sens.items():
        print(f"  {k:44s} {v['LCOS_EUR_per_MWh']:8.1f}   {v['NPV_with_arbitrage_MEUR']:9.1f}")
    for key in ("lca_study_period", "lca_80y"):
        l = out[key]
        print(f"\nLCA {key}: A {l['A_tCO2e']:,.0f} t, B {l['B_tCO2e']:,.0f} t, C {l['C_tCO2e']:,.0f} t, "
              f"total {l['total_ABC_tCO2e']:,.0f} tCO2e -> {l['gCO2e_per_kWh_delivered']:.2f} gCO2e/kWh delivered; "
              f"D credit {l['D_credit_tCO2e']:,.0f} t")
    print("\nA-stage breakdown:")
    for name, stage, t in out["lca_study_period"]["rows"]:
        print(f"  {stage:5s} {name:55s} {t:9,.0f} tCO2e  {100*t/out['lca_study_period']['A_tCO2e']:5.1f} %")


if __name__ == "__main__":
    main()
