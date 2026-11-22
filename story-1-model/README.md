# Story 1 model: Tonstad pumped storage

Python model behind the mechanical story (Iuliana Cibotar). Written with Claude (GenAI) under my direction and checked by the tests below; see the [GenAI declaration](../stories/1-mechanical.md).

| File | What it does |
|---|---|
| `psh_model.py` | Loads ENTSO-E prices, finds the best hour-by-hour pumping and generating (linear program), sizes flows, water volume and lake-level swing, and compares plant sizes (250–1,000 MW, 2–48 h of storage) |
| `econ_lca.py` | Cost per MWh, NPV and eight sensitivities (7%, 20 years, 2026 EUR); carbon footprint by life-cycle stage |
| `test_psh_model.py` | 9 tests against hand calculations |
| `make_synthetic_prices.py` | Makes a fake price file, for testing only |

## Run it

```bash
pip install numpy pandas scipy matplotlib
python test_psh_model.py                     # All 9 tests passed
python psh_model.py NO2_prices_2025.csv --outdir results
python econ_lca.py results/summary.json
```

**Price data (not included):** ENTSO-E Transparency Platform, Market → Energy Prices (day-ahead), bidding zone NO2, year 2025, export as CSV.

## Results with 2025 NO2 prices (costs in 2026 EUR, group basis)

| Result | Value |
|---|---|
| Energy delivered | 639 GWh/yr (≈160 full cycles) |
| Earnings from buying low and selling high | €33.8 M/yr |
| Build cost | €615 M (€1,200/kW in 2025 EUR from the Limberg III benchmark, +2.5% to 2026 EUR) |
| Cost per MWh delivered (7%, 20 years) | €91 |
| NPV | −€260 M |
| Carbon footprint | 2.9 g CO₂e/kWh over 20 years; 0.85 g over 80 years (group steel factor 1.92 kg CO₂e/kg) |

**Limitations:** perfect foresight (an upper bound), the plant does not move prices, constant efficiency at part load, and Tonstad's own generation schedule is not modelled. These are covered by the sensitivities.
