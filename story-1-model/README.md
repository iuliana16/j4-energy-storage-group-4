# Story 1 model: Tonstad pumped storage

Python model behind the mechanical story (Iuliana Cibotar). Written with Claude (GenAI) under my direction and checked by the tests below; see the [GenAI declaration](../stories/1-mechanical.md).

| File | What it does |
|---|---|
| `psh_model.py` | Loads ENTSO-E prices, finds the best hour-by-hour pumping and generating (linear program), sizes flows, water volume and lake-level swing, and compares plant sizes (250–1,000 MW, 2–48 h of storage) |
| `econ_lca.py` | Cost per MWh, NPV and eight sensitivities (7%, 20 years); carbon footprint by life-cycle stage |
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

## Results with 2025 NO2 prices

| Result | Value |
|---|---|
| Energy delivered | 639 GWh/yr (≈160 full cycles) |
| Earnings from buying low and selling high | €33.0 M/yr |
| Build cost | €600 M (€1,200/kW, Limberg III benchmark) |
| Cost per MWh delivered (7%, 20 years) | €89 |
| NPV | −€253 M |
| Carbon footprint | 3.1 g CO₂e/kWh over 20 years; 0.93 g over 80 years |

**Limitations:** perfect foresight (an upper bound), the plant does not move prices, constant efficiency at part load, and Tonstad's own generation schedule is not modelled. These are covered by the sensitivities.
