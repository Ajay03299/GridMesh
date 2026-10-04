# Editable GridMesh cost model

This standalone cost-model folder is not the full forecasting/dashboard project.
It needs Python and PyYAML (already included in GridMesh's dependencies).

From this folder:

```powershell
python build_full_service_costs.py
python -m unittest test_affordability
```

In the existing GridMesh checkout use `.venv\Scripts\python.exe` instead of
`python`. To use a different editable input or separate output folder:

```powershell
python build_full_service_costs.py --config configs/affordability_costs.yaml --output outputs/my_cost_case
```

Edit `configs/affordability_costs.yaml`: low/base/high paid hours and rates, tax
fractions, setup recovery months, participation and quoted missing items.
`null` is unknown, not zero. Every new quote needs source/date/location/units/
eligibility and payer. Confirm meter/edge capability before explicitly setting
retrofit to zero. USD conversion90/95/100 is a stress assumption, not market FX.

Outputs: itemised `COST_MODEL.md`,12 scenario rows as JSON/CSV, major-driver
sensitivity and separately labelled conditional-funding example. Unconfirmed
sponsor money never reduces ordinary required contributions. No selling-price
margin or actual bill savings established.

Read `docs/AFFORDABILITY.md` for the old4650 correction, pilot responsibilities,
scope, feasibility and limitations. Older cost configs are retained for backward
compatibility with the historical helper tests, but their unsupported pricing
inputs are null and do not drive the current calculation.

Only Slide11 changes in the v36 submission. Other slides, tables and original
experimental results remain unchanged. No commit, push, purchase or enquiry.
