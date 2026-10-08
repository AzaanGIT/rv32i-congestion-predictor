# RV32I Congestion Predictor

A website that predicts routing congestion for your RV32I core from
5 floorplan knobs (`utilization`, `aspect_ratio`, `core_margin`, `density`,
`layer_adj`), shows it per metal layer (top view + side/stack view), and — if
it's congested — tells you the exact new values to set to fix it, either
overall or for one specific layer.

Trained on `data/congestion_dataset_full.csv` (30,500 runs).

## What it does

**Page 1 — enter your design.** Set the 5 parameters, click "Predict congestion".

**Page 2 — results.**
- Routing success probability
- Total congestion % (`Total_usage_pct`)
- Per-layer congestion % for M1–M10, as a heatmap grid (top view) and a
  stacked bar view (side/elevation view)
- A plain verdict: low / moderate / high / critical
- **Suggest a fix** — pick a target congestion %, and it searches nearby
  parameter combinations for the **smallest exact change** that gets you
  under that target while keeping the design routable. Real numbers
  ("utilization: 92.0 → 86.0"), never vague directions.
- **Fix this layer** — click the button under any single layer (M1–M10) to
  get a fix targeted at just that layer instead of the total.
- **Search the full parameter space** — if no nearby fix is found, widen
  the search before giving up.

## How the models work

- `train_model.py` trains:
  - a `RandomForestClassifier` for routing success/fail (99.2% test accuracy)
  - one `RandomForestRegressor` per metal layer + total congestion (R² 0.99+
    on layers 2–10; layer 1 is always 0 in this dataset so it's trivial)
- `recommend.py` does a bounded coordinate search over the 5 knobs (starting
  from the most influential ones for success, per the classifier's feature
  importances), scores candidates by how little they change from your
  current input, and returns the closest one that clears your target — for
  the total, or for one layer if you passed `layer`.
- Model artifacts are pre-trained and included in `model/` (joblib-compressed
  to stay under 100MB so they fit in a normal GitHub repo). Re-run
  `python train_model.py` any time you replace the dataset.

## Run it locally

```bash
cd congestion_app
pip3 install -r requirements.txt
python3 app.py
```

Then open **http://localhost:5000** (or whatever port is printed — if 5000
is taken, e.g. by macOS AirPlay Receiver, edit the port number on the last
line of `app.py`).

A trained model is already included — you don't need to run
`train_model.py` first.

## Making it public (Render, free)

This deploys the site to a permanent public URL anyone can open in Chrome —
no need to keep your laptop on.

1. **Put the code on GitHub.**
   - Create a free account at github.com if you don't have one.
   - Create a new repository (e.g. `rv32i-congestion-predictor`).
   - Upload the entire `congestion_app` folder's contents to it (GitHub's
     web UI has an "Add file → Upload files" button — drag the folder's
     contents in, or use `git push` if you're comfortable with git).
2. **Create the Render service.**
   - Go to render.com, sign up free, and connect your GitHub account.
   - Click "New +" → "Web Service" → pick your repository.
   - Name it `rv32i-congestion-predictor` (this becomes part of your public
     URL: `rv32i-congestion-predictor.onrender.com`).
   - Render auto-detects Python and the `Procfile` (already included, it
     says `web: gunicorn app:app`) — leave build/start commands as
     suggested.
   - Choose the **Free** instance type.
   - Click "Create Web Service".
3. **Wait for the first deploy** (a few minutes — it's installing
   dependencies and loading the ~55MB model file). Render shows live logs;
   when it says "Your service is live", your link is ready.
4. Share the `.onrender.com` link — anyone can open it in Chrome from
   anywhere. Free-tier services sleep after 15 minutes idle and take ~30
   seconds to wake back up on the next visit; that's normal for the free
   plan.

If you'd rather not touch git/GitHub, Railway.app has a similar "New
Project → Deploy from GitHub" flow, or ask me and I'll adjust these steps
for whichever host you pick.

## About the Colab link

I can't open Google Colab notebooks (they're behind your Google login, so
I have no way to fetch that URL). If there's specific code or logic in that
notebook you want carried over here, paste the relevant cells and I'll fold
it in.

## Project structure

```
congestion_app/
├── app.py                 # Flask backend (routes: /, /api/meta, /api/predict, /api/recommend)
├── recommend.py           # the "what to change" search logic (overall or per-layer)
├── train_model.py         # trains + saves the classifier/regressors
├── requirements.txt
├── Procfile                # tells Render/Railway how to start the app
├── runtime.txt              # pins a Python version for deployment
├── data/
│   └── congestion_dataset_full.csv
├── model/                 # pre-trained artifacts (classifier, regressors, meta.json)
├── templates/
│   └── index.html          # both screens (input + results)
└── static/
    ├── css/style.css
    └── js/script.js
```
