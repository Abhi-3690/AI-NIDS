

## Streamlit SOC Dashboard

Install dependencies from `requirements.txt`, activate `.venv`, and run:

```powershell
$env:PYTHONPATH = "src"
python -m streamlit run src/nids/dashboard/app.py
```

Open `http://localhost:8501`. The dashboard uses a white claymorphism visual theme, responsive
layout, keyboard focus indicators, reduced-motion support, alert filters, prediction charts,
CSV export, and optional 10-second refresh. It reads the local `artifacts/alerts.jsonl` file; it
does not capture live traffic. Use the **Refresh now** control after generating new records.

The app title and SVG shield favicon are configured through Streamlit. Custom HTML meta descriptions,
custom 404 routes, and a multi-page mobile navigation menu are not native Streamlit website features;
those require a separate web frontend or hosting layer. The dashboard has no commercial checkout,
customer contact forms, marketing emails, reviews, or account system. See [`docs/PRIVACY_AND_USAGE.md`](docs/PRIVACY_AND_USAGE.md)
and [`SECURITY.md`](SECURITY.md) for the scope and limitations.
