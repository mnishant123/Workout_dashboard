# Workout Dashboard (Streamlit + Google Sheets)

A Streamlit app for daily workout logging with Google Sheets as the database.

## Features

- Data entry form with default date = today
- Upsert behavior on submit:
  - If the date already exists, the row is updated
  - If the date does not exist, a new row is appended
- Editable scoring weights in the sidebar
- KPI cards:
  - Today
  - Last 7 days
  - Last 30 days
  - Current streak
  - Best streak
- Charts:
  - Score trend
  - Running trend
  - Strength trend
  - Weekday heatmap (average score)
  - Monthly totals
- Date range filter for chart/table views

## Google Sheet Schema

Your worksheet must have this exact header row:

`Date, Plank_min, Squats, Crunches, Pushups, Skips_min, Stairs, Running_km, Cult_sessions, Total_score`

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a Google Cloud service account and enable Google Sheets API.
3. Share your target Google Sheet with the service account email.
4. Add Streamlit secrets in `.streamlit/secrets.toml`:

```toml
google_sheet_name = "YOUR_SPREADSHEET_NAME"
google_worksheet_name = "Sheet1"  # optional, defaults to Sheet1

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
universe_domain = "googleapis.com"
```

## Run

```bash
streamlit run app.py
```

## Notes

- `Total_score` is auto-calculated from your sidebar weights and entered metrics.
- KPIs are computed from the full dataset; date filter affects charts and data table.
- Streaks are based on days with `Total_score > 0`.
