```markdown
# Marketing Intelligence Dashboard

A comprehensive **Streamlit dashboard** that connects marketing campaign performance with business outcomes.

## Features
- Executive KPIs and metrics
- Channel performance analysis
- Business correlation insights
- Actionable recommendations
- Campaign drilldown capabilities

## Setup (Local)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your CSV files in the `data/` folder
3. Run the dashboard:
```bash
streamlit run app.py
```

## Data Requirements

* `Facebook.csv`, `Google.csv`, `TikTok.csv` — marketing campaign data
* `Business.csv` — business performance data

## Deployed App

 [**Marketing Intelligence Dashboard**](https://marketing-dashboard-znw5mmaxd8edl4addxtfu3.streamlit.app/)

## Deployment on Streamlit Cloud

To deploy this dashboard on **Streamlit Cloud**:

1. Push this repository to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click **New app** and connect your GitHub repo
4. Select:
   - **Repository:** `your-username/your-repo-name`
   - **Branch:** `main`
   - **File:** `app.py`
5. Click **Deploy**
6. Upload your data files (`Facebook.csv`, `Google.csv`, `TikTok.csv`, `Business.csv`) into the `/data` folder in the repo, or configure external data sources

Your app will be live at:
```
https://share.streamlit.io/your-username/your-repo-name/main/app.py
```
```
