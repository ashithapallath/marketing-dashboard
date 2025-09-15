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

 <img width="761" height="263" alt="mar1" src="https://github.com/user-attachments/assets/0b3b080f-acb3-4006-8bd7-effaf4ed14ec" />
 <img width="827" height="323" alt="mar2" src="https://github.com/user-attachments/assets/f051bf74-f9b5-46f0-bb15-e54def9d021e" />
 <img width="846" height="352" alt="mar3" src="https://github.com/user-attachments/assets/aa268b4f-0d91-46f0-b45e-a4eed3023af9" />
 <img width="654" height="280" alt="mar4" src="https://github.com/user-attachments/assets/2123627c-a27a-44e5-a92c-42b17c100fad" />


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
