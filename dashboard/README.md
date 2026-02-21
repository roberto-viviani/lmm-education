# LLM Education Platform Dashboard

A comprehensive Streamlit-based analytics dashboard for monitoring and analyzing LLM (Large Language Model) educational platform usage, interactions, and performance metrics.

## Overview

This dashboard provides real-time insights into:
- Platform usage patterns and user engagement
- Question-answer interactions and response quality
- Session analytics and user behavior

## Quick Start

### Prerequisites
- Python 3.8+
- Streamlit
- Pandas
- Plotly

### Installation

1. Navigate to the dashboard directory:
```bash
cd statistics/dashboard
```

2. Install dependencies:
```bash
pip install streamlit pandas plotly
```

3. Run the dashboard:
```bash
streamlit run Start.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 🌐 Language Support

The dashboard supports **German and English**. Switch languages using the buttons (DE / EN) in the top-right corner of any page. Language preference is maintained across navigation.

## 📋 Pages

### 1. **Welcome Page** (`Start.py`)
Entry point to the dashboard

update needed
---

### 2. **Overview Page** (`pages/02_Overview.py`)
Main analytics dashboard with comprehensive KPI metrics and visualizations.

#### Key Performance Indicators (KPIs)
Located at the top of the page:

- **Total Questions**: Total number of questions asked across all sessions
  - *Metric*: Count of all user queries
  
- **Total Sessions**: Number of unique user sessions
  - *Metric*: Count of unique session hashes
  
- **Avg Response**: Average length of model responses
  - *Metric*: Mean character count of responses
  
- **Avg Query**: Average length of user questions
  - *Metric*: Mean character count of queries

#### Time Period Filters
Select your analysis timeframe:
- Last Day (24 hours)
- Last Week (7 days)
- Last Month (30 days)
- Last 3 Months (90 days)
- Last 6 Months (180 days)

All charts and metrics update dynamically based on selected period.

#### Visualizations

1. **Questions Over Time** (Line Chart)
   - Daily question volume trend
   - Shows engagement patterns and peaks

2. **Hourly Activity** (Bar Chart)
   - Distribution of questions by hour (UTC)
   - Identifies peak usage times

3. **Model Analysis**
   - Top models by usage frequency
   - Response length distribution across models

4. **Content Analysis**
   - Query length distribution (histogram)
   - Response length distribution (histogram)
   - Shows content complexity patterns

---

### 3. **Session Explorer** (`pages/02_session_explorer.py`)
Detailed view of individual user sessions with question-answer pairs.

#### Features

**Sorting Options:**
Access the sort menu next to "Sessions" count:
- **Newest first** (default) - Most recent sessions first
- **Oldest first** - Chronological order from beginning
- **Most messages** - Sessions with most Q&A pairs
- **Least messages** - Sessions with fewer interactions

**Session Display:**
Each session shows:
- Session number and timestamp
- Message count
- Model used
- Session duration

**Expandable Session Details:**
Click any session to expand and view:
- Session metadata (messages, model, duration)
- Complete question-answer pairs
- Timestamps for each interaction
- Q/A in side-by-side format for easy reading

---

## Understanding the Metrics

### Questions Metric
**Definition**: Total number of questions asked by users in the selected time period.
- **Significance**: Key indicator of platform engagement
- **Use Case**: Monitor user activity levels and trends

### Sessions Metric
**Definition**: Number of unique user sessions (based on session hash).
- **Significance**: Shows distinct user engagement instances
- **Use Case**: Track number of separate user interactions with the platform

### Response Length Metric
**Definition**: Average character count of AI-generated responses.
- **Significance**: Indicates response detail and complexity
- **Use Case**: Analyze if responses are getting more/less detailed over time

### Query Length Metric
**Definition**: Average character count of user questions.
- **Significance**: Shows question complexity and detail level
- **Use Case**: Understand user query sophistication

---

## Data Flow

```
Data Source (messages.csv)
         ↓
    Load & Parse
         ↓
  Time Period Filter
         ↓
  Group & Aggregate
         ↓
  Visualize & Display
```

### Data Requirements
The dashboard expects a `messages.csv` file with columns:
- `record_id` - Unique message identifier
- `client_host` - User IP address
- `session_hash` - Unique session identifier
- `timestamp` - Message timestamp
- `history_length` - Conversation history length
- `model_name` - LLM model used
- `interaction_type` - Type of interaction
- `query` - User question
- `response` - AI response

---

## UI/UX Features

### Language Toggle
- **DE** button for German interface
- **EN** button for English interface
- Buttons located in top-right corner
- Language preference persists across pages

### Responsive Design
- Wide layout optimized for desktop viewing
- Responsive columns and charts
- Mobile-friendly navigation

### Interactive Charts
- Hover over charts to see exact values
- Zoom and pan capabilities on line charts
- Click legend items to toggle data series

---

## 🛠️ Technical Architecture

### Files Structure
```
dashboard/
├── Start.py                    # Welcome page
├── data.py                     # Data loading utilities
├── translations.py             # Internationalization
├── pages/
│   ├── 02_Overview.py         # Main analytics dashboard
│   └── 02_session_explorer.py # Session details viewer
└── README.md                   # This file
```

## Troubleshooting

### Dashboard Won't Load
- Verify `messages.csv` exists in parent directory
- Check Python dependencies: `pip install streamlit pandas plotly`
- Ensure Streamlit is up to date: `pip install --upgrade streamlit`

### Data Not Updating
- Page cache may need refresh: Press `R` in Streamlit UI
- Check `messages.csv` modification time
- Verify data file path is correct

### Charts Not Displaying
- Ensure data exists for selected time period
- Try shorter time range if data is sparse
- Check browser console for JavaScript errors

