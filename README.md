# Sentiment analysis of Pakistani dramas

Analyses YouTube comments on 9 Pakistani dramas from ARY Digital, Hum TV and Geo to compare audience sentiment between dramas and channels, and to see how mood changes through a drama.

## Folder layout

```
drama_sentiment/
  01_collect_data.ipynb          YouTube API: episode lists and comments
  02_clean_and_sentiment.ipynb   cleaning and sentiment model
  03_eda_charts.ipynb            charts and findings
  dashboard.py                   interactive dashboard (Streamlit)
  data/
    dramas_selected.csv          the 9 dramas
    episodes_clean.csv           clean episode list
    youtube_comments.csv         all downloaded comments
    comments_clean.csv           cleaned comments
    comments_sample.csv          500 comments per drama
    comments_sentiment.csv       sample with sentiment labels
    raw/                         one comments CSV per drama
  charts/                        PNG charts
```

## How to run

1. `pip install google-api-python-client pandas requests emoji transformers torch sentencepiece matplotlib seaborn wordcloud streamlit plotly`
2. Paste your own YouTube Data API key into notebook 01 (it is not included).
3. Run the notebooks in order. Each step skips work already saved, so re-running is safe.
4. Dashboard: `streamlit run dashboard.py`

## Method

- **Data:** newest top-level comments, up to 500 per episode, from each drama's official full-episode playlist. About 159,000 comments, 130,000 after cleaning.
- **Cleaning:** links, timestamps and mentions removed, emoji converted to words, duplicates and one-word comments dropped.
- **Sentiment:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`, run on 500 comments per drama (4,500 total) spread evenly across episodes.

## Limitations

- The model is not trained specifically on Roman Urdu, and its accuracy was not checked by hand-labelling.
- 500 comments per drama gives roughly a plus or minus 4 point margin.
- "Negative" often means anger at a character, not dislike of the drama.
- Comments are the newest per episode, not spread over each video's life.
- ARY has only 2 dramas; *Kabhi Main Kabhi Tum* was dropped because only clips are uploaded.
- IMDb ratings were not collected.

Author names were removed from the data files.
