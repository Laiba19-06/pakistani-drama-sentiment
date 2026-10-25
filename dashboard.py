"""
Pakistani Drama Sentiment Dashboard
Run from the drama_sentiment folder:
    streamlit run dashboard.py
Reads data/comments_sentiment.csv (or comments_sentiment.csv in the same folder).
"""
import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Drama Sentiment", layout="wide")

COLORS = {"positive": "#2e8b57", "neutral": "#b0b0b0", "negative": "#c0392b"}
ORDER = ["positive", "neutral", "negative"]
CHANNEL = {
    "Sinf-e-Aahan": "ARY", "Mere Humsafar": "ARY",
    "Ishq Murshid": "Hum TV", "Parizaad": "Hum TV", "Humsafar": "Hum TV",
    "Ehd-e-Wafa": "Hum TV", "Suno Chanda": "Hum TV",
    "Jaan Nisar": "Geo", "Tere Bin": "Geo",
}
STOP = set("""hai hain ho hoga hogi tha thi the ka ki ke ko se me mein pe par or aur ye yeh wo woh
 is us bhi to toh na nahi nhi kya kia ek hi hy ha hay bht bohat bahut jo ab ya kar kr kara karo
 rha rhi rahi raha rahe lag lagta lagti drama episode part yar yaar sab koi kuch sath tum main mai
 hum mera meri apna uss iss a an and are as at be but by for from had has have he her his i if in
 it its me my of on she so that their them they this was we what when who will with you your just
 very really one also like all can get would about there been were than then out not no do does did
 how why which more most some too been""".split())


@st.cache_data
def load():
    path = "data/comments_sentiment.csv" if os.path.exists("data/comments_sentiment.csv") else "comments_sentiment.csv"
    df = pd.read_csv(path)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df["channel"] = df["drama"].map(CHANNEL)
    df["words"] = df["text_clean"].astype(str).str.split().str.len()
    return df


def pct_table(d, by):
    t = (d.groupby(by)["sentiment"].value_counts(normalize=True)
           .unstack().reindex(columns=ORDER).fillna(0) * 100)
    return t


def top_words(texts, n=15):
    counts = {}
    for t in texts:
        for w in re.findall(r"[a-z]{3,}", str(t).lower()):
            if w not in STOP:
                counts[w] = counts.get(w, 0) + 1
    s = pd.Series(counts).sort_values(ascending=False).head(n)
    return s.rename_axis("word").reset_index(name="count")


try:
    df = load()
except FileNotFoundError:
    st.error("comments_sentiment.csv not found. Run this from the drama_sentiment folder.")
    st.stop()

# ---------------- sidebar ----------------
st.sidebar.header("Filters")
channels = st.sidebar.multiselect("Channel", sorted(df["channel"].unique()),
                                  default=sorted(df["channel"].unique()))
avail = sorted(df.loc[df["channel"].isin(channels), "drama"].unique())
dramas = st.sidebar.multiselect("Drama", avail, default=avail)
sents = st.sidebar.multiselect("Sentiment", ORDER, default=ORDER)
min_words = st.sidebar.slider("Minimum words per comment", 2, 30, 2)

f = df[df["drama"].isin(dramas) & df["sentiment"].isin(sents) & (df["words"] >= min_words)]
if f.empty:
    st.warning("No comments match these filters. Widen the selection in the sidebar.")
    st.stop()

# ---------------- header + numbers ----------------
st.title("How do viewers feel about Pakistani dramas?")
st.caption("YouTube comments on 9 dramas, labelled by a multilingual sentiment model "
           "(XLM-RoBERTa). Each drama has a 500-comment sample, so differences under "
           "about 4 points are within noise.")

share = f["sentiment"].value_counts(normalize=True) * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Comments shown", f"{len(f):,}")
c2.metric("Positive", f"{share.get('positive', 0):.1f}%")
c3.metric("Neutral", f"{share.get('neutral', 0):.1f}%")
c4.metric("Negative", f"{share.get('negative', 0):.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(["Dramas", "Episodes", "Words", "Comments"])

# ---------------- tab 1: dramas and channels ----------------
with tab1:
    left, right = st.columns([3, 2])
    with left:
        t = pct_table(f, "drama").sort_values("positive")
        fig = px.bar(t, x=ORDER, y=t.index, orientation="h", color_discrete_map=COLORS,
                     labels={"value": "% of comments", "y": "", "variable": "Sentiment"},
                     title="Sentiment by drama")
        fig.update_layout(barmode="stack", height=420, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        t = pct_table(f, "channel")
        fig = px.bar(t, x=t.index, y=ORDER, color_discrete_map=COLORS,
                     labels={"value": "% of comments", "x": "", "variable": "Sentiment"},
                     title="Sentiment by channel")
        fig.update_layout(barmode="stack", height=420, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ARY has 2 dramas here, Hum TV 5, Geo 2. Read the channel view with that in mind.")

    st.subheader("Drama ranking")
    rank = (f.groupby("drama")
              .agg(comments=("sentiment", "size"),
                   positive_pct=("sentiment", lambda s: (s == "positive").mean() * 100),
                   negative_pct=("sentiment", lambda s: (s == "negative").mean() * 100),
                   avg_score=("sent_score", "mean"))
              .sort_values("avg_score", ascending=False).round(2))
    rank.insert(0, "channel", [CHANNEL[d] for d in rank.index])
    st.dataframe(rank, use_container_width=True)
    st.caption("avg_score = positive minus negative probability, from -1 to +1. "
               "Angry comments about a villain count as negative even when they are praise for the drama.")

# ---------------- tab 2: episode trend ----------------
with tab2:
    mode = st.radio("Group episodes by", ["Five stages (start to end)", "Each episode"],
                    horizontal=True)
    d = f.copy()
    if mode.startswith("Five"):
        def stage(g):
            rel = (g["episode"] - g["episode"].min()) / max(1, g["episode"].max() - g["episode"].min())
            return np.minimum((rel * 5).astype(int), 4) + 1
        d["x"] = d.groupby("drama", group_keys=False).apply(stage)
        xlabel = "Stage of drama (1 = start, 5 = end)"
    else:
        d["x"] = d["episode"]
        xlabel = "Episode"
    trend = d.groupby(["drama", "x"])["sent_score"].mean().reset_index()
    fig = px.line(trend, x="x", y="sent_score", color="drama", markers=True,
                  labels={"x": xlabel, "sent_score": "Average sentiment (-1 to +1)", "drama": "Drama"},
                  title="How viewer mood changes through each drama")
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)
    if not mode.startswith("Five"):
        st.caption("With about 20 comments per episode, single episodes jump around. "
                   "The five-stage view is steadier.")

# ---------------- tab 3: words, length, likes ----------------
with tab3:
    a, b = st.columns(2)
    for col, s in zip([a, b], ["positive", "negative"]):
        with col:
            w = top_words(f.loc[f["sentiment"] == s, "text_clean"])
            if w.empty:
                st.info(f"No {s} comments in the current filter.")
                continue
            fig = px.bar(w.sort_values("count"), x="count", y="word", orientation="h",
                         title=f"Most common words in {s} comments",
                         color_discrete_sequence=[COLORS[s]])
            fig.update_layout(height=450, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Roman Urdu spelling varies a lot, so counts for words like 'acha' and 'accha' are split.")

    c, d_ = st.columns(2)
    with c:
        fig = px.box(f[f["words"] < 60], x="sentiment", y="words", color="sentiment",
                     category_orders={"sentiment": ORDER}, color_discrete_map=COLORS,
                     title="Comment length by sentiment", labels={"words": "Words"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with d_:
        lk = f.groupby("sentiment")["likes"].mean().reindex(ORDER).reset_index()
        fig = px.bar(lk, x="sentiment", y="likes", color="sentiment", color_discrete_map=COLORS,
                     title="Average likes per comment")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{(f['likes'] == 0).mean() * 100:.0f}% of these comments have 0 likes, "
                   "so this chart is weak evidence.")

# ---------------- tab 4: comments ----------------
with tab4:
    q = st.text_input("Search comments (for example a character name)")
    view = f if not q else f[f["text_clean"].str.contains(q, case=False, na=False, regex=False)]
    st.write(f"{len(view):,} comments")
    show = view[["drama", "episode", "sentiment", "sent_score", "likes", "text_clean"]].copy()
    show["sent_score"] = show["sent_score"].round(2)
    show = show.rename(columns={"text_clean": "comment"})

    def tint(v):
        return {"positive": "background-color:#dff0e5", "negative": "background-color:#f6dcd8",
                "neutral": "background-color:#eeeeee"}.get(v, "")

    st.dataframe(show.head(500).style.map(tint, subset=["sentiment"]),
                 use_container_width=True, height=520)
    st.caption("Showing up to 500 rows. The model can misread sarcasm, Bangla and heavily mixed Roman Urdu.")
