import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from preprocessor import parse_whatsapp_chat
from helper import (
    fetch_stats, monthly_timeline, daily_timeline,
    week_activity_map, month_activity_map, activity_heatmap,
    most_busy_Speakers, create_wordcloud,
    most_common_words, emoji_helper
)

# Inject custom CSS
with open("static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Register custom font
FONT_PATH = "NotoSans-Regular.ttf"
fm.fontManager.addfont(FONT_PATH)
fp = fm.FontProperties(fname=FONT_PATH)
plt.rcParams['font.family'] = fp.get_name()
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Social Chat Analyzer", layout="wide")
st.sidebar.title("SOCIAL CHAT ANALYZER")

uploaded = st.sidebar.file_uploader("Upload chat (.txt)", type=["txt"])
if uploaded:
    content = uploaded.getvalue().decode("utf-8")
    df = parse_whatsapp_chat(content)

    speakers = sorted(df.Speaker.unique())
    speakers.insert(0, "Overall")
    choice = st.sidebar.selectbox("Show analysis for", speakers)

    if st.sidebar.button("Show Analysis"):
        # --- Top metrics ---
        msgs, words, media, links = fetch_stats(choice, df)
        st.header("📊 Top Statistics")
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in zip(
            (c1, c2, c3, c4),
            ("Messages", "Words", "Media Shared", "Links Shared"),
            (msgs, words, media, links)
        ):
            col.metric(label, val)

        # --- Monthly timeline ---
        st.header("📈 Monthly Timeline")
        mt = monthly_timeline(choice, df)
        fig, ax = plt.subplots()
        ax.plot(mt['time'], mt['Message'], marker='o', label='Messages per Month')
        ax.set_xlabel("Month-Year")
        ax.set_ylabel("Number of Messages")
        ax.set_xticks(range(len(mt['time'])))
        ax.set_xticklabels(mt['time'], rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)

        # --- Daily timeline ---
        st.header("📅 Daily Timeline")
        dt = daily_timeline(choice, df)
        fig, ax = plt.subplots()
        ax.plot(dt['Date'], dt['Message'], marker='o', label='Messages per Day')
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Messages")
        ax.set_xticks(dt['Date'])
        ax.set_xticklabels(dt['Date'].dt.strftime('%d-%b-%Y'), rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)

        # --- Activity maps ---
        st.header("🗓️ Activity Map")
        d1, d2 = st.columns(2)

        with d1:
            st.subheader("Most Busy Day")
            bd = week_activity_map(choice, df)
            fig, ax = plt.subplots()
            ax.bar(bd.index, bd.values, label='Message Count')
            for p in ax.patches:
                ax.annotate(int(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height()),
                            ha='center', va='bottom')
            ax.set_xlabel("Day of Week")
            ax.set_ylabel("Messages")
            ax.set_xticks(range(len(bd.index)))
            ax.set_xticklabels(bd.index, rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)

        with d2:
            st.subheader("Most Busy Month")
            bm = month_activity_map(choice, df)
            fig, ax = plt.subplots()
            ax.bar(bm.index, bm.values, label='Message Count')
            for p in ax.patches:
                ax.annotate(int(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height()),
                            ha='center', va='bottom')
            ax.set_xlabel("Month")
            ax.set_ylabel("Messages")
            ax.set_xticks(range(len(bm.index)))
            ax.set_xticklabels(bm.index, rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)

        # --- Weekly heatmap ---
        st.header("🕗 Weekly Heatmap")
        hm = activity_heatmap(choice, df)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(hm, annot=True, fmt="d", ax=ax, cbar=True, cbar_kws={'label': 'Messages'})
        ax.set_xlabel("Hour Period")
        ax.set_ylabel("Day of Week")
        st.pyplot(fig)

        # --- Overall busiest speakers ---
        if choice == "Overall":
            st.header("🏆 Top Speakers")
            counts, pct = most_busy_Speakers(df)
            fig, ax = plt.subplots()
            ax.bar(counts.index, counts.values, label='Message Count')
            for p in ax.patches:
                ax.annotate(int(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height()),
                            ha='center', va='bottom')
            ax.set_xlabel("Speaker")
            ax.set_ylabel("Messages Sent")
            ax.set_xticks(range(len(counts.index)))
            ax.set_xticklabels(counts.index, rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)
            st.dataframe(pct)

        # --- Wordcloud ---
        st.header("☁️ Wordcloud")
        wc = create_wordcloud(choice, df)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # --- Most common words ---
        st.header("🔤 Most Common Words")
        common = most_common_words(choice, df)
        fig, ax = plt.subplots()
        ax.barh(common[0], common[1], label='Frequency')
        for p in ax.patches:
            ax.annotate(int(p.get_width()), (p.get_width(), p.get_y() + p.get_height()/2),
                        ha='left', va='center')
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Words")
        ax.set_yticks(range(len(common[0])))
        ax.set_yticklabels(common[0])
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)

        # --- Emoji analysis ---
        st.header("😀 Emoji Analysis")
        emojis = emoji_helper(choice, df)
        st.dataframe(emojis, hide_index=True)
