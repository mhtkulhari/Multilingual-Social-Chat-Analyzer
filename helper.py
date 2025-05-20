#helper.py

import pandas as pd
import emoji
import re
import seaborn as sns
import matplotlib.pyplot as plt
import json
import math
import google.generativeai as genai
import streamlit as st

from app.ml_models.summary_model.model import SummaryModel
from app.ml_models.emotion_model.model import EmotionModel
from app.ml_models.encoder_model.model import TextEncoderModel
from app.ml_models.clustering_model.model import TextClusteringModel
from app.ml_models.agreement_model.model import TextAgreementModel
from app.config import DECAY_FACTOR
from urlextract import URLExtract
from collections import Counter
from wordcloud import WordCloud
from types import SimpleNamespace
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations
from app.ml_models.summary_model.config import (DEFAULT_MODEL_NAME,DEFAULT_MODEL_CHAT_HISTORY,GEMINI_API_KEY)
from google.api_core.exceptions import NotFound

#1
def fetch_stats(selected_Speaker,df):

    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]

    # fetch the number of Messages
    num_Messages = df.shape[0]

    # fetch number of media Messages
    num_media_Messages = df['Message'].apply(lambda msg: len(re.findall(r'<Media omitted>', msg))).sum()
    # fetch number of links shared
    extract = URLExtract()
    links = []
    for Message in df['Message']:
        links.extend(extract.find_urls(Message))

    # Longest message by word count
    words_per_msg = df['Message'].apply(lambda m: len(m.split()))
    longest_msg = words_per_msg.max() if not words_per_msg.empty else 0

    # Average words per message
    avg_words = round(words_per_msg.mean()) if not words_per_msg.empty else 0

    # Active days
    active_days = df['Datetime'].dt.date.nunique() if not df.empty and 'Datetime' in df.columns else 0

    # Most used emojis (top 3)
    all_emojis = []
    for msg in df['Message']:
        all_emojis.extend([c for c in msg if c in emoji.EMOJI_DATA])
    emoji_counter = Counter(all_emojis)
    most_common_emoji = emoji_counter.most_common(1)

    return num_Messages,num_media_Messages,len(links),longest_msg, avg_words, active_days, most_common_emoji

#2
def monthly_timeline(selected_Speaker,df):

    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['Message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

#3
def daily_timeline(selected_Speaker, df):
    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Group by the 'Date' and count the messages
    daily_timeline = df.groupby('Date').count()['Message'].reset_index()

    # Sort by Date
    daily_timeline = daily_timeline.sort_values(by='Date')

    return daily_timeline

#4
def week_activity_map(selected_Speaker,df):

    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]

    return df['day_name'].value_counts()

#5
def month_activity_map(selected_Speaker,df):

    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]

    return df['month'].value_counts()


#6
def activity_heatmap(selected_Speaker, df):
    d = df if selected_Speaker == 'Everyone' else df[df['Speaker'] == selected_Speaker]
    pt = d.pivot_table(
        index='day_name',
        columns='period',
        values='Message',
        aggfunc='count'
    ).fillna(0)
    order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    return pt.reindex(order)

def period_24_to_12(period):
    # Handles strings like "13-14" and returns "1 PM – 2 PM"
    start, end = period.split('-')
    start = int(start)
    end = int(end)
 
    # Convert to 12-hour format with AM/PM
    def fmt(hr):
        if hr == 0 or hr == 24:
            return "12 AM"
        elif hr == 12:
            return "12 PM"
        elif hr > 12:
            return f"{hr-12} PM"
        else:
            return f"{hr} AM" if hr < 12 else f"{hr} PM"

    return f"{fmt(start)}-{fmt(end)}"

def plot_heatmap(pt, ax):
    # Convert counts to integers
    pt_int = pt.fillna(0).astype(int)
    max_count = pt_int.values.max()

    # Build annotation matrix: only the max count per row
    annot = pd.DataFrame("", index=pt_int.index, columns=pt_int.columns)
    for day in pt_int.index:
        row = pt_int.loc[day]
        m = row.max()
        if m > 0:
            annot.loc[day, row == m] = str(m)

    # White-to-dark-blue colormap
    cmap = LinearSegmentedColormap.from_list(
        "white_to_dark",
        [(1, 1, 1),    # white
         (0.0, 0.0, 0.5)]  # dark blue
    )

    sns.heatmap(
        pt_int,
        annot=annot,
        fmt="",               # use raw strings from annot
        cmap=cmap,
        vmin=0,
        vmax=max_count,
        cbar_kws={
            'label': 'Messages',
            'ticks': list(range(0, max_count + 1, max(1, max_count // 5)))
        },
        ax=ax,
        linewidths=0,
        linecolor='none',
        xticklabels=True,
        yticklabels=True
    )

    # Draw dotted grid lines at every cell boundary
    n_rows, n_cols = pt_int.shape
    for i in range(n_rows + 1):
        ax.hlines(i,0 , n_cols ,
                  colors='gray', linestyles=':', linewidth=1)
    for j in range(n_cols+1):
        ax.vlines(j, 0, n_rows,
                  colors='gray', linestyles=':', linewidth=1)

    # Force integer ticks on the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Axis labels
    ax.set_xlabel("Hour Period", fontsize=10)
    ax.set_ylabel("Day of Week", fontsize=10)

#7
def plot_most_busy_speakers(filtered_df, selected_Speaker, total_df):
    overall_total = total_df.shape[0]

    # Filter data
    if selected_Speaker == "Everyone":
        d = filtered_df
        top_n = 10
        counts = d.Speaker.value_counts().head(top_n)
    else:
        d = filtered_df[filtered_df.Speaker == selected_Speaker]
        counts = pd.Series([d.shape[0]], index=[selected_Speaker])

    # Percent of overall messages
    percents = (counts / overall_total * 100).round()
    perc_str = percents.map(lambda x: f"{x:.0f} ")

    fig, ax = plt.subplots(figsize=(9, 4))
    try:
        import seaborn as sns
        colors = sns.color_palette("rocket", len(counts))
    except ImportError:
        colors = None
    ax.bar(counts.index, counts.values, color=colors)

    for i, count in enumerate(counts.values):
        ax.annotate(
            f"{perc_str.iloc[i]}%",
            xy=(i, count),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center',
            fontsize=12
        )

    ax.set_xlabel("Speaker", fontsize=11)
    ax.set_ylabel("Number of Messages", fontsize=11)
    ax.set_title("Messages by Speaker", fontsize=12)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.margins(y=0.1)
    ax.tick_params(axis='x', rotation=60)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    return fig


#8
def create_wordcloud(selected_Speaker, df):
    try:
        # Filter by speaker if needed
        d = df if selected_Speaker == 'Everyone' else df[df['Speaker'] == selected_Speaker]

        # Only use actual message column
        if 'Translated_Message' in d.columns:
            texts = d['Translated_Message'].fillna("").astype(str)
        else:
            texts = d['Message'].fillna("").astype(str)

        # Drop messages that are only media placeholder
        texts = texts[~texts.str.fullmatch(r'\s*<Media omitted>\s*', na=False)]

        # Remove all placeholder inside messages
        cleaned = texts.str.replace(r'<Media omitted>', '', regex=True)

        # Remove all URLs
        url_pattern = r'(https?://\S+|www\.\S+)'
        cleaned = cleaned.str.replace(url_pattern, '', regex=True)

        # Explode to words, remove empty/space-only words
        words = cleaned.str.lower().str.split().explode()
        words = words[words.str.strip().astype(bool)]  # Only real words

        if words.empty:
            return None

        joined_text = " ".join(words.astype(str))

        if not joined_text.strip():
            return None

        wc = WordCloud(
            width=800,
            height=400,
            min_font_size=5,
            background_color='white'
        ).generate(joined_text)

        return wc
    except Exception as e:
        # Optionally log the error, for debugging:
        # import streamlit as st
        # st.write(f"Wordcloud generation error: {e}")
        return None

#9
def most_common_words(selected_Speaker, df):
    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]

    # Remove messages which are completely empty, media, or edit markers
    temp = df.copy()
    temp = temp[~temp['Message'].str.contains(r'^<Media omitted>.*$', regex=True)]
    temp['Message'] = temp['Message'].str.replace('<This message was edited>', '', regex=False)

    # Remove emojis
    def remove_emojis(text):
        return ''.join(ch for ch in text if not emoji.is_emoji(ch))
    temp['Cleaned'] = temp['Message'].apply(remove_emojis)

    # Tokenize
    words = []
    for message in temp['Cleaned']:
        for word in re.sub(r'<Media omitted>', '', message).lower().split():
            words.append(word)

    # Build DataFrame of top 20
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    most_common_df.columns = [0,1]
    return most_common_df


#10
def emoji_helper(selected_Speaker, df):
    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]
    
    emojis = []
    
    for Message in df['Message']:
        for c in str(Message):
            if emoji.is_emoji(c):
                emojis.append(c)
    
    emoji_counter = Counter(emojis)
    
    emoji_df = pd.DataFrame(emoji_counter.items(), columns=['Emoji', 'Count'])
    emoji_df = emoji_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    
    return emoji_df

def clean_text_for_lang_detection(text):
    # Remove links
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove emojis (very basic pattern, you can expand if needed)
    text = re.sub(r"[^\w\s,.!?]", "", text)
    # Remove extra whitespace
    return text.strip()

def detect_language_api(text):
    """
    Detects the dominant language in `text` using Gemini,
    returning only the language name from the supported set.
    On any error, returns a custom message.
    """
    try:
        if not text or not text.strip():
            return "Unknown"
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
        chat = model.start_chat(history=[])
        prompt = (
            "Given the following WhatsApp chat snippet, detect the *majority* language in which most words are written. "
            "Possible answers: English, Hindi, Hinglish, Marathi, Gujarati, Bengali, Punjabi, Tamil, Telugu, Kannada, Malayalam. "
            "Respond ONLY with the language name, nothing else:\n\n"
            f"{text}"
        )
        response = chat.send_message(prompt)
        lang = response.text.strip()
        # Post-process to clean possible Gemini extras
        try:
            data = json.loads(lang)
            for v in data.values():
                if isinstance(v, str):
                    return v.strip()
            return lang
        except Exception:
            if ":" in lang:
                lang = lang.split(":", 1)[-1].strip()
            lang = lang.strip("\"' ")
            return lang
    except Exception:
        return "Language Detection is taking too long. Meanwhile, explore other features!"


def get_dominant_language_single(df):
    """
    Uses the *first 5 cleaned messages* for a participant to detect dominant language.
    On any error, returns a custom message.
    """
    try:
        if df.empty:
            return "Unknown"
        msgs = df["Message"].dropna()
        # Filter out <Media omitted>
        msgs = msgs[~msgs.str.contains("<Media omitted>", na=False)]
        if msgs.empty:
            return "Unknown"
        # Clean each message, remove links and emojis
        cleaned = [clean_text_for_lang_detection(m) for m in msgs if m.strip()]
        cleaned = [m for m in cleaned if m]  # Only non-empty
        if not cleaned:
            return "Unknown"
        sample = cleaned[:5]
        prompt_text = " ".join(sample)
        return detect_language_api(prompt_text)
    except Exception:
        return "Language Detection is taking too long. Meanwhile, explore other features!"


def get_dominant_language_all(df):
    """
    Returns list of dicts: [{speaker, language}, ...] for all participants.
    On any error, returns a custom message.
    """
    try:
        results = []
        speakers = df["Speaker"].unique()
        for speaker in speakers:
            df_s = df[df["Speaker"] == speaker]
            lang = get_dominant_language_single(df_s)
            results.append({"speaker": speaker, "language": lang})
        return results
    except Exception:
        return "Language Detection is taking too long. Meanwhile, explore other features!"


#AI ANALYSIS

def summarize_conversation(conversation, participants, style="Detailed"):
    """
    style:
      - "Short": first 25% of sentences from full summary
      - "Concise": first 50% of sentences
      - "Detailed": entire summary
    """
    import math, re, json
    # 1) Filter by speaker
    if "Everyone" not in participants:
        conversation = [
            m for m in conversation if m["speaker"] in participants
        ]

    model = SummaryModel()

    def call_and_extract(conv):
        try:
            raw = model.predict_summmary({"conversation": conv})
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                    # Try both keys for backward-compatibility
                    return data.get("report", {}).get("summary") or data.get("summary", raw)
                except json.JSONDecodeError:
                    return raw
            else:
                return raw.get("report", {}).get("summary", "")
        except Exception:
            # Handles NotFound, quota, API, network, etc.
            return "Summary is taking too long to be generated. Meanwhile, explore other features!"

    # 2) Get full summary
    full_summary = call_and_extract(conversation).strip()
    if not full_summary:
        return ""

    # 3) Split into sentences
    sentences = re.split(r'(?<=[\.!?])\s+', full_summary)
    total = len(sentences)

    style_l = style.lower()
    if style_l == "short":
        count = max(1, math.ceil(0.25 * total))
    elif style_l == "concise":
        count = max(1, math.ceil(0.5 * total))
    else:  # detailed
        count = total

    selected = sentences[:count]
    return " ".join(selected).strip()


# configure once
genai.configure(api_key=GEMINI_API_KEY)

def translate_summary(text: str, target_lang: str) -> str:
    """
    Translates `text` into `target_lang` and returns only the translated text.
    For Hinglish, translates to Hindi in Roman (English) script.
    If API limit or any error occurs, returns a user-friendly message.
    """
    try:
        model = genai.GenerativeModel(
            model_name=DEFAULT_MODEL_NAME,
            generation_config=None,
            safety_settings=None
        )
        chat = model.start_chat(history=DEFAULT_MODEL_CHAT_HISTORY)

        if target_lang.lower() == "hinglish":
            prompt = (
                "Translate the following summary into Hindi written in English (Roman) script. "
                "For example, 'क्या कर रहा है?' should be translated as 'kya kar raha hai?'.\n"
                "Respond only with the translated text — no JSON, no quotes, no explanation:\n\n"
                f"{text}"
            )
        else:
            prompt = (
                f"Translate the following summary into {target_lang}.\n"
                "Respond only with the translated text — no JSON, no quotes, no explanation:\n\n"
                f"{text}"
            )

        response = chat.send_message(prompt)
        raw = response.text.strip()
        try:
            import json
            data = json.loads(raw)
            return data.get("summary", next(iter(data.values()), raw))
        except Exception:
            return raw
    except Exception:
        # Handles API quota, network, or other exceptions
        return "Summary is taking too long to be generated. Meanwhile, explore other features!"


    
def emotion_analysis(conversation, participants):
    """
    Returns a list of dicts:
      { speaker, primary_emotion, secondary_emotion }
    for the selected participants (or 'Everyone').
    If any error occurs, returns a special dict with a 'message' key.
    """
    from types import SimpleNamespace
    from collections import Counter

    # 1) Filter by participant
    if "Everyone" in participants:
        conv = conversation
    else:
        conv = [m for m in conversation if m["speaker"] in participants]

    # 2) Drop too-short messages
    dialogs = [
        SimpleNamespace(speaker=m["speaker"], message=m["message"])
        for m in conv
        if len(m["message"].split()) >= 3
    ]

    if not dialogs:
        return []  # nothing to analyze

    # 3) Call the emotion model, handling ALL exceptions
    model = EmotionModel()
    try:
        labels = model.predict_emotions(dialogs)
        # If the model returned a JSON string, parse out the list under "report"
        if isinstance(labels, str):
            parsed = json.loads(labels)
            labels = parsed.get("report", [])
        # 4) Count per speaker
        speaker_to_counter = {d.speaker: Counter() for d in dialogs}
        for label, dlg in zip(labels, dialogs):
            speaker_to_counter[dlg.speaker][label] += 1
        # 5) Build the final report
        report = []
        for speaker, counter in speaker_to_counter.items():
            top2 = counter.most_common(2)
            primary   = top2[0][0] if len(top2) >= 1 else None
            secondary = top2[1][0] if len(top2) >= 2 else None
            report.append({
                "speaker": speaker,
                "primary_emotion": primary,
                "secondary_emotion": secondary
            })
        return report
    except Exception:
        # Handles NotFound, API quota, timeout, etc.
        st.warning("Emotion analysis is taking too long. Meanwhile, explore other features!")
        # Optionally, return an empty list or a special error object
        return []


def relationship_analysis(conversation, participants, all_combinations=False):
    """
    Returns agreement scores for each speaker-pair.
    If all_combinations=False and exactly two participants are chosen,
    returns only that pair; otherwise returns all pairs.
    If any error occurs, shows a warning and returns [].
    """
    try:
        # prepare embeddings + clusters
        encoder   = TextEncoderModel()
        clusterer = TextClusteringModel()
        agreer    = TextAgreementModel()

        dialogs    = [m for m in conversation if len(m["message"].split()) >= 3]
        texts      = [m["message"] for m in dialogs]
        embeddings = encoder.calculate_embeddings(texts)
        labels     = clusterer.calculate_clusters(embeddings)

        # group messages by cluster label
        clusters = {}
        for dlg, lbl in zip(dialogs, labels):
            if lbl != -1:
                clusters.setdefault(lbl, []).append(dlg)

        speakers = set(m["speaker"] for m in conversation)
        from itertools import combinations
        pairs    = list(combinations(speakers, 2))

        results = []
        DECAY_FACTOR = 0.15  # Use your defined value if not global
        import math

        for sp1, sp2 in pairs:
            scores, weights = [], []
            for cluster in clusters.values():
                for i in range(len(cluster)):
                    for j in range(i+1, len(cluster)):
                        d1, d2 = cluster[i], cluster[j]
                        if {d1["speaker"], d2["speaker"]} == {sp1, sp2}:
                            lbl, sc = agreer.predict_category(d1["message"], d2["message"])
                            if lbl == 1:
                                sc = -sc
                            dist = abs(d1["index"] - d2["index"])
                            w    = math.exp(-DECAY_FACTOR * (dist - 1))
                            scores.append(sc * w)
                            weights.append(w)
            score = sum(scores)/sum(weights) if weights else 0.0
            results.append({
                "speaker1": sp1,
                "speaker2": sp2,
                "agreement_score": score
            })

        # if exactly two chosen & not all_combinations, filter to that pair
        if not all_combinations and len(participants) == 2:
            want = set(participants)
            results = [r for r in results
                       if set((r["speaker1"], r["speaker2"])) == want]

        return results
    except Exception:
        st.warning("Relationship analysis is taking too long. Meanwhile, explore other features!")
        return []
