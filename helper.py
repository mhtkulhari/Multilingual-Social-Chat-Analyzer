#helper.py

from urlextract import URLExtract
import pandas as pd
from collections import Counter
import emoji
from wordcloud import WordCloud
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

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
