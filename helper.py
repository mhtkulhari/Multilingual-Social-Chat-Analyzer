#helper.py

from urlextract import URLExtract
import pandas as pd
from collections import Counter
import emoji
from wordcloud import WordCloud
import re
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def fetch_stats(selected_Speaker,df):

    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]

    # fetch the number of Messages
    num_Messages = df.shape[0]

    # fetch the total number of words
    words = []
    for Message in df['Message']:
        words.extend(Message.split())

    # fetch number of media Messages
    num_media_Messages = df['Message'].apply(lambda msg: len(re.findall(r'<Media omitted>', msg))).sum()
    # fetch number of links shared
    extract = URLExtract()
    links = []
    for Message in df['Message']:
        links.extend(extract.find_urls(Message))

    return num_Messages,len(words),num_media_Messages,len(links)

def most_busy_Speakers(df):
    x = df['Speaker'].value_counts().head()
    df = round((df['Speaker'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'Speaker': 'percent'})
    return x,df

def create_wordcloud(selected_Speaker, df):
    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]

    temp = df.copy()
    temp = temp[temp['Translated_Message'] != '<Media omitted>']

    def clean_message(text):
        words = []
        for word in str(text).lower().split():
            words.append(word)
        return " ".join(words)

    combined_text = temp['Translated_Message'].apply(clean_message).str.cat(sep=" ")

    wc = WordCloud(
        width=800,
        height=400,
        min_font_size=5,
        background_color='white'
    ).generate(combined_text)

    return wc

def most_common_words(selected_Speaker, df):
    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]

    temp = df.copy()
    
    # Remove messages which are completely empty or just media
    temp = temp[~temp['Message'].str.contains(r'^<Media omitted>.*$', regex=True)]

    words = []
    for message in temp['Message']:
        # Remove all <Media omitted> inside messages
        cleaned_message = re.sub(r'<Media omitted>', '', message)
        
        for word in cleaned_message.lower().split():
            words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df



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


def monthly_timeline(selected_Speaker,df):

    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['Message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

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


def week_activity_map(selected_Speaker,df):

    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]

    return df['day_name'].value_counts()

def month_activity_map(selected_Speaker,df):

    if selected_Speaker != 'Everyone':
        df = df[df['Speaker'] == selected_Speaker]

    return df['month'].value_counts()

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

def plot_heatmap(pt, ax):
    # Convert to integers
    pt_int = pt.fillna(0).astype(int)
    max_count = pt_int.values.max()

    # Build annotation matrix: only the max per row
    annot = pd.DataFrame("", index=pt_int.index, columns=pt_int.columns)
    for day in pt_int.index:
        row = pt_int.loc[day]
        m = row.max()
        # mark all periods that hit the max (usually one)
        for period, val in row.items():
            if val == m and m > 0:
                annot.at[day, period] = str(m)

    sns.heatmap(
        pt_int,
        annot=annot,
        fmt="",
        cmap='Blues',
        vmin=0,
        vmax=max_count,
        cbar_kws={
            'label': 'Messages',
            'ticks': list(range(0, max_count + 1, max(1, max_count // 5)))
        },
        ax=ax
    )

    # Force integer colorbar ticks
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Labels
    ax.set_xlabel("Hour Period", fontsize=10)
    ax.set_ylabel("Day of Week", fontsize=10)
