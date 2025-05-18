from urlextract import URLExtract
import pandas as pd
from collections import Counter
import emoji
from wordcloud import WordCloud
import re

def fetch_stats(speaker, df):
    d = df if speaker == 'Overall' else df[df.Speaker == speaker]
    msgs = d.shape[0]
    words = sum(len(m.split()) for m in d.Message)
    media = d.Message.str.count(r'<Media omitted>').sum()
    extractor = URLExtract()
    links = sum(len(extractor.find_urls(m)) for m in d.Message)
    return msgs, words, media, links

def most_busy_Speakers(df):
    top5 = df.Speaker.value_counts().head()
    percent = (df.Speaker.value_counts(normalize=True) * 100)\
              .round(2).reset_index()
    percent.columns = ['name','percent']
    return top5, percent

def create_wordcloud(speaker, df):
    d = df if speaker == 'Overall' else df[df.Speaker == speaker]
    text = " ".join(
        d.Translated_Message.replace('<Media omitted>', '', regex=True)
         .str.lower().str.split().sum()
    )
    return WordCloud(
        width=800, height=400, min_font_size=5,
        background_color='black'
    ).generate(text)

def most_common_words(speaker, df):
    d = df if speaker == 'Overall' else df[df.Speaker == speaker]
    words = re.sub(r'<Media omitted>', '', " ".join(d.Message))\
             .lower().split()
    return pd.DataFrame(
        Counter(words).most_common(20),
        columns=['Word','Count']
    )

def emoji_helper(speaker, df):
    d = df if speaker == 'Overall' else df[df.Speaker == speaker]
    emojis = [c for m in d.Message for c in m if emoji.is_emoji(c)]
    df_e = pd.DataFrame(
        Counter(emojis).items(),
        columns=['Emoji','Count']
    )
    return df_e.sort_values('Count', ascending=False)\
               .reset_index(drop=True)

def monthly_timeline(speaker, df):
    d = df if speaker == 'Overall' else df[df.Speaker == speaker]
    tl = d.groupby(['year','month_num','month'])\
          .count()['Message'].reset_index()
    tl['time'] = tl.month + "-" + tl.year.astype(str)
    return tl

def daily_timeline(speaker, df):
    d = df if speaker == 'Overall' else df[df.Speaker == speaker]
    d.Date = pd.to_datetime(d.Date, format='%d/%m/%Y')
    return d.groupby('Date').count()['Message']\
            .reset_index().sort_values('Date')

def week_activity_map(speaker, df):
    d = df if speaker == 'Overall' else df[df.Speaker == speaker]
    return d.day_name.value_counts()

def month_activity_map(speaker, df):
    d = df if speaker == 'Overall' else df[df.Speaker == speaker]
    return d.month.value_counts()

def activity_heatmap(speaker, df):
    d = df if speaker == 'Overall' else df[df.Speaker == speaker]
    return d.pivot_table(
        index='day_name', columns='period',
        values='Message', aggfunc='count'
    ).fillna(0)
