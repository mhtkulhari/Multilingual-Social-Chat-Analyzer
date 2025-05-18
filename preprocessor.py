import re
import pandas as pd
import os, json, tempfile
import streamlit as st
from google.cloud import translate_v2 as translate
import html

# --- Google Translate setup ---
sa_info = json.loads(st.secrets["gcp"]["service_account"])
with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as fp:
    json.dump(sa_info, fp)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = fp.name
translate_client = translate.Client()

def parse_whatsapp_chat(content):
    lines = [l.strip() for l in content.splitlines()]
    system_patterns = [
        r'created group', r'added', r'removed', r'left', r'changed the subject',
        r'changed this group\'s icon', r'changed the group description',
        r'changed the group settings', r'messages were deleted', r'encryption',
        r'joined using this group\'s invite link', r'joined using an invite link',
        r'you were added', r'you were removed', r'you changed the subject',
        r'you changed this group\'s icon', r'you changed the group description',
        r'you changed the group settings', r'is now an admin',
        r'you are now an admin', r'blocked this contact', r'unblocked this contact',
        r'changed their phone number to a new number',
        r'this message was deleted', r'group video call started',
        r'group voice call started', r'missed group call', r'missed voice call',
        r'called you', r'you called', r'deleted this message',
        r'invited you to join the group', r'security code changed'
    ]
    sys_rx = re.compile('|'.join(system_patterns), re.IGNORECASE)

    def is_system(line):
        return bool(sys_rx.search(line))
    def is_new(line):
        return re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?:\s?[ap]m)? - ', line)

    # Merge multi‐line messages
    merged, curr = [], ""
    for l in lines:
        if is_new(l):
            if curr:
                merged.append(curr)
            curr = l
        else:
            curr += " " + l
    if curr:
        merged.append(curr)

    # Parse structured data
    chat_data = []
    for l in merged:
        if is_system(l):
            continue
        m = re.match(
            r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?:\s?[ap]m)?) - (.*?): (.*)',
            l
        )
        if m:
            date, time, speaker, msg = m.groups()
            chat_data.append({
                'Date': date, 'Time': time,
                'Speaker': speaker, 'Message': msg
            })

    if not chat_data:
        return pd.DataFrame()

    df = pd.DataFrame(chat_data)
    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %I:%M %p', errors='coerce'
    )
    df['year']      = df.Datetime.dt.year
    df['month_num'] = df.Datetime.dt.month
    df['month']     = df.Datetime.dt.strftime('%b')
    df['day']       = df.Datetime.dt.day
    df['day_name']  = df.Datetime.dt.strftime('%a')
    df['hour']      = df.Datetime.dt.hour.apply(lambda h: f"{h:02d}")
    df['minute']    = df.Datetime.dt.minute
    df['period']    = df.hour.apply(
        lambda h: f"{h}-{int(h)+1:02d}" if int(h) < 23 else "23-00"
    )

    # Merge short consecutive messages
    merged_rows, curr = [], df.iloc[0].copy()
    for i in range(1, len(df)):
        row = df.iloc[i]
        same = (curr.Speaker == row.Speaker) and (curr.Date == row.Date)
        diff = (row.Datetime - curr.Datetime).total_seconds() / 60
        if same and 0 <= diff <= 10:
            curr.Message += '. ' + row.Message
        else:
            merged_rows.append(curr)
            curr = row.copy()
    merged_rows.append(curr)
    mdf = pd.DataFrame(merged_rows)

    # Translate messages
    mdf['Translated_Message'] = mdf['Message'].apply(_translate_smart)

    cols = [
        'Date','Time','Speaker','Message','Translated_Message',
        'year','month_num','month','day','day_name','hour','minute','period'
    ]
    return mdf[cols]

def _translate_smart(text):
    if not text or pd.isnull(text):
        return text
    lang = translate_client.detect_language(text)['language']
    if lang == 'en':
        return text
    out = translate_client.translate(text, target_language='en')['translatedText']
    return html.unescape(out)
