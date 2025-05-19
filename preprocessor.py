import re
import pandas as pd
import os
from google.cloud import translate_v2 as translate
import html
import streamlit as st
import json
import tempfile
# 1) Read the JSON blob from secrets
sa_json = st.secrets["gcp"]["service_account"]

# 2) Parse it and write to a temp file
sa_info = json.loads(sa_json)
with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as fp:
    json.dump(sa_info, fp)
    creds_path = fp.name

# 3) Point Google’s SDK at that file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# 4) Initialize the client
translate_client = translate.Client()

# Set up Google Cloud Translate credentials
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "steady-shard-458110-g8-3fb5c1444f42.json"
# translate_client = translate.Client()

def parse_whatsapp_chat(file_content):
    # Split the content into lines
    lines = file_content.splitlines()

    # Precompile system patterns
    system_patterns = [
        r'created group', r'added', r'removed', r'left', r'changed the subject', r"changed this group's icon",
        r'changed the group description', r'changed the group settings', r'messages were deleted', r'encryption',
        r"joined using this group's invite link", r'joined using an invite link', r'you were added', r'you were removed',
        r'you changed the subject', r"you changed this group's icon", r'you changed the group description',
        r'you changed the group settings', r'is now an admin', r'you are now an admin', r'blocked this contact',
        r'unblocked this contact', r'changed their phone number to a new number', r'this message was deleted',
        r'group video call started', r'group voice call started', r'missed group call', r'missed voice call',
        r'called you', r'you called', r'deleted this message', r'invited you to join the group', r'security code changed'
    ]
    sys_rx = re.compile('|'.join(system_patterns), re.IGNORECASE)

    def is_system_message(line):
        return bool(sys_rx.search(line))

    def is_new_message(line):
        return re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?:\s?[ap]m)? - ', line) is not None

    # Merge multi-line messages
    merged_lines = []
    buffer = ''
    for line in lines:
        line = line.strip()
        if is_new_message(line):
            if buffer:
                merged_lines.append(buffer)
            buffer = line
        else:
            buffer += ' ' + line
    if buffer:
        merged_lines.append(buffer)

    # Parse messages
    chat_data = []
    for entry in merged_lines:
        if is_system_message(entry):
            continue
        m = re.match(r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?:\s?[ap]m)?) - (.*?): (.*)', entry)
        if m:
            date, time, speaker, msg = m.groups()
            chat_data.append({'Date': date, 'Time': time, 'Speaker': speaker, 'Message': msg})

    if not chat_data:
        return pd.DataFrame()

    df = pd.DataFrame(chat_data)
    # Remove edit markers
    df['Message'] = df['Message'].str.replace('<This message was edited>', '', regex=False)

    # Combine Date and Time into Datetime
    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %I:%M %p',
        errors='coerce'
    )

    # Extract components
    df['year'] = df['Datetime'].dt.year
    df['month_num'] = df['Datetime'].dt.month
    df['month'] = df['Datetime'].dt.strftime('%b')
    df['day'] = df['Datetime'].dt.day
    df['day_name'] = df['Datetime'].dt.strftime('%a')
    df['hour'] = df['Datetime'].dt.hour.apply(lambda x: str(x).zfill(2))
    df['minute'] = df['Datetime'].dt.minute
    df['period'] = df['hour'].apply(lambda h: f"{h}-{str(int(h)+1).zfill(2)}" if int(h)<23 else f"{h}-00")

    # Merge consecutive messages
    merged = []
    current = df.iloc[0].copy()
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        same = current['Speaker']==row['Speaker'] and current['Date']==row['Date']
        diff = (row['Datetime']-current['Datetime']).total_seconds()/60
        if same and 0<=diff<=10:
            current['Message'] += '. ' + row['Message']
        else:
            merged.append(current)
            current = row.copy()
    merged.append(current)
    merged_df = pd.DataFrame(merged)

    # Translate
    merged_df['Translated_Message'] = merged_df['Message'].apply(translate_text_smart)

    # Return with Datetime
    return merged_df[['Date','Time','Speaker','Message','Translated_Message',
                      'year','month_num','month','day','day_name','hour','minute','period','Datetime']]  


def translate_text_smart(text):
    if pd.isnull(text) or not str(text).strip():
        return text
    try:
        det = translate_client.detect_language(text)['language']
        if det=='en':
            return text
        res = translate_client.translate(text, target_language='en')['translatedText']
        return html.unescape(res)
    except:
        return text
