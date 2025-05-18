import streamlit as st
import os
import re
import pandas as pd
import html
import json
from google.cloud import translate_v2 as translate
from google.oauth2.service_account import Credentials

# --- Initialize translation client with service account from secrets ---
sa_json_str = st.secrets.get("gcp", {}).get("service_account")
if sa_json_str:
    try:
        sa_info = json.loads(sa_json_str)
        creds = Credentials.from_service_account_info(sa_info)
        translate_client = translate.Client(credentials=creds)
    except Exception as e:
        st.error("Failed to load GCP credentials: {}".format(e))
        translate_client = translate.Client()
else:
    # Fallback to default credentials (local env var)
    translate_client = translate.Client()


def parse_whatsapp_chat(file_content):
    # Split the content into lines
    lines = file_content.splitlines()

    # Precompile system patterns
    system_patterns = [
        r'created group', r'added', r'removed', r'left', r'changed the subject', r'changed this group\'s icon',
        r'changed the group description', r'changed the group settings', r'messages were deleted', r'encryption',
        r'joined using this group\'s invite link', r'joined using an invite link', r'you were added', r'you were removed',
        r'you changed the subject', r'you changed this group\'s icon', r'you changed the group description',
        r'you changed the group settings', r'is now an admin', r'you are now an admin', r'blocked this contact',
        r'unblocked this contact', r'changed their phone number to a new number', r'this message was deleted',
        r'group video call started', r'group voice call started', r'missed group call', r'missed voice call',
        r'called you', r'you called', r'deleted this message', r'invited you to join the group', r'security code changed',
    ]
    system_message_regex = re.compile('|'.join(system_patterns), re.IGNORECASE)

    def is_system_message(line):
        return bool(system_message_regex.search(line))

    def is_new_message(line):
        pattern = r'^\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?:\s?[ap]m)? - '
        return re.match(pattern, line) is not None

    # Step 1: Merge multi-line messages
    merged_lines = []
    current_message = ''

    for line in lines:
        line = line.strip()
        if is_new_message(line):
            if current_message:
                merged_lines.append(current_message)
            current_message = line
        else:
            current_message += ' ' + line

    if current_message:
        merged_lines.append(current_message)

    # Step 2: Parse into structured data
    chat_data = []

    for line in merged_lines:
        if is_system_message(line):
            continue

        match = re.match(r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?:\s?[ap]m)?) - (.*?): (.*)', line)
        if match:
            date, time, speaker, message = match.groups()
            chat_data.append({
                'Date': date,
                'Time': time,
                'Speaker': speaker,
                'Message': message
            })

    if not chat_data:
        return pd.DataFrame()

    df = pd.DataFrame(chat_data)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %I:%M %p', errors='coerce')
    df['year'] = df['Datetime'].dt.year
    df['month_num'] = df['Datetime'].dt.month
    df['month'] = df['Datetime'].dt.strftime('%b')
    df['day'] = df['Datetime'].dt.day
    df['day_name'] = df['Datetime'].dt.strftime('%a')
    df['hour'] = df['Datetime'].dt.hour.apply(lambda x: str(x).zfill(2))
    df['minute'] = df['Datetime'].dt.minute
    df['period'] = df['hour'].apply(lambda x: f"{x}-{str(int(x)+1).zfill(2) if int(x)<23 else '00'}")

    merged_chat = []
    current_row = df.iloc[0].copy()
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        if current_row['Speaker'] == row['Speaker'] and current_row['Date'] == row['Date']:
            time_diff = (row['Datetime'] - current_row['Datetime']).total_seconds() / 60.0
            if 0 <= time_diff <= 10:
                current_row['Message'] += '. ' + row['Message']
            else:
                merged_chat.append(current_row)
                current_row = row.copy()
        else:
            merged_chat.append(current_row)
            current_row = row.copy()
    merged_chat.append(current_row)
    merged_df = pd.DataFrame(merged_chat)

    # Translate to English
    merged_df['Translated_Message'] = merged_df['Message'].apply(translate_text_smart)
    return merged_df[['Date','Time','Speaker','Message','Translated_Message','year','month_num','month','day','day_name','hour','minute','period']]


def translate_text_smart(text):
    if pd.isnull(text) or not str(text).strip():
        return text
    try:
        det = translate_client.detect_language(text)
        if det.get('language') == 'en':
            return text
        res = translate_client.translate(text, target_language='en')
        return html.unescape(res.get('translatedText',''))
    except Exception:
        return text
