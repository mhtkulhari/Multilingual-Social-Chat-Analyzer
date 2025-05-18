import re
import pandas as pd
import os
import json
import tempfile
import streamlit as st
from google.cloud import translate_v2 as translate
import html

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

def parse_whatsapp_chat(file_content):
    lines = file_content.splitlines()
    system_patterns = [
        r'created group', r'added', r'removed', r'left', r'changed the subject',
        r'changed this group\'s icon', r'changed the group description',
        r'changed the group settings', r'messages were deleted', r'encryption',
        # … (rest of your patterns)
    ]
    system_message_regex = re.compile('|'.join(system_patterns), re.IGNORECASE)

    def is_system_message(line):
        return bool(system_message_regex.search(line))

    def is_new_message(line):
        pattern = r'^\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?:\s?[ap]m)? - '
        return re.match(pattern, line) is not None

    # Merge multi-line messages
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

    # Parse into structured data
    chat_data = []
    for line in merged_lines:
        if is_system_message(line):
            continue
        match = re.match(
            r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?:\s?[ap]m)?) - (.*?): (.*)',
            line
        )
        if match:
            date, time, speaker, message = match.groups()
            chat_data.append({
                'Date': date, 'Time': time,
                'Speaker': speaker, 'Message': message
            })
    if not chat_data:
        return pd.DataFrame()

    df = pd.DataFrame(chat_data)
    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %I:%M %p',
        errors='coerce'
    )
    df['year']      = df['Datetime'].dt.year
    df['month_num'] = df['Datetime'].dt.month
    df['month']     = df['Datetime'].dt.strftime('%b')
    df['day']       = df['Datetime'].dt.day
    df['day_name']  = df['Datetime'].dt.strftime('%a')
    df['hour']      = df['Datetime'].dt.hour.apply(lambda x: str(x).zfill(2))
    df['minute']    = df['Datetime'].dt.minute
    df['period']    = df['hour'].apply(
        lambda x: f"{x}-{str(int(x)+1).zfill(2) if int(x)<23 else '00'}"
    )

    # Merge consecutive messages by same speaker within 10
