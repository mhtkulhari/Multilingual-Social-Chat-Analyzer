import re
import pandas as pd
import os, json, tempfile
from google.cloud import translate_v2 as translate
from tqdm import tqdm
import html

# --- Load Google Cloud credentials from Streamlit secrets ---
# Expecting the JSON string under st.secrets['gcp']['service_account']
sa_json_str = st.secrets.get("gcp", {}).get("service_account")
if sa_json_str:
    # Write JSON to a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as fp:
        fp.write(sa_json_str)
        creds_path = fp.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# Initialize the translate client
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

    # Functions inside the main function
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

    # If no valid chat data, return an empty dataframe
    if not chat_data:
        return pd.DataFrame()

    df = pd.DataFrame(chat_data)

    # Combine Date and Time columns to create a Datetime column
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %I:%M %p', errors='coerce')

    # Extract additional date/time components
    df['year'] = df['Datetime'].dt.year
    df['month_num'] = df['Datetime'].dt.month
    df['month'] = df['Datetime'].dt.strftime('%b')  # Month name (Jan, Feb, ...)
    df['day'] = df['Datetime'].dt.day
    df['day_name'] = df['Datetime'].dt.strftime('%a')  # Day name (Mon, Tue, ...)
    df['hour'] = df['Datetime'].dt.hour.apply(lambda x: str(x).zfill(2))
    df['minute'] = df['Datetime'].dt.minute
    # Create the 'period' column by combining hours in a formatted way
    df['period'] = df['hour'].apply(lambda x: f"{x}-{str(int(x) + 1).zfill(2) if int(x) < 23 else '00'}")
    

    # Step 3: Merge consecutive messages
    merged_chat = []
    current_row = df.iloc[0].copy()

    for idx in range(1, len(df)):
        row = df.iloc[idx]
        same_speaker = current_row['Speaker'] == row['Speaker']
        same_date = current_row['Date'] == row['Date']

        if same_speaker and same_date:
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

    # Now we create merged_df
    merged_df = pd.DataFrame(merged_chat)

    # Translate the messages to English
    merged_df['Translated_Message'] = merged_df['Message'].apply(translate_text_smart)

    # Select desired columns
    merged_df = merged_df[['Date', 'Time', 'Speaker', 'Message', 'Translated_Message', 'year', 'month_num', 'month', 'day', 'day_name', 'hour', 'minute','period']]

    return merged_df


# Function to translate text using Google Translate API
def translate_text_smart(text):
    if pd.isnull(text) or not str(text).strip():
        return text  # Skip empty or NaN

    try:
        # Detect language first
        detection = translate_client.detect_language(text)
        detected_lang = detection['language']

        if detected_lang == 'en':
            # Already English, no need to translate
            return text
        else:
            # Translate if not English
            result = translate_client.translate(text, target_language='en')
            translated_text = result['translatedText']
            return html.unescape(translated_text)

    except Exception as e:
        return text
