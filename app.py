#app.py

import streamlit as st
import preprocessor
import helper
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import calendar
import regex

from matplotlib.ticker import MultipleLocator, MaxNLocator

# --- Register all the Noto Sans fonts you need ---
FONT_FILES = {
    'Devanagari': 'static/NotoSansDevanagari-Regular.ttf',
    'Gujarati':   'static/NotoSansGujarati-Regular.ttf',
    'Bengali':    'static/NotoSansBengali-Regular.ttf',
    'Gurmukhi':   'static/NotoSansGurmukhi-Regular.ttf',
    'Tamil':      'static/NotoSansTamil-Regular.ttf',
    'Telugu':     'static/NotoSansTelugu-Regular.ttf',
    'Kannada':    'static/NotoSansKannada-Regular.ttf',
    'Malayalam':  'static/NotoSansMalayalam-Regular.ttf',
    'Latin':      'static/NotoSans-Regular.ttf',
}
FONT_PROPS = {}
for script, path in FONT_FILES.items():
    fm.fontManager.addfont(path)
    FONT_PROPS[script] = fm.FontProperties(fname=path).get_name()

def choose_font_family(text: str):
    for script in ('Gujarati','Devanagari','Bengali','Tamil','Telugu','Kannada','Gurmukhi','Malayalam'):
        if regex.search(r'\p{{Script={}}}'.format(script), text):
            return FONT_PROPS[script]
    return FONT_PROPS['Latin']

with open("static/style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

mpl.rcParams['font.family'] = FONT_PROPS['Latin']
mpl.rcParams['axes.unicode_minus'] = False

st.sidebar.title("**SOCIAL CHAT ANALYZER**")
#st.sidebar.markdown("<br>", unsafe_allow_html=True)

# --- File uploader with expander, using session state ---
uploader_key = "whatsapp_chat_upload"

with st.sidebar.expander("📁 **Upload WhatsApp Chat Export**", expanded=st.session_state.get(uploader_key) is None):
    uploaded_file = st.file_uploader("", type=["txt"], key=uploader_key)
    if uploaded_file is not None:
        st.success("File uploaded successfully!")

uploaded_file = st.session_state.get(uploader_key)

df_placeholder = st.empty()  # Always define at the top

if uploaded_file is not None:
    file_content = uploaded_file.getvalue().decode("utf-8")
    df = preprocessor.parse_whatsapp_chat(file_content)

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    min_date = df['Datetime'].dt.date.min()
    max_date = df['Datetime'].dt.date.max()

    date_range = st.sidebar.date_input(
        "🗓️ **Date Range**",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
        st.sidebar.warning("⚠️ Please select complete date range (start–end)")
        st.stop()

    df_placeholder.dataframe(df)

    start_date, end_date = date_range

    df = df[
        (df['Datetime'].dt.date >= start_date) &
        (df['Datetime'].dt.date <= end_date)
    ]
    df_placeholder.dataframe(df)

    # Make a copy for AI analyses only
    df_analysis = df.copy()

    st.sidebar.markdown('<hr style="border:none;border-top:2.2px solid #334e68;margin:6px 0;">',unsafe_allow_html=True)

    # ==== Multiple Participant Selector ====
    participant_options = sorted(df['Speaker'].unique().tolist())
    participant_options.insert(0, "Everyone")
    selected_participants = st.sidebar.multiselect(
        "👤 **Select Participant(s)**",
        participant_options,
        default=["Everyone"]
    )

    # ==== Multiple Analysis Type Selector ====
    analysis_types = [
        "Overall",
        "General Stats & Timelines",
        "Most Busy Day/Month",
        "Online Status",
        "Most Busy Speakers",
        "Most Common Words",
        "Emoji Analysis",
        "Dominant Language"
    ]
    selected_analyses = st.sidebar.multiselect(
        "🔍 **Analysis Type(s)**",
        analysis_types,
        default=["Overall"]
    )
    run = st.sidebar.button("**Run Analysis**")
    
    # --- NEW: AI Buttons in Sidebar ---
    st.sidebar.markdown('<hr style="border:none;border-top:2.2px solid #334e68;margin:6px 0;">', unsafe_allow_html=True)

    # — NEW: Summary Language selector —
    summary_language = st.sidebar.selectbox(
        "🌐 **Summary Language**",
        [
            "English","Hindi","Hinglish","Marathi","Gujarati","Bengali", 
            "Punjabi","Tamil","Telugu","Kannada","Malayalam"
        ],
        index=0,
        key="summary_language"
    )
    # — end language selector —


    # SUMMARY with participant selector
    summary_style = st.sidebar.radio(
        "✏️ **Summary Style**",
        ["Short", "Concise", "Detailed"],
        index=2,                    # default to "Concise"
        key="summary_style",
        horizontal=True
    )
    summary_parts = st.sidebar.multiselect(
        "📝 **Summary Participant(s)**",
        participant_options,
        default=["Everyone"],
        key="summary_select"
    )
    summary_all = st.sidebar.checkbox("All Participants", key="summary_all")
    summary_btn = st.sidebar.button("SUMMARY")

    st.sidebar.markdown('<hr style="border:none;border-top:2.2px solid #334e68;margin:6px 0;">', unsafe_allow_html=True)

    # EMOTION ANALYSIS
    emo_parts = st.sidebar.multiselect(
        "🎭 **Emotion Analysis Participant(s)**",
        participant_options,
        default=["Everyone"],
        key="emo_select"
    )
    run_emo_btn = st.sidebar.button("**EMOTION ANALYSIS**")

    st.sidebar.markdown('<hr style="border:none;border-top:2.2px solid #334e68;margin:6px 0;">', unsafe_allow_html=True)

    # RELATIONSHIP ANALYSIS
    rel_choices = [p for p in participant_options if p != "Everyone"]
    rel_parts = st.sidebar.multiselect(
        "🤝 **Choose 2 Participants for Relationship**",
        rel_choices,
        key="rel_select"
    )
    all_combo = st.sidebar.checkbox("All combinations")
    run_rel_btn = st.sidebar.button("**RELATIONSHIP**")

    st.sidebar.markdown('<hr style="border:none;border-top:2.2px solid #334e68;margin:6px 0;">', unsafe_allow_html=True)

    
    # === EXISTING ANALYSIS ===
    if run:
        df_placeholder.empty()

        for analysis in selected_analyses:
            # Determine participants to compare (handles "Everyone" logic too)
            participants_to_compare = []
            if len(selected_participants) == 1:
                participants_to_compare = selected_participants
            else:
                # Preserve order: "Everyone" first if present, then others
                participants_to_compare = [p for p in selected_participants if p == "Everyone"] + \
                                         [p for p in selected_participants if p != "Everyone"]

            if len(participants_to_compare) == 1:
                # SINGLE OUTPUT CASE
                participant = participants_to_compare[0]
                part_df = df if participant == "Everyone" else df[df["Speaker"] == participant]
                st.markdown(f"<span class='participant-label'>👤 Participant:  </span><span class='participant-title'>{participant}</span>", unsafe_allow_html=True)

                if analysis in ["Overall", "General Stats & Timelines"]:
                    msgs, media, links, longest_msg, avg_words, active_days, top_emojis = helper.fetch_stats(participant, part_df)
                    st.subheader("🔢 General Stats")
                    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                    c1.metric("Messages", msgs)
                    c4.metric("Media Shared", media)
                    c5.metric("Links Shared", links)
                    c3.metric("Longest Msg", longest_msg)
                    c2.metric("Avrg Words/Msg", avg_words)
                    c6.metric("Active Days", active_days)
                    if top_emojis and len(top_emojis) > 0:
                        top_emoji = top_emojis[0][0]
                        c7.metric("Top Emoji", f"{top_emoji}")
                    else:
                        c7.metric("Top Emoji", "-")

                    st.subheader("📈 Monthly Timeline")
                    fig, ax = plt.subplots(figsize=(9, 4))
                    t = helper.monthly_timeline(participant, part_df)
                    ax.plot(t['time'], t['Message'], marker='o', label='Messages per Month')
                    ax.set_xlabel("Month-Year", fontsize=11)
                    ax.set_ylabel("Number of Messages", fontsize=11)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    ax.set_xticks(range(len(t['time'])))
                    ax.set_xticklabels(t['time'], rotation=90, ha='right')
                    ax.legend(fontsize=9)
                    st.pyplot(fig, clear_figure=True)

                    st.subheader("📅 Daily Timeline")
                    fig, ax = plt.subplots(figsize=(9, 4))
                    d = helper.daily_timeline(participant, part_df)
                    ax.plot(d['Date'], d['Message'], marker='.', label='Messages per Day')
                    ax.set_xlabel("Date", fontsize=11)
                    ax.set_ylabel("Number of Messages", fontsize=11)
                    ax.set_ylim(0, d['Message'].max() + 1)
                    ax.yaxis.set_major_locator(MultipleLocator(1))
                    ax.set_yticklabels([f"{int(y)}" for y in ax.get_yticks()])
                    max_ticks = 10
                    step = max(1, len(d) // max_ticks)
                    sampled = d['Date'].iloc[::step]
                    ax.set_xticks(sampled)
                    ax.set_xticklabels(sampled.dt.strftime('%d-%b-%Y'), rotation=90, ha='right')
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    ax.legend(fontsize=9)
                    st.pyplot(fig, clear_figure=True)

                if analysis in ["Overall", "Most Busy Day/Month"]:
                    st.subheader('🗓️ Activity Map')
                    st.markdown("<span style='font-size:1.3rem; font-weight:600;'>Most busy day</span>",unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(9, 4))
                    busy_day = helper.week_activity_map(participant, part_df)
                    st.write(f"(**{dict(zip(calendar.day_abbr, calendar.day_name)).get(busy_day.idxmax(), busy_day.idxmax())}** with {busy_day.max()} messages)")
                    ax.bar(busy_day.index, busy_day.values, color='purple')
                    ax.set_xlabel("Day", fontsize=11)
                    ax.set_ylabel("Number of Messages", fontsize=11)
                    ax.set_title("Messages by Weekday", fontsize=11)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
                    ax.set_xticklabels(busy_day.index, rotation='vertical')
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    st.pyplot(fig, clear_figure=True)

                    st.markdown("<span style='font-size:1.3rem; font-weight:600;'>Most busy month</span>",unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(9, 4))
                    busy_month = helper.month_activity_map(participant, part_df)
                    st.write(f"(**{dict(zip(calendar.month_abbr[1:], calendar.month_name[1:])).get(busy_month.idxmax(), busy_month.idxmax())}** with {busy_month.max()} messages)")
                    ax.bar(busy_month.index, busy_month.values, color='orange')
                    ax.set_xlabel("Month", fontsize=11)
                    ax.set_ylabel("Number of Messages", fontsize=11)
                    ax.set_title("Messages by Month", fontsize=11)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
                    ax.set_xticklabels(busy_month.index, rotation='vertical')
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    st.pyplot(fig, clear_figure=True)

                if analysis in ["Overall", "Online Status"]:
                    st.subheader("📰 Online Status")
                    hm = helper.activity_heatmap(participant, part_df)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    helper.plot_heatmap(hm, ax)
                    st.pyplot(fig, clear_figure=True)

                if analysis in ["Overall", "Most Busy Speakers"]:
                    st.subheader("📊 Most Busy Speakers")
                    fig = helper.plot_most_busy_speakers(part_df, participant, df)
                    st.pyplot(fig, clear_figure=True)

                if analysis in ["Overall", "Most Common Words"]:
                    st.subheader("☁️ Wordcloud")
                    df_wc = helper.create_wordcloud(participant, part_df)
                    if df_wc is not None:
                        fig, ax = plt.subplots()
                        ax.imshow(df_wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.info("Insufficient data to generate wordcloud.")

                    st.subheader("🔤 Most Common Words")
                    most_common_df = helper.most_common_words(participant, part_df)
                    df_sorted = most_common_df.sort_values(by=1, ascending=False)
                    preference = [
                        'Gujarati','Devanagari','Bengali',
                        'Tamil','Telugu','Kannada','Gurmukhi','Malayalam'
                    ]
                    fams = [FONT_PROPS[script] for script in preference] + [FONT_PROPS['Latin']]
                    mpl.rcParams['font.family'] = fams

                    fig, ax = plt.subplots()
                    ax.barh(df_sorted[0], df_sorted[1])
                    ax.invert_yaxis()
                    ax.xaxis.set_ticks_position('top')
                    ax.xaxis.set_label_position('top')
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.set_xlabel("Frequency", fontsize=11)
                    plt.xticks(rotation=0, fontsize=9)
                    st.pyplot(fig)
                    mpl.rcParams['font.family'] = FONT_PROPS['Latin']

                if analysis in ["Overall", "Dominant Language"]:
                    st.subheader("🌐 Dominant Language")
                    # Show table if "Everyone", else show only the text
                    if participant == "Everyone":
                        langs = helper.get_dominant_language_all(df)
                        # Count frequencies and order by most frequent language
                        lang_counts = pd.Series([row["language"] for row in langs]).value_counts()
                        order = list(lang_counts.index)
                        # Move Unknown to end
                        if "Unknown" in order:
                            order.remove("Unknown")
                            order.append("Unknown")
                        df_lang = pd.DataFrame(langs).rename(columns={"speaker": "Participant", "language": "Language"})
                        df_lang["Language"] = pd.Categorical(df_lang["Language"], categories=order, ordered=True)
                        df_lang = df_lang.sort_values("Language")
                        st.dataframe(df_lang, hide_index=True)
                    else:
                        lang = helper.get_dominant_language_single(part_df)
                        st.write( f"**{participant}** mostly used <u><i>{lang}</i></u>.",unsafe_allow_html=True)

            else:
                # MULTI-PARTICIPANT SIDE-BY-SIDE OUTPUT CASE
                num_participants = len(participants_to_compare)
                cols = st.columns(num_participants)
                for idx, participant in enumerate(participants_to_compare):
                    part_df = df if participant == "Everyone" else df[df["Speaker"] == participant]
                    with cols[idx]:
                        st.markdown(f"<span class='participant-label'>👤 Participant:  </span><span class='participant-title'>{participant}</span>", unsafe_allow_html=True)

                        if analysis in ["Overall", "General Stats & Timelines"]:
                            msgs, media, links, longest_msg, avg_words, active_days, top_emojis = helper.fetch_stats(participant, part_df)
                            st.subheader("🔢 General Stats")
                            # Prepare data as a list of tuples: (Stat Name, Value)
                            top_emoji = top_emojis[0][0] if top_emojis and len(top_emojis) > 0 else "-"
                            stats = [
                                ("Total Messages", msgs),
                                ("Avrg Words/Msg", avg_words),
                                ("Longest Msg", longest_msg),
                                ("Media Shared", media),
                                ("Links Shared", links),
                                ("Active Days", active_days),
                                ("Top Emoji", top_emoji),
                            ]
                            stats_df = pd.DataFrame(stats, columns=["Stats", "Count"])
                            st.dataframe(stats_df, hide_index=True)
                
                            st.subheader("📈 Monthly Timeline")
                            fig, ax = plt.subplots(figsize=(6, 2.5))
                            t = helper.monthly_timeline(participant, part_df)
                            ax.plot(t['time'], t['Message'], marker='o', label='Messages per Month')
                            ax.set_xlabel("Month-Year", fontsize=11)
                            ax.set_ylabel("Messages", fontsize=11)
                            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                            ax.set_xticks(range(len(t['time'])))
                            ax.set_xticklabels(t['time'], rotation=90, ha='right', fontsize=8)
                            ax.grid(axis='y', linestyle='--', alpha=0.5)
                            ax.legend(fontsize=8)
                            st.pyplot(fig, clear_figure=True)

                            st.subheader("📅 Daily Timeline")
                            fig, ax = plt.subplots(figsize=(6, 2.5))
                            d = helper.daily_timeline(participant, part_df)
                            ax.plot(d['Date'], d['Message'], marker='.', label='Messages per Day')
                            ax.set_xlabel("Date", fontsize=11)
                            ax.set_ylabel("Messages", fontsize=11)
                            ax.set_ylim(0, d['Message'].max() + 1)
                            ax.yaxis.set_major_locator(MultipleLocator(1))
                            max_ticks = 6
                            step = max(1, len(d) // max_ticks)
                            sampled = d['Date'].iloc[::step]
                            ax.set_xticks(sampled)
                            ax.set_xticklabels(sampled.dt.strftime('%d-%b'), rotation=90, ha='right', fontsize=8)
                            ax.grid(axis='y', linestyle='--', alpha=0.5)
                            ax.legend(fontsize=8)
                            st.pyplot(fig, clear_figure=True)

                        if analysis in ["Overall", "Most Busy Day/Month"]:
                            st.subheader('🗓️ Activity Map')
                            st.markdown("<span style='font-size:1.1rem; font-weight:600;'>Most busy day</span>",unsafe_allow_html=True)
                            fig, ax = plt.subplots(figsize=(6, 2.5))
                            busy_day = helper.week_activity_map(participant, part_df)
                            st.write(f"(**{dict(zip(calendar.day_abbr, calendar.day_name)).get(busy_day.idxmax(), busy_day.idxmax())}** with {busy_day.max()} messages)")
                            ax.bar(busy_day.index, busy_day.values, color='purple')
                            ax.set_xlabel("Day", fontsize=10)
                            ax.set_ylabel("Number of Messages", fontsize=10)
                            ax.set_title("Messages by Weekday", fontsize=10)
                            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                            ax.set_xticklabels(busy_day.index, rotation='vertical', fontsize=9)
                            ax.grid(axis='y', linestyle='--', alpha=0.5)
                            st.pyplot(fig, clear_figure=True)

                            st.markdown("<span style='font-size:1.1rem; font-weight:600;'>Most busy month</span>",unsafe_allow_html=True)
                            fig, ax = plt.subplots(figsize=(6, 2.5))
                            busy_month = helper.month_activity_map(participant, part_df)
                            st.write(f"(**{dict(zip(calendar.month_abbr[1:], calendar.month_name[1:])).get(busy_month.idxmax(), busy_month.idxmax())}** with {busy_month.max()} messages)")
                            ax.bar(busy_month.index, busy_month.values, color='orange')
                            ax.set_xlabel("Month", fontsize=10)
                            ax.set_ylabel("Number of Messages", fontsize=10)
                            ax.set_title("Messages by Month", fontsize=10)
                            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                            ax.set_xticklabels(busy_month.index, rotation='vertical', fontsize=9)
                            ax.grid(axis='y', linestyle='--', alpha=0.5)
                            st.pyplot(fig, clear_figure=True)

                        if analysis in ["Overall", "Online Status"]:
                            st.subheader("📰 Online Status")
                            hm = helper.activity_heatmap(participant, part_df)
                            st.markdown( f"(Highly Active from {helper.period_24_to_12(hm.sum(axis=0).idxmax())})", unsafe_allow_html=True)
                            fig, ax = plt.subplots(figsize=(5, 2.5))
                            helper.plot_heatmap(hm, ax)
                            st.pyplot(fig, clear_figure=True)

                        if analysis in ["Overall", "Most Busy Speakers"]:
                            st.subheader("📊 Most Busy Speakers")
                            fig = helper.plot_most_busy_speakers(part_df, participant, df)
                            st.pyplot(fig, clear_figure=True)

                        if analysis in ["Overall", "Most Common Words"]:
                            st.subheader("☁️ Wordcloud")
                            df_wc = helper.create_wordcloud(participant, part_df)
                            if df_wc is not None:
                                fig, ax = plt.subplots()
                                ax.imshow(df_wc, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
                            else:
                                st.info("Insufficient data to generate wordcloud.")

                            st.subheader("🔤 Most Common Words")
                            most_common_df = helper.most_common_words(participant, part_df)
                            df_sorted = most_common_df.sort_values(by=1, ascending=False)
                            preference = [
                                'Gujarati','Devanagari','Bengali',
                                'Tamil','Telugu','Kannada','Gurmukhi','Malayalam'
                            ]
                            fams = [FONT_PROPS[script] for script in preference] + [FONT_PROPS['Latin']]
                            mpl.rcParams['font.family'] = fams

                            fig, ax = plt.subplots()
                            ax.barh(df_sorted[0], df_sorted[1])
                            ax.invert_yaxis()
                            ax.xaxis.set_ticks_position('top')
                            ax.xaxis.set_label_position('top')
                            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                            ax.set_xlabel("Frequency", fontsize=10)
                            plt.xticks(rotation=0, fontsize=8)
                            st.pyplot(fig)
                            mpl.rcParams['font.family'] = FONT_PROPS['Latin']

                        if analysis in ["Overall", "Emoji Analysis"]:
                            emoji_df = helper.emoji_helper(participant, part_df)
                            st.subheader("😊 Emoji Analysis")
                            st.dataframe(emoji_df, hide_index=True)


                        if analysis in ["Overall", "Dominant Language"]:
                            st.subheader("🌐 Dominant Language")
                            
                            if participant == "Everyone":
                                # Only "Everyone" column gets the DataFrame
                                langs = helper.get_dominant_language_all(df)
                                lang_counts = pd.Series([row["language"] for row in langs]).value_counts()
                                order = list(lang_counts.index)
                                if "Unknown" in order:
                                    order.remove("Unknown")
                                    order.append("Unknown")
                                df_lang = pd.DataFrame(langs).rename(columns={"speaker": "Participant", "language": "Language"})
                                df_lang["Language"] = pd.Categorical(df_lang["Language"], categories=order, ordered=True)
                                df_lang = df_lang.sort_values("Language")
                                st.dataframe(df_lang, hide_index=True)
                            else:
                                # Individual column: show only that participant's dominant language
                                lang = helper.get_dominant_language_single(part_df)
                                st.write(
                                    f"**{participant}** mostly used <u><i>{lang}</i></u>.",
                                    unsafe_allow_html=True
                                )


            pass

    # === BUILD CONVERSATION FOR AI ===
    df_ai = preprocessor.clean_translated_messages(df_analysis)
    conversation = [
        {"speaker": row["Speaker"],
         "message": row["Translated_Message"],
         "index": int(row["msg_index"])}
        for _, row in df_ai.iterrows()
    ]

    if summary_btn:
        df_placeholder.empty()
        st.subheader("📝 Summary")
        if summary_all:
            parts = [p for p in participant_options if p != "Everyone"]
        else:
            parts = summary_parts

        for part in parts:
            if part == "Everyone":
                part_conv = conversation
            else:
                part_conv = [m for m in conversation if m["speaker"] == part]

            # 1) generate in‐style summary
            part_summary = helper.summarize_conversation(
                part_conv, [part], summary_style
            )

            # 2) translate if needed
            if summary_language != "English" and part_summary:
                part_summary = helper.translate_summary(
                    part_summary, summary_language
                )

             # 3) render
            st.markdown(
                f"<span class='participant-label'>👤 Participant:  </span>"
                f"<span class='participant-title'>{part}</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='ai-summary-box'>{part_summary}</div>",
                unsafe_allow_html=True
            )

    
    if run_emo_btn:
        df_placeholder.empty()
        st.subheader("🎭 Emotion Analysis")

        
        # 1) Clean & prepare the full conversation list
        df_clean = preprocessor.clean_translated_messages(df_analysis)
        conversation_list = [
            {
                "speaker": r["Speaker"],
                "message": r["Translated_Message"],
                "index": int(r["msg_index"])
            }
            for _, r in df_clean.iterrows()
        ]

        # 2) Determine helper argument
        participants_arg = ["Everyone"] if "Everyone" in emo_parts else emo_parts

        # 3) Call helper once
        emos = helper.emotion_analysis(conversation_list, participants_arg)

        # 4) Build DataFrame & drop any duplicate speaker rows
        df_emos = pd.DataFrame(emos).drop_duplicates(subset="speaker")
        df_emos = df_emos.rename(columns={
            "speaker": "Participant",
            "primary_emotion": "Primary Emotion",
            "secondary_emotion": "Secondary Emotion"
        })

        # 5) Display table without index
        st.dataframe(df_emos, hide_index=True)


        # 6) Write follow-up sentences by column name
        for _, row in df_emos.iterrows():
            p  = row["Participant"]
            pe = row["Primary Emotion"] or "no clear primary emotion"
            se = row["Secondary Emotion"]
            if se:
                st.markdown(f"""<p style="font-size:18px; line-height:1.3;"><strong>{p}</strong> is feeling <em><u>{pe}</u></em> and <em><u>{se}</u></em>.</p>""",unsafe_allow_html=True)
            else:
                st.markdown(f"""<p style="font-size:18px; line-height:1.3;"><strong>{p}</strong> is feeling <em><u>{pe}</u></em>.</p>""",unsafe_allow_html=True)


# — RELATIONSHIP ANALYSIS —
    if run_rel_btn:
        df_placeholder.empty()
        st.subheader("🔗 Relationship Analysis")

        intervals = [-1, -0.8, -0.1, 0.1, 0.8, 1]
        labels = [
            "Strongly Disagree\n[-1, -0.8]",
            "Disagree\n(-0.8, -0.1]",
            "Neutral\n(-0.1, 0.1)",
            "Agree\n[0.1, 0.8)",
            "Strongly Agree\n[0.8, 1]"
        ]
        colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']

        fig, ax = plt.subplots(figsize=(10, 1.5))
        for i in range(5):
            left = intervals[i]
            width = intervals[i+1] - intervals[i]
            ax.barh(0, width, left=left, height=1, color=colors[i])

        for i, lbl in enumerate(labels):
            mid = (intervals[i] + intervals[i+1]) / 2
            ax.text(mid, 0, lbl, va='center', ha='center', fontsize=9)

        ax.set_xlim(-1, 1)
        ax.axis('off')
        st.pyplot(fig)


        # 1) Build a single conversation list (same as emotion)
        df_clean = preprocessor.clean_translated_messages(df_analysis)
        conversation_list = [
            {"speaker": r["Speaker"], "message": r["Translated_Message"], "index": int(r["msg_index"])}
            for _, r in df_clean.iterrows()
        ]

        # 2) Helper to map score → label
        def describe(score: float) -> str:
            if score >= 0.8:
                return "Strongly Agree"
            elif 0.1 <= score < 0.8:
                return "Agree"
            elif -0.1 < score < 0.1:
                return "Neutral"
            elif -0.8 < score <= -0.1:
                return "Disagree"
            else:
                return "Strongly Disagree"

        # 3) Compute relationships
        rels = []
        if len(rel_parts) == 1:
            # one participant => pair with every other speaker
            sp1 = rel_parts[0]
            others = [p for p in participant_options if p not in ("Everyone", sp1)]
            for sp2 in others:
                pair = helper.relationship_analysis(conversation_list, [sp1, sp2], False)
                rels.extend(pair)
        else:
            # two or more => use helper directly
            rels = helper.relationship_analysis(conversation_list, rel_parts, all_combo)

        # 4) Display
        for r in rels:
            s1 = r["speaker1"]
            s2 = r["speaker2"]
            score = r["agreement_score"]
            label = describe(score)

            st.markdown(f"""<p style="font-size:22px; line-height:1.3;"><strong>{s1}</strong> and <strong>{s2}</strong> <em>{label}</em> with each other.<br><u>[ Agreement Score is {score:.4f} ]</u></p>""",unsafe_allow_html=True)
