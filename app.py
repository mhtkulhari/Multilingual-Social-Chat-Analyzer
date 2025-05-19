import streamlit as st
import preprocessor
import helper
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, MaxNLocator
import calendar
import regex

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
        "Emoji Analysis"
    ]
    selected_analyses = st.sidebar.multiselect(
        "🔍 **Analysis Type(s)**",
        analysis_types,
        default=["Overall"]
    )

    if st.sidebar.button("**Run Analysis**"):
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

                if analysis in ["Overall", "Emoji Analysis"]:
                    emoji_df = helper.emoji_helper(participant, part_df)
                    st.subheader("😊 Emoji Analysis")
                    st.dataframe(emoji_df, hide_index=True)

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
