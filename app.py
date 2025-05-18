
import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

# Inject custom CSS (read as UTF-8)
with open("static/style.css", "r", encoding="utf-8") as f:
   css = f.read()
   st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# -- UI setup for font --
font_path = 'NotoSans-Regular.ttf'  # Replace with your font file name
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)

mpl.rcParams['font.family'] = font_prop.get_name()
mpl.rcParams['axes.unicode_minus'] = False  # To handle minus signs correctly


st.sidebar.title("SOCIAL CHAT ANALYZER")

uploaded_file = st.sidebar.file_uploader("Upload WhatsApp Chat")
if uploaded_file is not None:
    # Get file content as text
    file_content = uploaded_file.getvalue().decode("utf-8")
    
    # Pass the text content to the parsing function
    df = preprocessor.parse_whatsapp_chat(file_content)

    # Display the dataframe in Streamlit
    st.dataframe(df)

    # Fetch unique Speakers
    Speaker_list = df['Speaker'].unique().tolist()
    Speaker_list.sort()
    Speaker_list.insert(0, "Everyone")

    selected_Speaker = st.sidebar.selectbox("Select Participant", Speaker_list)

    if st.sidebar.button("Show Analysis"):

        msgs, words, media, links = helper.fetch_stats(selected_Speaker,df)
        st.title("🔢 Top Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Messages", msgs)
        c2.metric("Words", words)
        c3.metric("Media", media)
        c4.metric("Links", links)


        # --- Monthly Timeline (full-width) ---
        st.subheader("📈 Monthly Timeline")
        fig, ax = plt.subplots(figsize=(9, 4))   # wide (16), same height (4)
        t = helper.monthly_timeline(selected_Speaker, df)
        ax.plot(t['time'], t['Message'], marker='o', label='Messages per Month')
        ax.set_xlabel("Month-Year", fontsize=11)
        ax.set_ylabel("Number of Messages", fontsize=11)
        # Force integer y‐axis
        ax.set_ylim(0, t['Message'].max() + 1)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_yticklabels([f"{int(y)}" for y in ax.get_yticks()])
        ax.set_xticks(range(len(t['time'])))
        ax.set_xticklabels(t['time'], rotation=90, ha='right')

        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
        st.pyplot(fig, clear_figure=True)

        # --- Daily Timeline (full-width) ---
        st.subheader("📅 Daily Timeline")
        fig, ax = plt.subplots(figsize=(9, 4))   # wide (16), same height (4)
        d = helper.daily_timeline(selected_Speaker, df)
        ax.plot(d['Date'], d['Message'], marker='.', label='Messages per Day')
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Number of Messages", fontsize=11)
        # Force integer y‐axis
        ax.set_ylim(0, d['Message'].max() + 1)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_yticklabels([f"{int(y)}" for y in ax.get_yticks()])
        ax.set_xticks(d['Date'])
        ax.set_xticklabels(d['Date'].dt.strftime('%d-%b-%Y'), rotation=90, ha='right')

        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
        st.pyplot(fig, clear_figure=True)



        


        # activity map
        st.title('🗓️ Activity Map')

        st.subheader("Most busy day")
        fig, ax = plt.subplots(figsize=(9, 4))   # wide (16), same height (4)
        busy_day = helper.week_activity_map(selected_Speaker, df)
        ax.bar(busy_day.index, busy_day.values, color='purple')
        ax.set_xlabel("Day", fontsize=11)
        ax.set_ylabel("Number of Messages", fontsize=11)
        ax.set_title("Messages by Weekday", fontsize=11)
        # force integer y‐axis
        ax.set_ylim(0, busy_day.values.max() + 1)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_yticklabels([f"{int(y)}" for y in ax.get_yticks()])
        ax.set_xticklabels(busy_day.index, rotation='vertical')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig, clear_figure=True)

        st.subheader("Most busy month")
        fig, ax = plt.subplots(figsize=(9, 4))   # wide (16), same height (4)
        busy_month = helper.month_activity_map(selected_Speaker, df)
        ax.bar(busy_month.index, busy_month.values, color='orange')
        ax.set_xlabel("Month", fontsize=11)
        ax.set_ylabel("Number of Messages", fontsize=11)
        ax.set_title("Messages by Month", fontsize=11)
        # Force integer y‐axis
        ax.set_ylim(0, busy_month.values.max() + 1)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_yticklabels([f"{int(y)}" for y in ax.get_yticks()])
        ax.set_xticklabels(busy_month.index, rotation='vertical')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig, clear_figure=True)


        st.title("📰 Heatmap (Day vs Hour)")
        hm = helper.activity_heatmap(selected_Speaker, df)
        fig, ax = plt.subplots(figsize=(8,4))
        helper.plot_heatmap(hm, ax)
        st.pyplot(fig, clear_figure=True)


        # finding the busiest Speakers in the group(Group level)
        if selected_Speaker == 'Everyone':
            st.title('Most Busy Speakers')
            x,new_df = helper.most_busy_Speakers(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)


        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_Speaker,df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc, interpolation='bilinear')
        ax.axis('off')  # Important: Hide axis
        st.pyplot(fig)

        most_common_df = helper.most_common_words(selected_Speaker, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation=0)
        st.title('Most Common Words')
        st.pyplot(fig)



        # emoji analysis
        emoji_df = helper.emoji_helper(selected_Speaker,df)
        st.title("Emoji Analysis")
        st.dataframe(emoji_df, hide_index=True)

