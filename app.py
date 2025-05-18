#NLP-Driven Multilingual Social Chat Analyzer for Emotion Detection, Behavior Analysis & Relationship Insights

import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import matplotlib as mpl

with open("static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -- UI setup for font --
font_path = 'NotoSans-Regular.ttf'  # Replace with your font file name
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)

mpl.rcParams['font.family'] = font_prop.get_name()
mpl.rcParams['axes.unicode_minus'] = False  # To handle minus signs correctly


st.sidebar.title("SOCIAL CHAT ANALYZER")

uploaded_file = st.sidebar.file_uploader("Choose a file")
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
    Speaker_list.insert(0, "Overall")

    selected_Speaker = st.sidebar.selectbox("Show analysis wrt", Speaker_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_Speaker,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_Speaker,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['Message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_Speaker, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['Date'], daily_timeline['Message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)


        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_Speaker,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_Speaker, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        Speaker_heatmap = helper.activity_heatmap(selected_Speaker,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(Speaker_heatmap)
        st.pyplot(fig)

        # finding the busiest Speakers in the group(Group level)
        if selected_Speaker == 'Overall':
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
