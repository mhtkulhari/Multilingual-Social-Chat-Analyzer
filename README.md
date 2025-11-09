This project is a complete NLP-based WhatsApp Chat Analyzer that turns raw exported chats into meaningful insights.
Instead of just counting messages, it performs deep analysis: it shows activity timelines, busiest users, most used words, emoji patterns, language distribution, and per-user statistics.
What makes it unique is multilingual support—especially for Indian languages and code-mixed text. The system auto-detects the language, translates it if needed, and then runs the analysis, so even Hindi-English mixed chats work smoothly.
On top of analytics, it applies AI models to generate automatic conversation summaries, detect emotions for each participant, and even calculate relationship or agreement scores between users based on message similarity.
Everything is accessible through a Streamlit web app: you just upload a WhatsApp export file, and the dashboards, word clouds, summaries, and emotion charts are generated instantly.
The entire pipeline—parsing, preprocessing, translation, emotion model, summarizer, visualization—is modular, scalable, and can be extended to other platforms like Telegram.
So in short: upload chat → system cleans, translates, analyzes → outputs complete statistical, emotional, and social insights in a click.
