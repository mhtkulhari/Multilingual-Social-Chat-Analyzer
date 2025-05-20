import json
import math
from collections import Counter
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from .data_models.summary_report_model import SummaryReport, TranscriptSummary
from .data_models.conversation_model import Transcript
from .data_models.emotions_report_model import EmotionsReport, SpeakerEmotions
from .data_models.speaker_relationship_report import SpeakerRelationship, SpeakerRelationshipsReport
from .ml_models.summary_model.model import SummaryModel
from .ml_models.emotion_model.model import EmotionModel
from .ml_models.encoder_model.model import TextEncoderModel 
from .ml_models.clustering_model.model import TextClusteringModel
from .ml_models.agreement_model.model import TextAgreementModel
from .config import DECAY_FACTOR
from fastapi.middleware.cors import CORSMiddleware




load_dotenv()

summary_model = SummaryModel()
emotion_model = EmotionModel()
text_encoder_model = TextEncoderModel()
text_clustering_model = TextClusteringModel()
text_agremment_model = TextAgreementModel()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins for production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get('/')
async def health_check():
    return {'status': 'OK'}

@app.post('/summarize')
async def summarize_transcript(transcript : Transcript) -> SummaryReport:
    print(transcript)
    summary = summary_model.predict_summmary(transcript=transcript)
    summary = json.loads(summary)
    summary = TranscriptSummary(**summary)
    summary_report = SummaryReport(report=summary)
    return summary_report

@app.post('/emotions')
async def emotions_analysis(transcript : Transcript) -> EmotionsReport:
    conversation = transcript.conversation
    speakers = set([dialog.speaker for dialog in conversation])
    dialogs = [i for i in conversation if len(i.message.split()) >= 3]
    emotions = emotion_model.predict_emotions(dialogs)
    speaker_to_emotion_mapper = {speaker: Counter() for speaker in speakers}
    for emotion, dialog in zip(emotions, dialogs):
        speaker_to_emotion_mapper[dialog.speaker][emotion] += 1
    data = []
    for speaker, emotions_counter in speaker_to_emotion_mapper.items():
        speaker_top_emotions = emotions_counter.most_common(2)
        while len(speaker_top_emotions) < 2:
            speaker_top_emotions.append((None, None))
        primary_emotion, secondary_emotion = speaker_top_emotions
        data.append(SpeakerEmotions(speaker=speaker, 
                                    primary_emotion=primary_emotion[0],
                                    secondary_emotion=secondary_emotion[0]))
    emotions_report = EmotionsReport(report=data)
    return emotions_report

@app.post('/relationship')
async def analyse_speaker_relationships(transcript: Transcript) -> SpeakerRelationshipsReport:
    from itertools import combinations

    conversation = transcript.conversation
    speakers = list(set(dialog.speaker for dialog in conversation))
    dialogs = [dialog for dialog in conversation if len(dialog.message.split()) >= 3]

    # Embed texts for clustering
    texts = [dialog.message for dialog in dialogs]
    embeddings = text_encoder_model.calculate_embeddings(texts)
    cluster_labels = text_clustering_model.calculate_clusters(embeddings)

    # Group dialogs by clusters
    clusters = {}
    for dialog, label in zip(dialogs, cluster_labels):
        if label != -1:
            clusters.setdefault(label, []).append(dialog)

    # Calculate agreement scores for unique speaker pairs
    relationships = []
    speaker_pairs = list(combinations(speakers, 2))  # Generate unique pairs of speakers

    for speaker1, speaker2 in speaker_pairs:
        agreement_scores = []
        max_possible_scores = []

        # Generate dialog pairs involving speaker1 and speaker2
        dialog_pairs = []
        for cluster in clusters.values():
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    dialog1, dialog2 = cluster[i], cluster[j]
                    if {
                        dialog1.speaker, dialog2.speaker
                    } == {speaker1, speaker2}:
                        dialog_pairs.append((dialog1, dialog2))

        # Compute agreement scores
        for dialog1, dialog2 in dialog_pairs:
            label, init_score = text_agremment_model.predict_category(dialog1.message, dialog2.message)
            if label == 1:
                init_score = -init_score

            distance = abs(dialog1.index - dialog2.index)
            weight = math.exp(-DECAY_FACTOR * (distance - 1))
            agreement_score = init_score * weight
            max_possible_score = weight

            agreement_scores.append(agreement_score)
            max_possible_scores.append(max_possible_score)

        # Calculate final agreement score for the pair
        final_agreement_score = (
            sum(agreement_scores) / sum(max_possible_scores)
            if max_possible_scores
            else 0
        )
        relationships.append(
            SpeakerRelationship(
                speaker1=speaker1,
                speaker2=speaker2,
                agreement_score=final_agreement_score
            )
        )

    # Return a comprehensive relationships report
    return SpeakerRelationshipsReport(report=relationships)
