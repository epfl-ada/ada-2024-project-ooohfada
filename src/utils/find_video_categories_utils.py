import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import ldamodel
from tqdm import tqdm
from gensim import corpora
import string
from gensim.models import CoherenceModel
from src.utils.recovery_analysis_utils import str_to_list

CASEFOLD = False
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_str(s):
    if not isinstance(s, str) or not s.strip():  # Cases where s = None
        return []
    s = s.lower()
    tokens = word_tokenize(s.lower() if CASEFOLD else s, preserve_line=True)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens

def load_data():
    decline_events = pd.read_csv('data/sampled_decline_events_with_videos.csv')
    videos = pd.read_csv('data/videos_around_declines.csv')
    return decline_events, videos

def process_data(decline_events):
    decline_events['Videos_before'] = decline_events['Videos_before'].apply(str_to_list)
    decline_events['Videos_after'] = decline_events['Videos_after'].apply(str_to_list)
    return decline_events

def create_tags_dataframe(decline_events, videos):
    # Create a data_frame with 2 index: the index of the decline and the source (before and after)
    df_before = decline_events[['Videos_before']].explode('Videos_before')
    df_before['Source'] = 'Before'
    df_before = df_before.rename(columns={'Videos_before': 'Video'})

    df_after = decline_events[['Videos_after']].explode('Videos_after')
    df_after['Source'] = 'After'
    df_after = df_after.rename(columns={'Videos_after': 'Video'})

    df_tags = pd.concat([df_before, df_after], axis=0).reset_index().rename(columns={'index': 'Decline'})
    df_tags = df_tags.set_index(['Decline', 'Source'])
    df_tags.sort_values(by=['Decline', 'Source'])
    df_tags = df_tags.dropna()

    # Map to obtain the tags of all videos for each video before and after decline
    df_tags['Tags'] = df_tags['Video'].map(lambda video: videos.loc[video, 'tags'] if video in videos.index else None)

    # Get for each decline only 2 rows with the tags corresponding to the before and the after, handling NaNs and non-list values
    df_tags = df_tags.groupby(['Decline', 'Source'])['Tags'].apply(
        lambda x: list(set([item for sublist in x.dropna() for item in (sublist if isinstance(sublist, list) else [sublist])]))
    ).reset_index(name='Tags_combined')

    df_tags.set_index(['Decline', 'Source'], inplace=True)
    
    # Map the tags to a string, separating them by new lines
    df_tags['Tags_combined'] = df_tags['Tags_combined'].map(lambda tags: '\\n'.join(tags) if tags else None)
    
    return df_tags

def create_dictionary_and_corpus(df_tags):
    dictionary = corpora.Dictionary(df_tags['Tokens'])
    corpus = [dictionary.doc2bow(token_list) for token_list in df_tags['Tokens']]
    return dictionary, corpus

def assign_dominant_topic(tokens, lda_model, dictionary):
    if not tokens or not isinstance(tokens, list):
        return None, None
    bow = dictionary.doc2bow(tokens)
    topic_probs = lda_model.get_document_topics(bow)
    if topic_probs:
        dominant_topic, prob = max(topic_probs, key=lambda x: x[1])
        return dominant_topic, prob
    return None, None

def create_pivot_table(df_small):
    df_pivot = df_small.pivot_table(
        index='Decline',
        columns='Source',
        values=['Tokens', 'Dominant_Topic'],
        aggfunc={
            'Tokens': lambda x: ' '.join([item for sublist in x for item in sublist]),
            'Dominant_Topic': lambda x: x.mode()[0]
        }
    )
    return df_pivot

def token_change(tokens_before, tokens_after):
    if not isinstance(tokens_before, list):
        tokens_before = []
    if not isinstance(tokens_after, list):
        tokens_after = []
        
    set_before = set(tokens_before)
    set_after = set(tokens_after)
    return set_before != set_after

def detect_changes(df_pivot):
    df_pivot['Token_Change'] = df_pivot.apply(
        lambda row: token_change(row[('Tokens', 'Before')], row[('Tokens', 'After')]), axis=1
    )

    df_pivot['Topic_Change'] = df_pivot.apply(
        lambda row: row[('Dominant_Topic', 'Before')] != row[('Dominant_Topic', 'After')], axis=1
    )
    return df_pivot

def calculate_coherence(lda, df_small, dictionary):
    coherence_model_lda = CoherenceModel(model=lda, texts=df_small['Tokens'], dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    return coherence_lda