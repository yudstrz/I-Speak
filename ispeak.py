# === Install Library Utama (MODIFIED) ===
#pip install h5py openai-whisper gtts librosa transformers torch nltk scikit-learn streamlit sentence-transformers soundfile textblob --no-warn-script-location

#python -m streamlit run ispeak.py


# ============================================================
#                IMPORT LIBRARY (MODIFIED)
# ============================================================
import os, sys, re, tempfile, warnings, pickle, h5py
import numpy as np
import pandas as pd

# Streamlit
import streamlit as st

# Audio Processing (MODIFIED: Added soundfile, removed parselmouth)
import librosa
import soundfile as sf  # Replaces FFmpeg functionality
from gtts import gTTS
import whisper
from scipy.signal import find_peaks

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# NLTK (EXPANDED: Replaces spaCy functionality)
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet as wn
from nltk import pos_tag, ne_chunk, ngrams
from nltk.tree import Tree
from nltk.chunk import ne_chunk

# Sentence Transformers (SBERT)
from sentence_transformers import SentenceTransformer, util

# TextBlob (REPLACES LanguageTool)
from textblob import TextBlob

# Hilangkan warning
warnings.filterwarnings("ignore")

# ===================================
# STREAMLIT PAGE CONFIG
# ===================================
st.set_page_config(
    page_title="I-Speak Automated Speech Assessment",
    layout="wide"
)

st.title("I-Speak Automated Speech Assessment")
st.markdown("---")

# ===================================
# FEATURE MAPPING CONSTANTS
# ===================================

# CEFR level mapping
CEFR_MAPPING = {
    0: "A1",
    1: "A2", 
    2: "B1",
    3: "B2",
    4: "C1",
    5: "C2"
}

# Define the exact order of 39 numerical features for the model
NUMERICAL_FEATURES_ORDER = [
    "Durasi (s)", "MFCC (%)", "Semantic Coherence (%)", "Pause Freq",
    "Token Count", "Type Count", "TTR", "Pitch Range (Hz)",
    "Articulation Rate", "MLR", "Mean Pitch", "Stdev Pitch",
    "Mean Energy", "Stdev Energy", "Num Prominences", 
    "Prominence Dist Mean", "Prominence Dist Std", "WPM", "WPS",
    "Total Words", "Linking Count", "Discourse Count", "Filled Pauses",
    "Topic Similarity (%)", "Grammar Errors", "Idioms Found",
    "CEFR A1", "CEFR A2", "CEFR B1", "CEFR B2", "CEFR C1", "CEFR C2",
    "CEFR UNKNOWN", "Bigram Count", "Trigram Count", "Fourgram Count",
    "Synonym Variations", "Avg Tree Depth", "Max Tree Depth"
]

# Mapping features to subconstructs (using exact feature names from NUMERICAL_FEATURES_ORDER)
SUBCONSTRUCTS = {
    "Fluency": ["Total Words", "WPM", "WPS", "Filled Pauses", "MLR", "Pause Freq", "Durasi (s)"],
    "Pronunciation": ["Articulation Rate", "Pitch Range (Hz)", "MFCC (%)"],
    "Prosody": ["Mean Pitch", "Stdev Pitch", "Mean Energy", "Stdev Energy",
                "Num Prominences", "Prominence Dist Mean", "Prominence Dist Std"],
    "Coherence and Cohesion": ["Semantic Coherence (%)", "Discourse Count", "Linking Count"],
    "Topic Relevance": ["Topic Similarity (%)"],
    "Complexity": ["Idioms Found", "Bigram Count", "Trigram Count", "Fourgram Count",
                   "Synonym Variations", "CEFR A1", "CEFR A2", "CEFR B1", "CEFR B2",
                   "CEFR C1", "CEFR C2", "CEFR UNKNOWN", "Avg Tree Depth", "Max Tree Depth",
                   "Token Count", "Type Count", "TTR"],
    "Accuracy": ["Grammar Errors"]
}

# Model files mapping
MODEL_FILES = {
    "Fluency": "Fluency_rf_classification.h5",
    "Pronunciation": "Pronunciation_rf_classification.h5",
    "Prosody": "Prosody_rf_classification.h5",
    "Coherence and Cohesion": "Coherence_and_Cohesion_rf_classification.h5",
    "Topic Relevance": "Topic_Relevance_rf_classification.h5",
    "Complexity": "Complexity_rf_classification.h5",
    "Accuracy": "Accuracy_rf_classification.h5",
    "CEFR": "CEFR_rf_classification.h5"
}

# ===================================
# HELPER FUNCTIONS (AFTER STREAMLIT IMPORT)
# ===================================

def load_model_from_h5(filename, key_name=None):
    """Load model from H5 file"""
    try:
        with h5py.File(filename, "r") as f:
            if key_name is None:
                key_name = list(f.keys())[0]
            model_bytes = bytes(f[key_name][()])
            model = pickle.loads(model_bytes)
        return model
    except Exception as e:
        st.error(f"Error loading model {filename}: {e}")  # Now st is available
        return None

# ===================================
# CACHED FUNCTIONS (AFTER STREAMLIT IMPORT)
# ===================================
@st.cache_resource
def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        # Standard NLTK downloads
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        # Additional for parsing
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('brown', quiet=True)
        nltk.download('treebank', quiet=True)
        
        return True
    except Exception as e:
        st.warning(f"Some NLTK resources might not be available: {e}")
        return False

@st.cache_resource
def load_models_and_resources():
    """Load all required models and resources (MODIFIED)"""
    try:
        # Download NLTK resources first
        nltk_success = download_nltk_resources()
        if not nltk_success:
            st.warning("Some NLTK resources might not be available")
        
        # Load Whisper model
        whisper_model = whisper.load_model("tiny")
        st.success("✅ Whisper model loaded")
        
        # NLTK setup (REPLACES spaCy)
        try:
            # Test NLTK functionality
            test_tokens = word_tokenize("This is a test.")
            test_pos = pos_tag(test_tokens)
            st.success("✅ NLTK loaded (replacing spaCy)")
        except Exception as e:
            st.error(f"❌ NLTK setup failed: {e}")
        
        # Load SentenceTransformer
        try:
            sbert = SentenceTransformer('stsb-roberta-large')
            st.success("✅ SentenceTransformer loaded")
        except:
            try:
                sbert = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("✅ SentenceTransformer (backup) loaded")
            except:
                st.warning("⚠️ SentenceTransformer not available")
                sbert = None
        
        # TextBlob is already tested above - no additional setup needed
        st.success("✅ TextBlob loaded (replacing LanguageTool)")
        
        # Load stop words
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        return whisper_model, sbert, stop_words
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_reference_data():
    """Load reference datasets"""
    try:
        # Load CEFR dataset
        if os.path.exists('English_CEFR_Words.csv'):
            cefr_df = pd.read_csv('English_CEFR_Words.csv', delimiter=';', names=['Headword', 'CEFR'], skiprows=1)
            cefr_df['Headword'] = cefr_df['Headword'].str.lower()
            cefr_df = cefr_df.dropna()
            st.success("✅ CEFR dataset loaded")
        else:
            st.warning("⚠️ CEFR dataset not found. Please upload 'English_CEFR_Words.csv'")
            cefr_df = pd.DataFrame(columns=['Headword', 'CEFR'])
        
        # Load idioms dataset
        if os.path.exists('idioms_english.csv'):
            df_idioms = pd.read_csv('idioms_english.csv')
            idioms_english = df_idioms['idioms'].dropna().str.lower().tolist()
            st.success("✅ Idioms dataset loaded")
        else:
            st.warning("⚠️ Idioms dataset not found. Please upload 'idioms_english.csv'")
            idioms_english = []
        
        return cefr_df, idioms_english
        
    except Exception as e:
        st.error(f"Error loading reference data: {e}")
        return pd.DataFrame(), []

@st.cache_resource
def load_prediction_models():
    """Load all prediction models"""
    loaded_models = {}
    model_status = []
    
    for name, file in MODEL_FILES.items():
        if os.path.exists(file):
            key_name = "rf_cefr" if name == "CEFR" else "rf_clf"
            model = load_model_from_h5(file, key_name)
            if model is not None:
                loaded_models[name] = model
                model_status.append(f"✅ {name}")
            else:
                model_status.append(f"❌ {name} (failed to load)")
        else:
            model_status.append(f"⚠️ {name} (file not found)")
    
    return loaded_models, model_status


# ===================================
# LEXICAL BUNDLES DEFINITIONS
# ===================================
valid_bigrams = {
    "for example", "in fact", "of course", "such as", "in particular",
    "as well", "due to", "in general", "this means", "this suggests",
    "in conclusion", "as shown", "in short", "in turn", "on average",
    "as expected", "more importantly", "in summary", "at least", "most likely",
    "less than", "more than", "according to", "as noted", "for instance",
    "so that", "such that", "even though", "as a", "on top", "as mentioned",
    "from which", "in contrast", "in addition", "in response", "as discussed",
    "by contrast", "to ensure", "with regard", "with respect", "as stated",
    "in brief", "on purpose", "in effect", "in excess", "in theory",
    "at best", "at worst", "as shown", "on average", "it seems", "it appears",
    "for this", "in spite", "in line", "by using", "on behalf", "in turn",
    "in favor", "by means", "at times", "among others", "to conclude",
    "for instance", "on occasion", "it means", "for comparison", "with this",
    "in context", "with regard", "over time", "in reference", "in depth",
    "in support", "to illustrate", "to emphasize", "for emphasis", "under consideration",
    "above all", "as follows", "in summary", "more precisely", "more clearly",
    "in reality", "as previously", "in brief", "at present", "in practice",
    "in theory", "in contrast", "by contrast", "by definition", "without doubt",
    "beyond that", "more generally", "from there", "with caution", "as required",
    "in hindsight", "at large"
}

valid_trigrams = {
    "as a result", "on the other", "in terms of", "as well as",
    "one of the", "in order to", "the end of", "the fact that",
    "on the basis", "at the same", "at the end", "in the case",
    "the rest of", "in addition to", "the purpose of", "the use of",
    "the development of", "with respect to", "as a consequence",
    "in the process", "as part of", "due to the", "the nature of",
    "it is important", "it is necessary", "it should be", "the number of",
    "there is a", "there are a", "from the point", "in the context",
    "in the light", "on the part", "at the beginning", "it is possible",
    "it is clear", "it is evident", "according to the", "with regard to",
    "the result of", "the role of", "as a result of", "in contrast to",
    "this means that", "this suggests that", "this indicates that",
    "for the purpose", "in comparison to", "in relation to", "with the aim",
    "it can be", "this is because", "there seems to", "it is likely",
    "the majority of", "in the following", "a wide range", "it should also",
    "in the form", "for the sake", "on the whole", "it may be",
    "this is due", "it is argued", "it has been", "in some cases",
    "in such a", "one could argue", "as shown in", "it is worth",
    "in accordance with", "a number of", "in spite of", "in favour of",
    "in the event", "the focus of", "the aim of", "to the extent",
    "in support of", "in line with", "to be able", "the presence of",
    "in general terms", "as can be", "to some extent", "based on the",
    "to illustrate this", "the significance of", "from the perspective",
    "the findings of", "in academic writing", "research has shown",
    "recent studies have", "according to recent", "on the one hand",
    "on the other hand", "it is evident that", "it is assumed that",
    "this highlights the", "this demonstrates the", "as illustrated in",
    "in this paper"
}

valid_fourgrams = {
    "as a result of", "at the end of", "in the case of", "as can be seen",
    "in the context of", "on the basis of", "at the same time",
    "in terms of the", "in the process of", "with the help of",
    "as a part of", "as shown in figure", "it is important to",
    "in relation to the", "this is due to", "the role of the",
    "as illustrated in figure", "in this study we", "the results of the",
    "it is necessary to", "there is a need", "at the beginning of",
    "one of the most", "from the point of", "with respect to the",
    "in the case where", "in line with the", "in order to ensure",
    "the fact that the", "with the aim of", "to a large extent",
    "in spite of the", "from the perspective of", "in accordance with the",
    "for the purpose of", "in the same way", "it should be noted",
    "this can be seen", "this is evident in", "the purpose of this",
    "to some extent it", "in a number of", "this is not to",
    "is one of the", "it is possible to", "this highlights the importance",
    "to the extent that", "this is because of", "in this section we",
    "a wide range of", "the nature of the", "the main objective of",
    "in this chapter we", "it is also important", "the extent to which",
    "the implications of this", "with regard to the", "in the following section",
    "can be used to", "at the heart of", "it has been shown",
    "there appears to be", "this may be due", "one of the key",
    "the relationship between the", "it is worth noting", "there is evidence that",
    "in the light of", "the significance of the", "as a consequence of",
    "can be seen as", "the basis of the", "the findings of this",
    "as will be discussed", "there is no doubt", "the aim of this",
    "in a similar way", "at the time of", "can be interpreted as",
    "can be described as", "it should also be", "the data suggest that",
    "on the one hand", "on the other hand", "in the form of",
    "as shown in table", "in contrast to the", "in relation to their",
    "may be seen as", "on the grounds that", "this is supported by",
    "this raises the question", "should be taken into", "as described in the",
    "the results indicate that", "as mentioned in the", "to be taken into",
    "in this particular case", "this supports the idea", "the reason for this",
    "at the same level"
}

# ===================================
# ANALYSIS FUNCTIONS (MODIFIED)
# ===================================

def preprocess_text(text):
    """Clean and preprocess text"""
    return re.sub(r'[^\w\s]', '', text.lower())

def tokenize_text(text):
    """Tokenize text and keep only alphabetic tokens"""
    try:
        return [token for token in word_tokenize(text) if token.isalpha()]
    except LookupError:
        # Fallback if NLTK data is not available
        return [token for token in text.split() if token.isalpha()]

def get_sentence_embedding(sentence):
    """Get sentence embedding using SentenceTransformer"""
    if sbert is not None:
        return sbert.encode(sentence)
    return np.zeros(384)  # fallback

def compute_semantic_coherence(text):
    """Compute semantic coherence between sentences"""
    try:
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback sentence splitting
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 100.0
        
        sims = []
        for i in range(len(sentences) - 1):
            emb1 = get_sentence_embedding(sentences[i])
            emb2 = get_sentence_embedding(sentences[i + 1])
            sim = cosine_similarity([emb1], [emb2])[0][0] * 100
            sims.append(sim)
        
        return np.mean(sims)
    except:
        return 0.0

def count_long_pauses_from_audio(audio_path, pause_threshold=0.5):
    """Count long pauses in audio (MODIFIED: Uses librosa only)"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        intervals = librosa.effects.split(y, top_db=30)
        if len(intervals) < 2:
            return 0
        pauses = []
        for i in range(1, len(intervals)):
            pause_duration = (intervals[i][0]/sr) - (intervals[i-1][1]/sr)
            pauses.append(pause_duration)
        return int(np.sum(np.array(pauses) > pause_threshold))
    except:
        return 0

def calculate_ttr(text):
    """Calculate Type-Token Ratio"""
    tokens = re.findall(r'\b\w+\b', text.lower())
    token_count = len(tokens)
    type_count = len(set(tokens))
    ttr = type_count / token_count if token_count > 0 else 0

