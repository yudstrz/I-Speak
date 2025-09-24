# === Streamlit Cloud Compatible Version ===
# ============================================================
#                IMPORT LIBRARY
# ============================================================
import os, sys, re, tempfile, warnings, pickle, h5py
import numpy as np
import pandas as pd

# Streamlit
import streamlit as st

# Audio Processing
import librosa, soundfile as sf
from gtts import gTTS
import whisper

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# NLTK
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet as wn
from nltk import pos_tag, ne_chunk, ngrams
from nltk.tree import Tree

# Sentence Transformers (SBERT)
from sentence_transformers import SentenceTransformer, util

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
# INITIALIZATION & CACHING - DENGAN FALLBACKS
# ===================================
@st.cache_resource
def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        # Updated NLTK downloads for newer versions
        resources = [
            'punkt', 'punkt_tab', 'averaged_perceptron_tagger', 
            'averaged_perceptron_tagger_eng', 'words', 'maxent_ne_chunker',
            'wordnet', 'stopwords', 'omw-1.4'
        ]
        
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except:
                st.warning(f"Could not download {resource}")
                
        return True
    except Exception as e:
        st.warning(f"Some NLTK resources might not be available: {e}")
        return False

@st.cache_resource
def load_models_and_resources():
    """Load all required models and resources - WITH FALLBACKS"""
    try:
        # Download NLTK resources first
        nltk_success = download_nltk_resources()
        if not nltk_success:
            st.warning("Some NLTK resources might not be available")
        
        # Load Whisper model
        try:
            whisper_model = whisper.load_model("tiny")
            st.success("✅ Whisper model loaded")
        except Exception as e:
            st.error(f"❌ Could not load Whisper model: {e}")
            whisper_model = None
        
        # === SPACY MODEL - OPTION A: AUTO-DOWNLOAD, OPTION B: NLTK FALLBACK ===
        nlp = None
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
                st.success("✅ spaCy model loaded")
            except OSError:
                st.info("📦 Downloading spaCy model...")
                import subprocess
                try:
                    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                    nlp = spacy.load("en_core_web_sm")
                    st.success("✅ spaCy model downloaded and loaded")
                except Exception as e:
                    st.warning(f"⚠️ Could not download spaCy model: {e}")
                    st.info("🔄 Using NLTK fallback for NLP tasks")
                    nlp = None
        except ImportError:
            st.warning("⚠️ spaCy not available, using NLTK fallback")
            nlp = None
        
        # Try to add benepar if available
        if nlp is not None:
            try:
                import benepar
                if "benepar" not in nlp.pipe_names:
                    benepar.download('benepar_en3')
                    nlp.add_pipe("benepar", config={"model": "benepar_en3"}, last=True)
                    st.success("✅ spaCy + Benepar loaded")
            except:
                st.info("ℹ️ Benepar not available, using spaCy only")
        
        # Load SentenceTransformer
        sbert = None
        try:
            sbert = SentenceTransformer('stsb-roberta-large')
            st.success("✅ SentenceTransformer loaded")
        except:
            try:
                sbert = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("✅ SentenceTransformer (backup) loaded")
            except Exception as e:
                st.warning(f"⚠️ SentenceTransformer not available: {e}")
                sbert = None
        
        # === LANGUAGE TOOL - OPTION A: LANGUAGE-TOOL-PYTHON, OPTION B: TEXTBLOB ===
        tool = None
        grammar_method = "none"
        
        # Option A: Try language-tool-python
        try:
            import language_tool_python
            tool = language_tool_python.LanguageTool('en-US')
            grammar_method = "languagetool"
            st.success("✅ LanguageTool loaded")
        except ImportError:
            st.info("📦 LanguageTool not installed, trying TextBlob...")
        except Exception as e:
            st.warning(f"⚠️ LanguageTool error: {str(e)}")
            st.info("💡 This may be due to Java not being installed")
        
        # Option B: Fallback to TextBlob
        if tool is None:
            try:
                from textblob import TextBlob
                tool = "textblob"  # Store string indicator
                grammar_method = "textblob"
                st.success("✅ TextBlob loaded as grammar fallback")
            except ImportError:
                st.warning("⚠️ Neither LanguageTool nor TextBlob available")
                tool = None
                grammar_method = "none"
        
        # === PROSODY ANALYSIS - OPTION A: PARSELMOUTH, OPTION B: LIBROSA ONLY ===
        prosody_method = "none"
        try:
            import parselmouth
            prosody_method = "parselmouth"
            st.success("✅ Parselmouth loaded for prosody analysis")
        except ImportError:
            st.warning("⚠️ Parselmouth not available, using librosa-only prosody")
            prosody_method = "librosa"
        
        # Load stop words
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        return whisper_model, nlp, sbert, tool, stop_words, grammar_method, prosody_method
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, "none", "none"

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

# ===================================
# LEXICAL BUNDLES DEFINITIONS (shortened for space)
# ===================================
valid_bigrams = {
    "for example", "in fact", "of course", "such as", "in particular",
    "as well", "due to", "in general", "this means", "this suggests",
    "in conclusion", "as shown", "in short", "in turn", "on average"
    # ... add more as needed
}

valid_trigrams = {
    "as a result", "on the other", "in terms of", "as well as",
    "one of the", "in order to", "the end of", "the fact that"
    # ... add more as needed
}

valid_fourgrams = {
    "as a result of", "at the end of", "in the case of", "as can be seen"
    # ... add more as needed
}

# ===================================
# ANALYSIS FUNCTIONS - WITH FALLBACKS
# ===================================

def preprocess_text(text):
    """Clean and preprocess text"""
    return re.sub(r'[^\w\s]', '', text.lower())

def tokenize_text(text):
    """Tokenize text and keep only alphabetic tokens - WITH FALLBACK"""
    try:
        return [token for token in word_tokenize(text) if token.isalpha()]
    except LookupError:
        # Fallback if NLTK data is not available
        return [token for token in text.split() if token.isalpha()]

def get_sentence_embedding(sentence, sbert):
    """Get sentence embedding using SentenceTransformer"""
    if sbert is not None:
        try:
            return sbert.encode(sentence)
        except:
            return np.zeros(384)  # fallback
    return np.zeros(384)  # fallback

def compute_semantic_coherence(text, sbert):
    """Compute semantic coherence between sentences - WITH FALLBACK"""
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
            emb1 = get_sentence_embedding(sentences[i], sbert)
            emb2 = get_sentence_embedding(sentences[i + 1], sbert)
            if np.all(emb1 == 0) or np.all(emb2 == 0):
                sim = 50.0  # fallback similarity
            else:
                sim = cosine_similarity([emb1], [emb2])[0][0] * 100
            sims.append(sim)
        
        return np.mean(sims)
    except:
        return 50.0  # fallback value

def count_long_pauses_from_audio(audio_path, pause_threshold=0.5):
    """Count long pauses in audio - WITH FALLBACK"""
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
    except Exception as e:
        st.warning(f"Could not analyze pauses: {e}")
        return 0

def calculate_ttr(text):
    """Calculate Type-Token Ratio"""
    tokens = re.findall(r'\b\w+\b', text.lower())
    token_count = len(tokens)
    type_count = len(set(tokens))
    ttr = type_count / token_count if token_count > 0 else 0.0
    return token_count, type_count, ttr

def calculate_bert_topic_similarity(original_text, reference_text, sbert):
    """Calculate topic similarity using BERT embeddings - WITH FALLBACK"""
    try:
        if sbert is not None:
            embeddings = sbert.encode([original_text, reference_text])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100
            return similarity
        else:
            # Simple word overlap fallback
            words1 = set(original_text.lower().split())
            words2 = set(reference_text.lower().split())
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            return (overlap / total * 100) if total > 0 else 0.0
    except:
        return 50.0  # fallback

def get_pitch_range(audio_path, prosody_method):
    """Get pitch range from audio - WITH FALLBACKS"""
    try:
        if prosody_method == "parselmouth":
            import parselmouth
            sound = parselmouth.Sound(audio_path)
            pitch = sound.to_pitch()
            pitch_values = [p for p in pitch.selected_array['frequency'] if p > 50]
            return np.ptp(pitch_values) if pitch_values else 0.0
        else:
            # Librosa fallback
            y, sr = librosa.load(audio_path, sr=16000)
            # Simple pitch estimation using zero crossing rate as proxy
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            return np.std(zcr) * 1000  # Scale to reasonable Hz range
    except Exception as e:
        st.warning(f"Pitch range calculation error: {e}")
        return 0.0

def get_articulation_rate(result):
    """Get articulation rate from Whisper result"""
    try:
        segments = result.get("segments", [])
        if not segments:
            return 0
        total_words = sum(len(seg["text"].split()) for seg in segments)
        duration = segments[-1]["end"] - segments[0]["start"]
        return total_words / duration if duration > 0 else 0
    except:
        return 0

def calculate_mlr(result, pause_threshold=0.5):
    """Calculate Mean Length of Run"""
    try:
        segments = result.get("segments", [])
        runs, current_run_word_count, prev_end = [], 0, None
        
        for seg in segments:
            start, end = seg["start"], seg["end"]
            word_count = len(seg["text"].split())
            
            if prev_end is None or start - prev_end <= pause_threshold:
                current_run_word_count += word_count
            else:
                runs.append(current_run_word_count)
                current_run_word_count = word_count
            prev_end = end
        
        if current_run_word_count > 0:
            runs.append(current_run_word_count)
        
        return np.mean(runs) if runs else 0
    except:
        return 0

def analyze_prosody(audio_path, prosody_method):
    """Analyze prosodic features - WITH FALLBACKS"""
    try:
        if prosody_method == "parselmouth":
            import parselmouth
            from parselmouth.praat import call
            from scipy.signal import find_peaks
            
            y, sr = librosa.load(audio_path, sr=16000)
            sound = parselmouth.Sound(audio_path)
            pitch = sound.to_pitch()
            energy = librosa.feature.rms(y=y)[0]
            intensity = sound.to_intensity()
            
            intensity_values = intensity.values[0]
            intensity_times = intensity.xs()
            
            # Prominence detection
            threshold = np.mean(intensity_values) + np.std(intensity_values)
            peaks, _ = find_peaks(intensity_values, height=threshold)
            prominences = [intensity_times[i] for i in peaks]
            
            return {
                "mean_pitch": call(pitch, "Get mean", 0, 0, "Hertz"),
                "stdev_pitch": call(pitch, "Get standard deviation", 0, 0, "Hertz"),
                "mean_energy": np.mean(energy),
                "stdev_energy": np.std(energy),
                "num_prominences": len(prominences),
                "mean_distance_between_prominence": np.mean(np.diff(prominences)) if len(prominences) > 1 else 0,
                "std_distance_between_prominence": np.std(np.diff(prominences)) if len(prominences) > 1 else 0
            }
        else:
            # Librosa-only fallback
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Basic energy features
            energy = librosa.feature.rms(y=y)[0]
            
            # Simple pitch estimation using spectral centroid as proxy
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # Simple prominence detection using energy peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(energy, height=np.mean(energy) + np.std(energy))
            
            return {
                "mean_pitch": np.mean(spectral_centroids),
                "stdev_pitch": np.std(spectral_centroids),
                "mean_energy": np.mean(energy),
                "stdev_energy": np.std(energy),
                "num_prominences": len(peaks),
                "mean_distance_between_prominence": np.mean(np.diff(peaks)) if len(peaks) > 1 else 0,
                "std_distance_between_prominence": np.std(np.diff(peaks)) if len(peaks) > 1 else 0
            }
    except Exception as e:
        st.warning(f"Prosody analysis error: {e}")
        return {
            "mean_pitch": 0, "stdev_pitch": 0, "mean_energy": 0, "stdev_energy": 0,
            "num_prominences": 0, "mean_distance_between_prominence": 0, "std_distance_between_prominence": 0
        }

def grammar_accuracy(text, tool, grammar_method):
    """Calculate grammar accuracy - WITH FALLBACKS"""
    try:
        if grammar_method == "languagetool" and tool is not None:
            matches = tool.check(text)
            num_errors = len(matches)
        elif grammar_method == "textblob":
            from textblob import TextBlob
            blob = TextBlob(text)
            corrected = str(blob.correct())
            # Simple error count based on differences
            original_words = text.split()
            corrected_words = corrected.split()
            num_errors = sum(1 for a, b in zip(original_words, corrected_words) if a != b)
        else:
            # No grammar checker available
            return 100, 0, 0
        
        tokens = tokenize_text(text)
        total_words = len(tokens)
        if total_words == 0:
            return 100, num_errors, total_words
        
        correct_words = max(0, total_words - num_errors)
        accuracy_score = (correct_words / total_words) * 100
        return accuracy_score, num_errors, total_words
        
    except Exception as e:
        st.warning(f"Grammar analysis error: {e}")
        return 100, 0, 0

def get_avg_and_max_tree_depth(text, nlp):
    """Get average and maximum syntactic tree depth - WITH FALLBACK"""
    try:
        if nlp is not None:
            doc = nlp(text)
            all_depths = []
            
            for sent in doc.sents:
                try:
                    if hasattr(sent._, 'parse_string'):
                        parse_tree = Tree.fromstring(sent._.parse_string)
                        depth = parse_tree.height() - 1
                        all_depths.append(depth)
                except:
                    continue
            
            if not all_depths:
                # Fallback to simple depth estimation
                sentences = text.split('.')
                avg_depth = len(sentences) * 2  # Simple heuristic
                max_depth = avg_depth
                return avg_depth, max_depth
            
            avg_depth = sum(all_depths) / len(all_depths)
            max_depth = max(all_depths)
            return avg_depth, max_depth
        else:
            # NLTK/simple fallback
            sentences = text.split('.')
            avg_depth = len(sentences) * 2  # Simple heuristic
            max_depth = avg_depth
            return avg_depth, max_depth
    except:
        return 3.0, 5.0  # reasonable defaults

# ===================================
# OTHER ANALYSIS FUNCTIONS (SIMPLIFIED FOR SPACE)
# ===================================

def linking_discourse_filled_counts(text):
    """Count linking words, discourse markers, and filled pauses"""
    # Simplified implementation
    linking_words = {"and", "but", "or", "so", "because", "therefore", "however", "moreover"}
    discourse_markers = {"you know", "i mean", "like", "well", "actually", "basically"}
    filled_pauses = {"um", "uh", "er", "ah", "hmm", "mm"}
    
    text_lower = text.lower()
    tokens = text_lower.split()
    joined = ' '.join(tokens)
    
    linking = [w for w in tokens if w in linking_words]
    discourse = [m for m in discourse_markers if m in joined]
    filled = [f for f in filled_pauses if f in tokens]
    
    return {
        "linking_count": len(linking),
        "linking_words_found": linking,
        "discourse_count": len(discourse),
        "discourse_markers_found": discourse,
        "filled_count": len(filled),
        "filled_pauses_found": filled
    }

def speech_rate(text, duration_seconds):
    """Calculate speech rate metrics"""
    words = tokenize_text(text)
    num_words = len(words)
    if duration_seconds == 0:
        return 0, 0, num_words
    wps = num_words / duration_seconds
    wpm = (num_words / duration_seconds) * 60
    return wpm, wps, num_words

def find_idioms_in_text(text, idioms_list):
    """Find idioms in text"""
    text_lower = text.lower()
    found_idioms = []
    for idiom in idioms_list:
        pattern = r'\b' + re.escape(idiom) + r'\b'
        if re.search(pattern, text_lower):
            found_idioms.append(idiom)
    return found_idioms

def map_words_to_cefr(text, cefr_df):
    """Map words to CEFR levels"""
    tokens = [token.lower() for token in tokenize_text(text)]
    word_levels = {}
    if len(cefr_df) > 0:
        cefr_dict = dict(zip(cefr_df['Headword'], cefr_df['CEFR']))
        for word in tokens:
            word_levels[word] = cefr_dict.get(word, 'UNKNOWN')
    else:
        for word in tokens:
            word_levels[word] = 'UNKNOWN'
    return word_levels

def count_cefr_distribution(word_levels):
    """Count CEFR level distribution"""
    levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'UNKNOWN']
    dist = Counter(word_levels.values())
    return {level: dist.get(level, 0) for level in levels}

def count_all_lexical_bundles(text):
    """Count lexical bundles (simplified)"""
    try:
        tokens = tokenize_text(text)
        bigram_count = sum(1 for gram in ngrams(tokens, 2) if ' '.join(gram) in valid_bigrams)
        trigram_count = sum(1 for gram in ngrams(tokens, 3) if ' '.join(gram) in valid_trigrams)
        fourgram_count = sum(1 for gram in ngrams(tokens, 4) if ' '.join(gram) in valid_fourgrams)
        
        return {
            "bigram_count": bigram_count,
            "trigram_count": trigram_count,
            "fourgram_count": fourgram_count
        }
    except:
        return {"bigram_count": 0, "trigram_count": 0, "fourgram_count": 0}

def count_synonym_variations(text):
    """Count synonym variations - simplified"""
    try:
        tokens = tokenize_text(text)
        # Simplified - count unique words as proxy
        return len(set(tokens)) // 10, []  # rough estimate
    except:
        return 0, []

def process_audio_file(audio_path, whisper_model):
    """Main audio processing function - WITH ERROR HANDLING"""
    try:
        # Transcription
        result = whisper_model.transcribe(
            audio_path,
            language="en",
            condition_on_previous_text=False,
            temperature=0.0,
            fp16=False
        )
        original_text = result["text"]
        
        # Load original audio
        y1, sr = librosa.load(audio_path, sr=16000)
        dur = librosa.get_duration(y=y1, sr=sr)
        
        # Generate TTS for MFCC comparison
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tts_file:
                tts_path = tts_file.name
            
            gTTS(original_text).save(tts_path)
            y2, _ = librosa.load(tts_path, sr=16000)
            sf.write(tts_path, y2, 16000)
            
            # MFCC comparison
            mfcc1 = StandardScaler().fit_transform(librosa.feature.mfcc(y=y1, sr=16000, n_mfcc=13))
            mfcc2 = StandardScaler().fit_transform(librosa.feature.mfcc(y=y2, sr=16000, n_mfcc=13))
            max_len = max(mfcc1.shape[1], mfcc2.shape[1])
            pad = lambda m: np.pad(m, ((0, 0), (0, max_len - m.shape[1])), mode='constant')
            cosine_sim = cosine_similarity(
                pad(mfcc1).flatten().reshape(1, -1),
                pad(mfcc2).flatten().reshape(1, -1)
            )[0][0] * 100
            
            # Clean up TTS file
            os.unlink(tts_path)
        except Exception as e:
            st.warning(f"TTS comparison failed: {e}")
            cosine_sim = 50.0  # fallback value
        
        return result, original_text, dur, cosine_sim
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, "", 0, 0

# ===================================
# MODEL LOADING FUNCTIONS
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
        st.error(f"Error loading model {filename}: {e}")
        return None

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

def extract_features_from_data(data):
    """Extract the 39 numerical features in correct order"""
    features = []
    for feature_name in NUMERICAL_FEATURES_ORDER:
        if feature_name in data:
            features.append(float(data[feature_name]))
        else:
            st.warning(f"Missing feature: {feature_name}, using default value 0")
            features.append(0.0)
    
    return np.array(features)

def predict_subconstruct(data, model, feature_names):
    """Predict a single subconstruct score"""
    try:
        X = np.array([[data[feat] for feat in feature_names]])
        prediction = model.predict(X)
        return int(prediction[0])
    except Exception as e:
        st.error(f"Error in subconstruct prediction: {e}")
        return 0

def predict_all_subconstructs_and_cefr(data, loaded_models):
    """Predict all subconstructs and CEFR level"""
    predictions = {}
    
    try:
        # Step 1: Predict each subconstruct (7 scores)
        subconstruct_scores = []
        for sub_name, feature_names in SUBCONSTRUCTS.items():
            if sub_name in loaded_models:
                try:
                    score = predict_subconstruct(data, loaded_models[sub_name], feature_names)
                    predictions[sub_name] = score
                    subconstruct_scores.append(score)
                except Exception as e:
                    st.error(f"Error predicting {sub_name}: {e}")
                    predictions[sub_name] = 0
                    subconstruct_scores.append(0)
            else:
                predictions[sub_name] = 0
                subconstruct_scores.append(0)
        
        # Step 2: Extract 39 numerical features
        numerical_features = extract_features_from_data(data)
        
        # Step 3: Combine into 46-feature array
        final_features = np.concatenate([numerical_features, subconstruct_scores])
        
        # Step 4: Predict CEFR using the 46-feature array
        if "CEFR" in loaded_models:
            try:
                X_cefr = final_features.reshape(1, -1)
                model_cefr = loaded_models["CEFR"]
                y_cefr = model_cefr.predict(X_cefr)
                predictions["CEFR"] = int(y_cefr[0])
            except Exception as e:
                st.error(f"Error predicting CEFR: {e}")
                predictions["CEFR"] = 0
        else:
            predictions["CEFR"] = 0
        
        return predictions, final_features
        
    except Exception as e:
        st.error(f"Error in prediction pipeline: {e}")
        default_predictions = {name: 0 for name in list(SUBCONSTRUCTS.keys()) + ["CEFR"]}
        default_features = np.zeros(46)
        return default_predictions, default_features

def build_comprehensive_feature_dict(uploaded_file, original_text, duration, mfcc_sim, 
                                    coherence, pause_freq, token_count, type_count, ttr,
                                    pitch_range, articulation, mlr, prosody, wpm, wps, 
                                    total_words, flu, sim, err_count, found_idioms,
                                    cefr_counts, bundles, syn_count, avg_depth, max_depth):
    """Build comprehensive feature dictionary with all 39 numerical features"""
    data = {
        # Basic info (non-numerical)
        "Filename": uploaded_file.name,
        "Transkrip": original_text,
        
        # 39 Numerical features (in exact order for model)
        "Durasi (s)": float(duration),
        "MFCC (%)": float(mfcc_sim),
        "Semantic Coherence (%)": float(coherence),
        "Pause Freq": int(pause_freq),
        "Token Count": int(token_count),
        "Type Count": int(type_count),
        "TTR": float(ttr),
        "Pitch Range (Hz)": float(pitch_range),
        "Articulation Rate": float(articulation),
        "MLR": float(mlr),
        "Mean Pitch": float(prosody['mean_pitch']),
        "Stdev Pitch": float(prosody['stdev_pitch']),
        "Mean Energy": float(prosody['mean_energy']),
        "Stdev Energy": float(prosody['stdev_energy']),
        "Num Prominences": int(prosody['num_prominences']),
        "Prominence Dist Mean": float(prosody['mean_distance_between_prominence']),
        "Prominence Dist Std": float(prosody['std_distance_between_prominence']),
        "WPM": float(wpm),
        "WPS": float(wps),
        "Total Words": int(total_words),
        "Linking Count": int(flu['linking_count']),
        "Discourse Count": int(flu['discourse_count']),
        "Filled Pauses": int(flu['filled_count']),
        "Topic Similarity (%)": float(sim),
        "Grammar Errors": int(err_count),
        "Idioms Found": int(len(found_idioms)),
        "CEFR A1": int(cefr_counts.get("A1", 0)),
        "CEFR A2": int(cefr_counts.get("A2", 0)),
        "CEFR B1": int(cefr_counts.get("B1", 0)),
        "CEFR B2": int(cefr_counts.get("B2", 0)),
        "CEFR C1": int(cefr_counts.get("C1", 0)),
        "CEFR C2": int(cefr_counts.get("C2", 0)),
        "CEFR UNKNOWN": int(cefr_counts.get("UNKNOWN", 0)),
        "Bigram Count": int(bundles['bigram_count']),
        "Trigram Count": int(bundles['trigram_count']),
        "Fourgram Count": int(bundles['fourgram_count']),
        "Synonym Variations": int(syn_count),
        "Avg Tree Depth": float(avg_depth),
        "Max Tree Depth": float(max_depth),
    }
    
    return data

# Load models and data
st.info("🔄 Loading models and resources...")
whisper_model, nlp, sbert, tool, stop_words, grammar_method, prosody_method = load_models_and_resources()
cefr_df, idioms_english = load_reference_data()

# ===================================
# STREAMLIT UI
# ===================================

def main():
    """Main Streamlit application"""
    
    # Check if models are loaded successfully
    if whisper_model is None:
        st.error("Cannot proceed without Whisper model. Please check your installation.")
        st.stop()
    
    # Display method status
    st.sidebar.subheader("🔧 Analysis Methods")
    st.sidebar.info(f"Grammar: {grammar_method}")
    st.sidebar.info(f"Prosody: {prosody_method}")
    st.sidebar.info(f"SBERT: {'Available' if sbert else 'Fallback'}")
    st.sidebar.info(f"spaCy: {'Available' if nlp else 'NLTK fallback'}")
    
    # Display speech prompt in main area
    st.header("Speech Prompt")
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #1f77b4;">📝 Free Speech Topic</h3>
        <p style="font-size: 16px; margin-bottom: 0; color: #333;">
            <strong>"Talk about the university course you enjoyed the most, describe one course you found difficult, and explain whether you think universities should focus more on practical skills or theoretical knowledge."</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Reference topic input in sidebar
    ref_topic = st.sidebar.text_area(
        "Reference Topic for Similarity Analysis:",
        value="Talk about university courses, discussing both enjoyable and difficult subjects, and comparing practical skills versus theoretical knowledge in higher education.",
        height=100,
        help="This topic will be used to calculate semantic similarity with the speech content"
    )
    
    # Load prediction models
    st.sidebar.subheader("📊 Model Status")
    loaded_models, model_status = load_prediction_models()
    
    for status in model_status:
        if "✅" in status:
            st.sidebar.success(status)
        elif "❌" in status:
            st.sidebar.error(status)
        else:
            st.sidebar.warning(status)
    
    if not loaded_models:
        st.warning("No prediction models loaded. You can still perform feature extraction, but predictions won't be available.")
    
    # Main content
    st.header("Audio Upload & Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac'],
        help="Upload your audio file for comprehensive speech analysis"
    )
    
    if uploaded_file is not None:
        st.info(f"📁 File: {uploaded_file.name} | Size: {uploaded_file.size} bytes")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_audio_path = tmp_file.name
        
        # Process button
        if st.button("Start Analysis", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                with st.spinner("🔄 Processing audio... This may take a few minutes."):
                    
                    # === AUDIO PROCESSING ===
                    status_text.text("🎵 Transcribing audio...")
                    progress_bar.progress(10)
                    
                    result, original_text, duration, mfcc_sim = process_audio_file(temp_audio_path, whisper_model)
                    
                    if result is None or not original_text.strip():
                        st.error("❌ Failed to process audio file or no speech detected")
                        return
                    
                    progress_bar.progress(30)
                    status_text.text("✅ Audio transcription completed!")
                    
                    # Display transcript
                    st.subheader("📝 Transcript")
                    st.write(f"**Text:** {original_text}")
                    st.write(f"**Duration:** {duration:.2f} seconds")
                    
                    # === FEATURE EXTRACTION ===
                    status_text.text("🔬 Extracting features...")
                    progress_bar.progress(40)
                    
                    # Semantic Coherence
                    coherence = compute_semantic_coherence(original_text, sbert)
                    
                    # Pause Metrics
                    pause_freq = count_long_pauses_from_audio(temp_audio_path)
                    
                    # TTR
                    token_count, type_count, ttr = calculate_ttr(original_text)
                    
                    progress_bar.progress(50)
                    
                    # Pitch Range & Articulation
                    pitch_range = get_pitch_range(temp_audio_path, prosody_method)
                    articulation = get_articulation_rate(result)
                    mlr = calculate_mlr(result)
                    
                    # Prosody Features
                    prosody = analyze_prosody(temp_audio_path, prosody_method)
                    
                    progress_bar.progress(60)
                    
                    # Speech Rate
                    wpm, wps, total_words = speech_rate(original_text, duration)
                    
                    # Fluency Markers
                    flu = linking_discourse_filled_counts(original_text)
                    
                    progress_bar.progress(70)
                    
                    # Topic Similarity
                    sim = calculate_bert_topic_similarity(original_text, ref_topic, sbert)
                    
                    # Grammar Analysis
                    acc, err_count, _ = grammar_accuracy(original_text, tool, grammar_method)
                    
                    # Idiom Detection
                    found_idioms = find_idioms_in_text(original_text, idioms_english)
                    
                    progress_bar.progress(80)
                    
                    # CEFR Distribution
                    word_levels = map_words_to_cefr(original_text, cefr_df)
                    cefr_counts = count_cefr_distribution(word_levels)
                    
                    # Lexical Bundles
                    bundles = count_all_lexical_bundles(original_text)
                    
                    # Synonym Variations
                    syn_count, syn_pairs = count_synonym_variations(original_text)
                    
                    # Tree Depth
                    avg_depth, max_depth = get_avg_and_max_tree_depth(original_text, nlp)
                    
                    progress_bar.progress(90)
                    
                    # === BUILD FEATURE DICTIONARY ===
                    data = build_comprehensive_feature_dict(
                        uploaded_file, original_text, duration, mfcc_sim, coherence, 
                        pause_freq, token_count, type_count, ttr, pitch_range, 
                        articulation, mlr, prosody, wpm, wps, total_words, flu, 
                        sim, err_count, found_idioms, cefr_counts, bundles, 
                        syn_count, avg_depth, max_depth
                    )
                    
                    status_text.text("✅ Feature extraction completed!")
                    
                    # === PREDICTIONS ===
                    if loaded_models:
                        status_text.text("🤖 Making predictions...")
                        predictions, final_feature_array = predict_all_subconstructs_and_cefr(data, loaded_models)
                        st.info(f"🔍 Feature array length: {len(final_feature_array)} (expected: 46)")
                    else:
                        predictions = {name: "N/A" for name in list(SUBCONSTRUCTS.keys()) + ["CEFR"]}
                        final_feature_array = np.zeros(46)
                    
                    progress_bar.progress(100)
                    status_text.text("🎉 Analysis completed!")
                    
                    # === DISPLAY RESULTS ===
                    st.header("📊 Analysis Results")

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader("🎯 Prediction Scores")
                        
                        if loaded_models:
                            pred_data = []
                            for subconstruct, score in predictions.items():
                                cefr_level = CEFR_MAPPING.get(score, "Unknown")
                                pred_data.append({
                                    'Subconstruct': subconstruct,
                                    'Score': score,
                                    'CEFR Level': cefr_level
                                })
                            
                            pred_df = pd.DataFrame(pred_data)
                            st.dataframe(pred_df, use_container_width=True)
                            
                            cefr_score = predictions.get("CEFR", 0)
                            cefr_level = CEFR_MAPPING.get(cefr_score, "Unknown")
                            st.success(f"🎖️ **Final CEFR Assessment: {cefr_level}** (Score: {cefr_score})")
                            
                            numeric_predictions = {k: v for k, v in predictions.items() if isinstance(v, (int, float))}
                            if numeric_predictions:
                                pred_chart_df = pd.DataFrame({
                                    'Subconstruct': list(numeric_predictions.keys()),
                                    'Score': list(numeric_predictions.values())
                                })
                                st.bar_chart(pred_chart_df.set_index('Subconstruct'))
                        else:
                            st.info("Prediction models not available. Showing feature extraction results only.")

                    with col2:
                        st.subheader("📈 Key Metrics")
                        st.metric("Duration", f"{duration:.2f}s")
                        st.metric("Word Count", total_words)
                        st.metric("WPM", f"{wpm:.1f}")
                        st.metric("Grammar Errors", err_count)
                        st.metric("Topic Similarity", f"{sim:.1f}%")
                        st.metric("MFCC Similarity", f"{mfcc_sim:.1f}%")
                    
                    # Feature breakdown in tabs
                    st.subheader("🔍 Feature Analysis")
                    tab1, tab2, tab3 = st.tabs(["Core Features", "Language Features", "Detailed Results"])
                    
                    with tab1:
                        st.write("**Audio Features:**")
                        core_features = {
                            "Duration": f"{duration:.2f}s",
                            "MFCC Similarity": f"{mfcc_sim:.1f}%",
                            "Pitch Range": f"{pitch_range:.1f} Hz",
                            "Mean Pitch": f"{prosody['mean_pitch']:.1f}",
                            "Mean Energy": f"{prosody['mean_energy']:.3f}",
                            "Articulation Rate": f"{articulation:.2f}",
                            "MLR": f"{mlr:.2f}",
                            "Pause Frequency": pause_freq
                        }
                        st.json(core_features)
                    
                    with tab2:
                        st.write("**Text Features:**")
                        text_features = {
                            "Total Words": total_words,
                            "Token Count": token_count,
                            "Type Count": type_count,
                            "TTR": f"{ttr:.3f}",
                            "WPM": f"{wpm:.1f}",
                            "WPS": f"{wps:.2f}",
                            "Semantic Coherence": f"{coherence:.1f}%",
                            "Topic Similarity": f"{sim:.1f}%",
                            "Grammar Errors": err_count,
                            "Idioms Found": len(found_idioms),
                            "Filled Pauses": flu['filled_count']
                        }
                        st.json(text_features)
                    
                    with tab3:
                        if loaded_models:
                            st.write("**Complete Feature Array (46 features):**")
                            
                            # Split into numerical and subconstruct features
                            col_feat1, col_feat2 = st.columns(2)
                            
                            with col_feat1:
                                st.write("**Numerical Features (39):**")
                                numerical_df = pd.DataFrame({
                                    'Feature': NUMERICAL_FEATURES_ORDER,
                                    'Value': final_feature_array[:39]
                                })
                                st.dataframe(numerical_df, height=400)
                            
                            with col_feat2:
                                st.write("**Subconstruct Scores (7):**")
                                subconstruct_df = pd.DataFrame({
                                    'Subconstruct': list(SUBCONSTRUCTS.keys()),
                                    'Score': final_feature_array[39:46]
                                })
                                st.dataframe(subconstruct_df)
                        
                        # CEFR distribution
                        st.write("**CEFR Word Distribution:**")
                        cefr_data = pd.DataFrame({
                            'Level': list(cefr_counts.keys()),
                            'Count': list(cefr_counts.values())
                        })
                        if cefr_data['Count'].sum() > 0:
                            st.bar_chart(cefr_data.set_index('Level'))
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    
                    if loaded_models:
                        full_results = {**data, **predictions}
                        full_results["final_feature_array"] = final_feature_array.tolist()
                    else:
                        full_results = data
                    
                    results_df = pd.DataFrame([full_results])
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Results (CSV)",
                        data=csv_data,
                        file_name=f"speech_analysis_{uploaded_file.name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Clean up temporary file
                    os.unlink(temp_audio_path)
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                if 'temp_audio_path' in locals():
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass
            finally:
                progress_bar.empty()
                status_text.empty()

if __name__ == "__main__":
    main()
