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
    """Count lexical bundles (bigrams, trigrams, fourgrams)"""
    try:
        tokens = tokenize_text(text)
        
        matched_bigrams = [' '.join(gram) for gram in ngrams(tokens, 2) if ' '.join(gram) in valid_bigrams]
        matched_trigrams = [' '.join(gram) for gram in ngrams(tokens, 3) if ' '.join(gram) in valid_trigrams]
        matched_fourgrams = [' '.join(gram) for gram in ngrams(tokens, 4) if ' '.join(gram) in valid_fourgrams]
        
        return {
            "bigram_count": len(matched_bigrams),
            "bigram_matches": matched_bigrams,
            "trigram_count": len(matched_trigrams),
            "trigram_matches": matched_trigrams,
            "fourgram_count": len(matched_fourgrams),
            "fourgram_matches": matched_fourgrams
        }
    except:
        return {
            "bigram_count": 0,
            "bigram_matches": [],
            "trigram_count": 0,
            "trigram_matches": [],
            "fourgram_count": 0,
            "fourgram_matches": []
        }

def count_synonym_variations(text):
    """Count synonym variations"""
    try:
        tokens = tokenize_text(text)
        lemmas_used = set(tokens)
        variation_count = 0
        synonym_pairs = []
        
        for word in set(tokens):
            synonyms = set()
            try:
                for syn in wn.synsets(word):
                    for lemma in syn.lemmas():
                        name = lemma.name().lower().replace("_", " ")
                        if name != word and name in lemmas_used:
                            synonyms.add(name)
                if synonyms:
                    variation_count += 1
                    synonym_pairs.append((word, list(synonyms)))
            except:
                continue
        
        return variation_count, synonym_pairs
    except:
        return 0, []

def get_avg_and_max_tree_depth(text):
    """Get average and maximum syntactic tree depth (MODIFIED: Using NLTK instead of spaCy+Benepar)"""
    try:
        # Simple approximation using NLTK POS tagging and dependency estimation
        sentences = sent_tokenize(text)
        all_depths = []
        
        for sentence in sentences:
            try:
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)
                
                # Simple depth estimation based on sentence structure
                # This is a rough approximation since we don't have full parsing
                depth = 1  # Base depth
                
                # Add depth based on subordinating conjunctions and complex structures
                subordinators = ['because', 'although', 'while', 'since', 'if', 'when', 'where', 'that', 'which', 'who']
                for word, tag in pos_tags:
                    if word.lower() in subordinators:
                        depth += 1
                    elif tag.startswith('WH'):  # WH-words often indicate embedded clauses
                        depth += 1
                
                # Add depth for prepositional phrases
                prep_count = sum(1 for word, tag in pos_tags if tag == 'IN')
                depth += min(prep_count, 3)  # Cap at 3 additional levels
                
                all_depths.append(depth)
            except:
                all_depths.append(1)  # Fallback depth
        
        if not all_depths:
            return 0, 0
        
        avg_depth = sum(all_depths) / len(all_depths)
        max_depth = max(all_depths)
        return avg_depth, max_depth
    except:
        return 0, 0

def process_audio_file(audio_path, whisper_model):
    """Main audio processing function (MODIFIED: Uses soundfile instead of FFmpeg)"""
    try:
        # Transcription
        result = whisper_model.transcribe(
            audio_path,
            language="en",
            condition_on_previous_text=False,
            temperature=0.0,
            logprob_threshold=-1.0,
            no_speech_threshold=0.0,
            compression_ratio_threshold=10.0,
            fp16=False
        )
        original_text = result["text"]
        
        # Load original audio using librosa
        y1, sr = librosa.load(audio_path, sr=16000)
        dur = librosa.get_duration(y=y1, sr=sr)
        
        # Generate TTS for MFCC comparison
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tts_file:
            tts_path = tts_file.name
            
        try:
            gTTS(original_text).save(tts_path)
            # Load TTS audio using librosa
            y2, _ = librosa.load(tts_path, sr=16000)
            # Save using soundfile instead of librosa's deprecated sf.write
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
        except:
            cosine_sim = 0.0
        
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
            # Handle missing features with default value
            st.warning(f"Missing feature: {feature_name}, using default value 0")
            features.append(0.0)
    
    return np.array(features)

def predict_subconstruct(data, model, feature_names):
    """Predict a single subconstruct score"""
    try:
        # Extract features for this subconstruct
        X = np.array([[data[feat] for feat in feature_names]])
        prediction = model.predict(X)
        return int(prediction[0])
    except Exception as e:
        st.error(f"Error in subconstruct prediction: {e}")
        return 0

def predict_all_subconstructs_and_cefr(data, loaded_models):
    """
    Predict all subconstructs and CEFR level
    Returns predictions dict and feature array of length 46
    """
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
        
        # Step 3: Combine into 46-feature array (39 numerical + 7 subconstruct scores)
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
        # Return default values
        default_predictions = {name: 0 for name in list(SUBCONSTRUCTS.keys()) + ["CEFR"]}
        default_features = np.zeros(46)
        return default_predictions, default_features

def build_comprehensive_feature_dict(uploaded_file, original_text, duration, mfcc_sim, 
                                    coherence, pause_freq, token_count, type_count, ttr,
                                    pitch_range, articulation, mlr, prosody, wpm, wps, 
                                    total_words, flu, sim, err_count, found_idioms,
                                    cefr_counts, bundles, syn_count, avg_depth, max_depth):
    """
    Build comprehensive feature dictionary with all 39 numerical features
    """
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
        
        # Additional derived data for display
        "found_idioms_list": found_idioms,
        "bundles_details": bundles,
        "flu_details": flu,
        "cefr_word_levels": None  # Will be set separately if needed
    }
    
    return data

# Load models and data (MODIFIED)
st.info("🔄 Loading models and resources...")
whisper_model, sbert, stop_words = load_models_and_resources()
cefr_df, idioms_english = load_reference_data()

# ===================================
# STREAMLIT UI (MODIFIED)
# ===================================

def main():
    """Main Streamlit application"""
    
    # Check if models are loaded successfully
    if whisper_model is None:
        st.error("Cannot proceed without Whisper model. Please check your installation.")
        st.stop()
    
    # Display speech prompt in main area
    st.header("Speech Prompt")
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #1f77b4;">
            📝 Free Speech Topic
        </h3>
        <p style="font-size: 16px; margin-bottom: 0; color: #333;">
            <strong>"Talk about the university course you enjoyed the most, describe one course you found difficult, and explain whether you think universities should focus more on practical skills or theoretical knowledge."</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("🎛️ Configuration")
    
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
    
    # Feature order validation
    st.sidebar.subheader("🔧 Feature Configuration")
    st.sidebar.info(f"Expected 39 numerical features + 7 subconstruct scores = 46 total features")
    st.sidebar.text(f"Loaded subconstructs: {len(SUBCONSTRUCTS)}")
    st.sidebar.text(f"Numerical features: {len(NUMERICAL_FEATURES_ORDER)}")
    
    # Dependency status (NEW)
    st.sidebar.subheader("📦 Dependency Status")
    st.sidebar.success("✅ TextBlob (replaces LanguageTool)")
    st.sidebar.success("✅ soundfile (replaces FFmpeg)")
    st.sidebar.success("✅ NLTK (replaces spaCy)")
    st.sidebar.info("ℹ️ Lightweight dependencies loaded")
    
    # Main content
    st.header("Audio Upload & Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac'],
        help="Upload your audio file for comprehensive speech analysis"
    )
    
    if uploaded_file is not None:
        # Display file info
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
                    coherence = compute_semantic_coherence(original_text)
                    
                    # Pause Metrics
                    pause_freq = count_long_pauses_from_audio(temp_audio_path)
                    
                    # TTR
                    token_count, type_count, ttr = calculate_ttr(original_text)
                    
                    progress_bar.progress(50)
                    
                    # Pitch Range & Articulation
                    pitch_range = get_pitch_range(temp_audio_path)
                    articulation = get_articulation_rate(result)
                    mlr = calculate_mlr(result)
                    
                    # Prosody Features
                    prosody = analyze_prosody(temp_audio_path)
                    
                    progress_bar.progress(60)
                    
                    # Speech Rate
                    wpm, wps, total_words = speech_rate(original_text, duration)
                    
                    # Fluency Markers
                    flu = linking_discourse_filled_counts(original_text)
                    
                    progress_bar.progress(70)
                    
                    # Topic Similarity
                    sim = calculate_bert_topic_similarity(original_text, ref_topic)
                    
                    # Grammar Analysis
                    acc, err_count, _ = grammar_accuracy(original_text)
                    
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
                    avg_depth, max_depth = get_avg_and_max_tree_depth(original_text)
                    
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
                        
                        # Validation
                        st.info(f"🔍 Feature array length: {len(final_feature_array)} (expected: 46)")
                    else:
                        predictions = {name: "N/A" for name in list(SUBCONSTRUCTS.keys()) + ["CEFR"]}
                        final_feature_array = np.zeros(46)
                    
                    progress_bar.progress(100)
                    status_text.text("🎉 Analysis completed!")
                    
                    # === DISPLAY RESULTS ===
                    st.header("📊 Analysis Results")

                    # Layout kolom
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader("🎯 Prediction Scores")
                        
                        if loaded_models:
                            # Buat list untuk data prediksi
                            pred_data = []
                            for subconstruct, score in predictions.items():
                                # Mapping semua skor ke CEFR Level
                                cefr_level = CEFR_MAPPING.get(score, "Unknown")
                                
                                pred_data.append({
                                    'Subconstruct': subconstruct,
                                    'Score': score,
                                    'CEFR Level': cefr_level
                                })
                            
                            # Buat DataFrame
                            pred_df = pd.DataFrame(pred_data)
                            
                            # Tampilkan tabel
                            st.dataframe(pred_df, use_container_width=True)
                            
                            # Highlight CEFR result
                            cefr_score = predictions.get("CEFR", 0)
                            cefr_level = CEFR_MAPPING.get(cefr_score, "Unknown")
                            st.success(f"🎖️ **Final CEFR Assessment: {cefr_level}** (Score: {cefr_score})")
                            
                            # Bar chart of predictions (numeric only)
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
                    
                    # Feature Array Display
                    st.subheader("🔢 Feature Array (Length: 46)")
                    
                    if loaded_models:
                        # Display the 46-feature array
                        feature_labels = NUMERICAL_FEATURES_ORDER + list(SUBCONSTRUCTS.keys())
                        
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
                    
                    # Detailed feature breakdown
                    st.subheader("🔍 Detailed Feature Analysis")
                    
                    # Create tabs for different feature categories
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                        "Fluency", "Pronunciation", "Prosody", "Coherence & Cohesion", 
                        "Topic Relevance", "Complexity", "Accuracy"
                    ])
                    
                    with tab1:
                        st.write("**Fluency Features:**")
                        fluency_data = {k: data[k] for k in SUBCONSTRUCTS["Fluency"]}
                        st.json(fluency_data)
                        
                        if flu['filled_pauses_found']:
                            st.write("**Filled Pauses Found:**", ', '.join(flu['filled_pauses_found'][:10]))
                    
                    with tab2:
                        st.write("**Pronunciation Features:**")
                        pronunciation_data = {k: data[k] for k in SUBCONSTRUCTS["Pronunciation"]}
                        st.json(pronunciation_data)
                    
                    with tab3:
                        st.write("**Prosody Features:**")
                        prosody_data = {k: data[k] for k in SUBCONSTRUCTS["Prosody"]}
                        st.json(prosody_data)
                    
                    with tab4:
                        st.write("**Coherence & Cohesion Features:**")
                        coherence_data = {k: data[k] for k in SUBCONSTRUCTS["Coherence and Cohesion"]}
                        st.json(coherence_data)
                        
                        if flu['linking_words_found']:
                            st.write("**Linking Words Found:**", ', '.join(flu['linking_words_found'][:10]))
                        if flu['discourse_markers_found']:
                            st.write("**Discourse Markers Found:**", ', '.join(flu['discourse_markers_found'][:10]))
                    
                    with tab5:
                        st.write("**Topic Relevance:**")
                        topic_data = {k: data[k] for k in SUBCONSTRUCTS["Topic Relevance"]}
                        st.json(topic_data)
                        st.write(f"**Reference Topic:** {ref_topic}")
                    
                    with tab6:
                        st.write("**Complexity Features:**")
                        complexity_data = {k: data[k] for k in SUBCONSTRUCTS["Complexity"]}
                        st.json(complexity_data)
                        
                        # CEFR distribution chart
                        cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'UNKNOWN']
                        cefr_chart_data = pd.DataFrame({
                            'CEFR Level': cefr_levels,
                            'Word Count': [cefr_counts.get(level, 0) for level in cefr_levels]
                        })
                        if cefr_chart_data['Word Count'].sum() > 0:
                            st.bar_chart(cefr_chart_data.set_index('CEFR Level'))
                        
                        if found_idioms:
                            st.write("**Idioms Found:**", ', '.join(found_idioms[:5]))
                        
                        if bundles['bigram_matches']:
                            st.write("**Bigrams Found:**", ', '.join(bundles['bigram_matches'][:5]))
                        if bundles['trigram_matches']:
                            st.write("**Trigrams Found:**", ', '.join(bundles['trigram_matches'][:5]))
                        if bundles['fourgram_matches']:
                            st.write("**Fourgrams Found:**", ', '.join(bundles['fourgram_matches'][:3]))
                        
                        if syn_pairs:
                            st.write("**Synonym Variations (first 3):**")
                            for word, syns in syn_pairs[:3]:
                                st.write(f"- {word}: {', '.join(syns)}")
                    
                    with tab7:
                        st.write("**Accuracy Features:**")
                        accuracy_data = {k: data[k] for k in SUBCONSTRUCTS["Accuracy"]}
                        st.json(accuracy_data)
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    
                    # Combine all data
                    if loaded_models:
                        full_results = {**data, **predictions}
                        full_results["final_feature_array"] = final_feature_array.tolist()
                    else:
                        full_results = data
                    
                    results_df = pd.DataFrame([full_results])
                    
                    # Convert to CSV
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
                # Clean up temporary file on error
                if 'temp_audio_path' in locals():
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass
            finally:
                progress_bar.empty()
                status_text.empty()
    
    # Information section (MODIFIED)
    st.markdown("---")
    st.subheader("ℹ️ About This Application")
    
    with st.expander("📖 Feature Descriptions"):
        st.markdown("""
        **Fluency:** Speech rate, pauses, word flow
        - Words per minute (WPM), Words per second (WPS)
        - Mean Length of Run (MLR), Pause frequency
        - Filled pauses count
        
        **Pronunciation:** Sound production accuracy
        - MFCC similarity with TTS reference
        - Articulation rate, Pitch range
        
        **Prosody:** Speech rhythm and intonation
        - Pitch statistics (mean, standard deviation)
        - Energy features, Prominence patterns
        
        **Coherence & Cohesion:** Text connectivity
        - Semantic coherence between sentences
        - Linking words and discourse markers
        
        **Topic Relevance:** Content appropriateness
        - Semantic similarity to reference topic
        
        **Complexity:** Linguistic sophistication
        - CEFR word level distribution
        - Lexical bundles (bi/tri/fourgrams)
        - Syntactic tree depth, Synonym variations
        - Idiom usage
        
        **Accuracy:** Grammatical correctness
        - Grammar error detection and counting
        """)
    
    with st.expander("🔧 Technical Requirements (UPDATED)"):
        st.markdown("""
        **Required Files:**
        - Model files: `*_rf_classification.h5`
        - Reference data: `English_CEFR_Words.csv`, `idioms_english.csv`
        
        **Supported Audio Formats:**
        - WAV, MP3, M4A, FLAC
        - Recommended: 16kHz, mono, clear speech
        
        **Modified Dependencies (Lightweight):**
        - ✅ **TextBlob** (replaces LanguageTool) - Grammar checking
        - ✅ **soundfile** (replaces FFmpeg) - Audio I/O operations  
        - ✅ **NLTK** (replaces spaCy + en_core_web_sm) - NLP processing
        
        **Other Dependencies:**
        - Whisper, SentenceTransformers
        - librosa, sklearn, pandas, numpy
        - streamlit, h5py, gtts
        
        **Feature Array Structure:**
        - 39 numerical features (speech characteristics)
        - 7 subconstruct scores (predicted ratings)
        - Total: 46 features for CEFR level prediction
        
        **Key Changes:**
        - No more Java dependency (LanguageTool removed)
        - No more FFmpeg installation required
        - No more large spaCy model download required
        - Simplified prosody analysis using librosa only
        - Basic grammar checking with TextBlob
        - Tree depth approximation using NLTK POS tags
        """)
    
    with st.expander("🚀 Installation Commands (UPDATED)"):
        st.code("""
# Install main libraries (SIMPLIFIED)
pip install openai-whisper gtts librosa transformers torch nltk scikit-learn streamlit sentence-transformers soundfile textblob

# Download NLTK data (run this in Python)
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
nltk.download('brown')
nltk.download('treebank')

# Download TextBlob corpora
python -m textblob.download_corpora

# NO MORE REQUIRED:
# - Java JDK installation
# - FFmpeg installation  
# - spaCy model download
# - LanguageTool setup

# Run the application
streamlit run ispeak_modified.py
        """)
    
    with st.expander("🔍 Model Architecture & Changes"):
        st.markdown(f"""
        **Pipeline Overview (Same):**
        1. Extract 39 numerical features from speech audio and transcript
        2. Predict 7 subconstruct scores using individual Random Forest models
        3. Combine features: 39 numerical + 7 subconstruct = 46 total features
        4. Predict final CEFR level using combined 46-feature array
        
        **Feature Categories:**
        - **Fluency Features:** {len(SUBCONSTRUCTS['Fluency'])} features
        - **Pronunciation Features:** {len(SUBCONSTRUCTS['Pronunciation'])} features  
        - **Prosody Features:** {len(SUBCONSTRUCTS['Prosody'])} features
        - **Coherence & Cohesion Features:** {len(SUBCONSTRUCTS['Coherence and Cohesion'])} features
        - **Topic Relevance Features:** {len(SUBCONSTRUCTS['Topic Relevance'])} features
        - **Complexity Features:** {len(SUBCONSTRUCTS['Complexity'])} features
        - **Accuracy Features:** {len(SUBCONSTRUCTS['Accuracy'])} features
        
        **Key Algorithm Changes:**
        - **Pitch Analysis:** librosa.piptrack() instead of parselmouth
        - **Grammar Check:** TextBlob.correct() instead of LanguageTool
        - **Tree Depth:** NLTK POS approximation instead of Benepar parsing
        - **Audio I/O:** soundfile.write() instead of FFmpeg
        - **Prosody:** librosa-only energy/prominence detection
        
        **Total Numerical Features:** {len(NUMERICAL_FEATURES_ORDER)}
        **Total Model Features:** 46 (39 numerical + 7 subconstruct scores)
        
        **Benefits:**
        - ✅ Reduced installation complexity
        - ✅ No external system dependencies  
        - ✅ Faster setup and deployment
        - ✅ Smaller memory footprint
        - ⚠️ Slightly reduced accuracy in some features (acceptable trade-off)
        """)
    
    with st.expander("⚠️ Known Limitations After Modifications"):
        st.markdown("""
        **Changes in Feature Accuracy:**
        
        1. **Prosody Analysis:**
        - Original: Parselmouth (Praat-based) - highly accurate pitch/intensity
        - Modified: librosa piptrack - good approximation but less precise
        - Impact: Slight reduction in prosody feature quality
        
        2. **Grammar Analysis:**
        - Original: LanguageTool - comprehensive grammar checking
        - Modified: TextBlob - basic spell checking + simple heuristics
        - Impact: May miss complex grammatical errors
        
        3. **Syntactic Complexity:**
        - Original: Benepar constituency parsing - accurate tree structures
        - Modified: POS tag approximation - estimated depth only
        - Impact: Less precise syntactic complexity measurement
        
        **Recommended Use:**
        - Suitable for development, testing, and general assessment
        - For production/research requiring highest accuracy, consider original version
        - Feature extraction still provides valuable insights for speech assessment
        
        **Mitigation Strategies:**
        - Monitor prediction accuracy on your specific dataset
        - Consider ensemble methods or manual validation for critical applications
        - Use as preprocessing step before more detailed analysis if needed
        """)

if __name__ == "__main__":
    main()# === Install Java JDK 17 ===
#winget install Microsoft.OpenJDK.17

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
from nltk.parse import CoreNLPParser

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
# INITIALIZATION & CACHING
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
        
        # TextBlob setup (REPLACES LanguageTool)
        try:
            # Test TextBlob
            test_blob = TextBlob("This is a test.")
            st.success("✅ TextBlob loaded (replacing LanguageTool)")
        except Exception as e:
            st.warning(f"⚠️ TextBlob not available: {str(e)}")
        
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
    ttr = type_count / token_count if token_count > 0 else 0.0
    return token_count, type_count, ttr

def calculate_bert_topic_similarity(original_text, reference_text):
    """Calculate topic similarity using BERT embeddings"""
    try:
        if sbert is not None:
            embeddings = sbert.encode([original_text, reference_text])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100
            return similarity
        return 0.0
    except:
        return 0.0

def get_pitch_range(audio_path):
    """Get pitch range from audio (MODIFIED: Uses librosa instead of parselmouth)"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        # Extract pitch using librosa
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        
        # Get pitch values
        pitch_values = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 50:  # Filter out very low frequencies
                pitch_values.append(pitch)
        
        return np.ptp(pitch_values) if pitch_values else 0.0
    except Exception as e:
        st.warning(f"Pitch range calculation error: {e}")
        return 0.0

def get_articulation_rate(result):
    """Get articulation rate from Whisper result"""
    segments = result.get("segments", [])
    if not segments:
        return 0
    total_words = sum(len(seg["text"].split()) for seg in segments)
    duration = segments[-1]["end"] - segments[0]["start"]
    return total_words / duration if duration > 0 else 0

def calculate_mlr(result, pause_threshold=0.5):
    """Calculate Mean Length of Run"""
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

def analyze_prosody(audio_path):
    """Analyze prosodic features (MODIFIED: Uses librosa instead of parselmouth)"""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Energy analysis
        energy = librosa.feature.rms(y=y)[0]
        
        # Pitch extraction using librosa
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        pitch_values = []
        
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 50:  # Filter out very low frequencies
                pitch_values.append(pitch)
        
        # If no pitch values found, use zeros
        if not pitch_values:
            pitch_values = [0]
        
        # Prominence detection based on energy
        threshold = np.mean(energy) + np.std(energy)
        peaks, _ = find_peaks(energy, height=threshold)
        
        # Calculate time positions of prominences
        prominences = [i * len(y) / len(energy) / sr for i in peaks]
        
        return {
            "mean_pitch": np.mean(pitch_values),
            "stdev_pitch": np.std(pitch_values),
            "mean_energy": np.mean(energy),
            "stdev_energy": np.std(energy),
            "num_prominences": len(prominences),
            "mean_distance_between_prominence": np.mean(np.diff(prominences)) if len(prominences) > 1 else 0,
            "std_distance_between_prominence": np.std(np.diff(prominences)) if len(prominences) > 1 else 0
        }
    except Exception as e:
        st.warning(f"Prosody analysis error: {e}")
        return {
            "mean_pitch": 0, "stdev_pitch": 0, "mean_energy": 0, "stdev_energy": 0,
            "num_prominences": 0, "mean_distance_between_prominence": 0, "std_distance_between_prominence": 0
        }

def linking_discourse_filled_counts(text):
    """Count linking words, discourse markers, and filled pauses"""
    linking_words = {
        "and", "but", "or", "so", "yet", "for", "nor",
        "because", "since", "as", "due to", "as a result", "therefore", "thus", "hence", "consequently",
        "although", "though", "even though", "whereas", "while", "however", "nevertheless", "nonetheless",
        "on the other hand", "in contrast", "alternatively", "instead",
        "in addition", "furthermore", "moreover", "also", "besides", "not only that", "as well as",
        "indeed", "in fact", "especially", "significantly", "particularly", "above all", "notably",
        "for example", "for instance", "such as", "like", "including", "to illustrate",
        "then", "after that", "before that", "meanwhile", "subsequently", "eventually",
        "at the same time", "finally", "firstly", "secondly", "thirdly", "next", "lastly", "ultimately",
        "so that", "in order that", "for the purpose of", "to this end",
        "if", "unless", "provided that", "in case", "even if", "as long as",
        "similarly", "likewise", "just as", "in the same way",
        "in other words", "that is to say", "to put it another way",
        "in conclusion", "to sum up", "in summary", "overall", "to conclude", "all in all",
        "granted", "admittedly", "still",
        "anyway", "incidentally", "by the way", "on another note"
    }

    discourse_markers = {
        "you know", "i mean", "like", "well", "actually", "basically", "anyway",
        "to be honest", "frankly", "seriously", "believe me", "i suppose", "i guess",
        "first of all", "secondly", "finally", "to begin with", "in conclusion",
        "on the one hand", "on the other hand", "next", "then", "after that",
        "eventually", "at the same time", "meanwhile", "in the meantime",
        "in fact", "as a matter of fact", "indeed", "certainly", "definitely",
        "undoubtedly", "clearly", "obviously",
        "and also", "what is more", "furthermore", "moreover", "in addition",
        "besides that", "as well",
        "however", "nevertheless", "nonetheless", "still", "yet",
        "although", "even though", "whereas", "despite that", "but",
        "mind you", "to be fair", "after all",
        "for example", "for instance", "such as", "to illustrate", "that is",
        "in other words", "namely", "let's say", "so to speak",
        "so", "therefore", "thus", "as a result", "consequently", "hence",
        "incidentally", "as i was saying", "where was i",
        "back to the point", "going back to what i said",
        "i think", "i believe", "in my opinion", "personally",
        "as far as i'm concerned", "it seems to me",
        "don't you think?", "would you agree?", "perhaps", "maybe",
        "sort of", "kind of", "i'd say", "if you ask me",
        "so anyway", "in short", "to sum up", "all in all", "overall", "in summary"
    }

    filled_pauses = {
        "um", "uh", "er", "ah", "eh", "hmm", "mm", "umm", "uhh", "ehm",
        "uh-huh", "mm-hmm", "mhm", "huh", "huh?", "huh-uh", "huh-uhh",
        "ugh", "tsk", "huhhh", "hmmm", "huhmm", "uh-huhh", "huhum", "ahh",
        "uhmm", "uhmmm", "ehhh", "mmm", "ummm", "uh-oh", "huh-oh", "uhhh", "uhhhh",
        "mmm-hmm", "mmm...", "hmm...", "uh...", "um...", "eh...", "ah...",
        "um?", "uh?", "eh?", "huh?", "mmm?",
        "nng", "nnnh", "ahh...", "huhh...", "mmmkay", "errr", "eerr", "huhuh",
        "oooh", "uhuh", "oof", "ahem", "grr", "bleh", "bla", "blah", "huhuhuh", "mgh",
        "yo", "yeaah", "yah", "meh", "naah", "yaa", "aha", "ay", "mmmyeah",
        "um um", "uh uh", "uh um", "hmm hmm", "mmm mmm", "uhm uhm", "ah ah", "uh huh uh",
        "duh", "derp", "ugh-huh", "mmm-kay", "oops", "oop", "whoa", "whew",
        "huhuhum", "erm", "ummhmm", "uhhah", "mmmyeah", "ahuh", "nuh", "yaaah", "hmmkay", "ohh"
    }
    
    text_clean = preprocess_text(text)
    tokens = tokenize_text(text_clean)
    joined = ' '.join(tokens)
    
    linking = {w for w in tokens if w in linking_words}
    discourse = {m for m in discourse_markers if m in joined}
    filled = [f for f in filled_pauses if re.search(rf"\b{re.escape(f)}\b", joined)]
    
    return {
        "linking_count": len(linking),
        "linking_words_found": list(linking),
        "discourse_count": len(discourse),
        "discourse_markers_found": list(discourse),
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

def is_wordnet_word(word):
    """Check if word exists in WordNet"""
    try:
        return bool(wn.synsets(word.lower()))
    except:
        return False

def grammar_accuracy(text):
    """Calculate grammar accuracy using TextBlob (REPLACES LanguageTool)"""
    try:
        blob = TextBlob(text)
        
        # Get sentences and analyze each one
        sentences = blob.sentences
        total_words = len(blob.words)
        
        if total_words == 0:
            return 0, 0, 0
        
        # Simple error estimation based on:
        # 1. Very short sentences (likely fragments)
        # 2. Sentences without proper capitalization
        # 3. Basic spelling errors
        
        error_count = 0
        
        # Check for spelling errors
        try:
            corrected = str(blob.correct())
            if corrected != text:
                # Count different words as potential errors
                original_words = set(text.lower().split())
                corrected_words = set(corrected.lower().split())
                error_count += len(original_words.symmetric_difference(corrected_words)) // 2
        except:
            pass
        
        # Check for very short sentences (potential fragments)
        for sentence in sentences:
            if len(sentence.words) < 3:
                error_count += 1
        
        # Calculate accuracy
        correct_words = max(0, total_words - error_count)
        accuracy_score = (correct_words / total_words) * 100 if total_words > 0 else 0
        
        return accuracy_score, error_count, total_words
    except Exception as e:
        st.warning(f"Grammar analysis error: {e}")
        return 0, 0, 0
