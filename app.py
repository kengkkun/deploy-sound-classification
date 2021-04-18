import os
import json
import requests
import streamlit as st
import tensorflow as tf
import SessionState
import numpy as np
from utils import classes_and_models, update_logger, predict_json, load_audio_file

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'sound-classification-311017-142f151a7f9b.json'
PROJECT = "sound-classification-311017"
REGION = "asia-southeast1"

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Welcome to environments sound classification")
st.header("Identify what's in your sounds!")


@st.cache
def make_prediction(sound, model, class_names):
    # sound = load_and_prep_sound(sound)

    sound = load_audio_file(sound)

    sound = np.float32(sound.reshape(sound.shape[0], sound.shape[1], 1))

    sound = tf.cast(tf.expand_dims(sound, axis=0), tf.int16)

    # st.write(sound.shape)

    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=sound)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    return sound, pred_class, pred_conf


choose_model = st.sidebar.selectbox(
    "Pick model version (beta)",
    ("Model 1 (20 classes)",  # original 10 classes
     "Model 2",  # original 10 classes + donuts
     "Model 3")  # 11 classes (same as above) + not_food class
)

# Model choice logic
if (choose_model == "Model 1 (20 classes)"):
    CLASSES = classes_and_models["model_1"]["classes"]
    MODEL = classes_and_models["model_1"]["model_name"]


# Display info about model and classes
if st.checkbox('Show classes'):
    st.write(
        f'You chose {MODEL}, these are the classes of environmental sound it can identify:\n', CLASSES)

# File uploader allows user to add their own sound
uploaded_file = st.file_uploader(
    label='Upload a sound of environmental')

session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload a sound.")
    st.stop()
else:
    session_state.uploaded_sound = uploaded_file.read()
    st.audio(session_state.uploaded_sound, format='audio/wav')
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True

# And if they did...
if session_state.pred_button:
    session_state.sound, session_state.pred_class, session_state.pred_conf = make_prediction(
        session_state.uploaded_sound, model=MODEL, class_names=CLASSES)
    st.write(f"Prediction: {session_state.pred_class}, \
               Confidence: {session_state.pred_conf:.3f}")

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something...)
        print(update_logger(sound=session_state.sound,
                            model_used=MODEL,
                            pred_class=session_state.pred_class,
                            pred_conf=session_state.pred_conf,
                            correct=True))
    elif session_state.feedback == "No":
        session_state.correct_class = st.text_input(
            "What should the correct label be?")
        if session_state.correct_class:
            st.write(
                "Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(sound=session_state.sound,
                                model_used=MODEL,
                                pred_class=session_state.pred_class,
                                pred_conf=session_state.pred_conf,
                                correct=False,
                                user_label=session_state.correct_class))
