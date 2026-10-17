"""Gradio demo: upload an infant cry recording, see the top-3 predicted need states.

Usage: python app.py   (then open http://127.0.0.1:7860)
Deploying as a Hugging Face Space: see docs/DEPLOY_HF_SPACE.md
"""
import gradio as gr
import joblib

from src import config
from src.features import extract_features, load_audio
from src.model import load_cnn, predict_proba

DISPLAY_NAMES = {"belly_pain": "belly pain", "burping": "burping", "cold_hot": "cold / hot",
                 "discomfort": "discomfort", "hungry": "hungry", "lonely": "lonely",
                 "scared": "scared", "tired": "tired"}
RF_CHOICE = "Random Forest (218 audio features) - better on test"
CNN_CHOICE = "CNN (log-mel spectrogram)"
MIN_SOUND_SEC = 1.0

rf_bundle = joblib.load(config.MODELS_DIR / "rf.joblib")
cnn_model, _ = load_cnn()


def predict(audio_path, model_choice):
    if audio_path is None:
        raise gr.Error("Upload a .wav recording first.")
    try:
        y = load_audio(audio_path)   # 22050 Hz, silence trimmed, peak-normalised
    except Exception as e:
        raise gr.Error(f"Could not read that file as audio ({type(e).__name__}). "
                       "Please upload a .wav file.")
    if len(y) < MIN_SOUND_SEC * config.SAMPLE_RATE:
        raise gr.Error(f"Less than {MIN_SOUND_SEC:.0f} s of sound after trimming silence. "
                       "Try a longer recording.")

    if model_choice == CNN_CHOICE:
        proba = predict_proba(cnn_model, [y])[0]
    else:
        X = rf_bundle["scaler"].transform(extract_features(y)[None])
        proba = rf_bundle["model"].predict_proba(X)[0]
    return {DISPLAY_NAMES[c]: float(p) for c, p in zip(config.CLASSES, proba)}


with gr.Blocks(title="Infant Cry State Recognition") as demo:
    gr.Markdown(
        "# Infant Cry State Recognition\n"
        "Upload a short recording of a baby crying (.wav, a few seconds). The model guesses "
        "which of 8 need states it sounds like: hungry, tired, discomfort, belly pain, "
        "burping, scared, lonely or cold/hot.")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["upload"], type="filepath", label="Cry recording (.wav)")
            model_choice = gr.Radio([RF_CHOICE, CNN_CHOICE], value=RF_CHOICE, label="Model")
            button = gr.Button("Predict", variant="primary")
        with gr.Column():
            output = gr.Label(num_top_classes=3, label="Top-3 predicted states")
    gr.Markdown(
        "**Honest read:** this is a student research demo trained on ~400 clips, most of them "
        "'hungry'. It is good at recognising 'hungry' and weak on the rare states, so treat "
        "the output as a guess, not advice. It is not a medical device. If a baby seems "
        "unwell, contact a doctor.")
    button.click(predict, inputs=[audio, model_choice], outputs=output)


if __name__ == "__main__":
    demo.launch()
