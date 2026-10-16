# Infant State Recognition from Cry Audio 👶🔊

Parents often can't tell why a baby is crying. This project classifies a short infant cry recording into one of **8 need states**: hungry, tired, discomfort, belly pain, burping, scared, lonely, and cold/hot.

## Approach

All of the work is in [`Baby_Cry_Scene.ipynb`](Baby_Cry_Scene.ipynb), which runs on Google Colab.

1. **Data cleaning.** The dataset has 512 labelled `.wav` clips. Silence is trimmed with a 30 dB threshold, and clips that are corrupt, shorter than 1 second after trimming, or more than 80% silent are dropped, leaving **511 clips**.
2. **Feature engineering.** Each clip becomes a **218-dimensional vector**, built from the mean and standard deviation over time of:
   - MFCC and delta-MFCC
   - zero-crossing rate, spectral centroid and RMS energy
   - chroma, a 40-band log-mel spectrogram, spectral contrast and tonnetz

   The audio uses 22,050 Hz sampling with 25 ms windows and a 10 ms hop.
3. **Handling imbalance.** The data is heavily imbalanced: "hungry" has 381 clips and "cold_hot" has 7, a 54:1 ratio. The split is stratified 80/20, the scaler is fit on training data only to avoid leakage, and **balanced class weights** are used.
4. **Model.** A cost-sensitive **Random Forest** with 300 trees, evaluated with 5-fold stratified cross-validation.

## Results

| Metric | Score |
|---|---|
| Test accuracy | 81.6% |
| Test macro F1 | 32.0% |
| 5-fold CV macro F1 | 0.309 ± 0.060 |

**Honest read:** accuracy looks good mainly because about 75% of the test set is "hungry", which the model detects well (F1 0.91). Burping (F1 0.86) and scared (F1 0.80) also work. Rare classes such as belly pain, lonely and cold/hot have only 1–6 test clips and are not yet recognised. That's why **macro F1 is the metric to improve**.

## Next steps

- [ ] Data augmentation for rare classes: pitch shift, time stretch and added noise.
- [ ] A CNN on log-mel spectrograms, compared against the Random Forest.
- [ ] Confusion matrix and per-class error analysis.
- [ ] Save the model and add a small Streamlit or Hugging Face Spaces demo.

## Run it

1. Open the notebook in Google Colab.
2. Put the dataset folder (one sub-folder per class) on your Google Drive.
3. Update `CONFIG['dataset_path']` in Cell 2.
4. Run all cells.

**Stack:** Python · librosa · NumPy · Pandas · scikit-learn · Matplotlib · Seaborn
