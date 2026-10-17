# Deploying the demo as a Hugging Face Space

The demo (`app.py`) only needs the code in `src/` and the two small trained models in
`models/` (`rf.joblib` ≈ 3 MB, `cnn.pt` ≈ 1.5 MB). The dataset is **not** needed and must
not be uploaded.

## 1. Run it locally first

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-app.txt
python app.py            # open http://127.0.0.1:7860
```

If `models/` is empty, train first: `python -m src.train_rf` and `python -m src.train_cnn`.

## 2. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Pick a name (e.g. `infant-cry-state-recognition`), **SDK: Gradio**, hardware **CPU basic**
   (free is enough), visibility as you like.
3. Log in from the terminal so git can push. Use an access token with *write* scope from
   <https://huggingface.co/settings/tokens>:

   ```bash
   pip install -U huggingface_hub
   hf auth login
   ```

## 3. Copy the app files into the Space repo

```bash
git clone https://huggingface.co/spaces/<your-username>/<space-name> hf-space
cp -r app.py requirements.txt src hf-space/
mkdir -p hf-space/models && cp models/rf.joblib models/cnn.pt hf-space/models/
rm -rf hf-space/src/__pycache__
```

Spaces read their settings from YAML at the top of the Space's own `README.md`. Replace the
generated `hf-space/README.md` with:

```markdown
---
title: Infant Cry State Recognition
emoji: 👶
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
---

Upload a baby-cry recording (.wav) and see the top-3 predicted need states.
Code, results and limitations: https://github.com/<your-github-username>/Infant-State-Recognition-System
```

`sdk_version` installs Gradio, so `requirements.txt` doesn't need to list it.

Optional, for a faster build: add this as the first line of `hf-space/requirements.txt` to
get the small CPU-only PyTorch wheel instead of the CUDA one:

```
--extra-index-url https://download.pytorch.org/whl/cpu
```

## 4. Push

```bash
cd hf-space
git add .
git commit -m "Deploy infant cry demo"
git push
```

The Space builds automatically (a few minutes the first time). Follow progress in the
**Logs** tab. When it shows *Running*, the demo is live at
`https://huggingface.co/spaces/<your-username>/<space-name>`. Put that link in the main
README's demo section.

## Updating

After retraining, copy the new `models/*.{joblib,pt}` (and any changed `src/` files) into
`hf-space/` again, then commit and push. Keep the scikit-learn and torch versions in the
Space's `requirements.txt` identical to the ones used for training, because pickled
models aren't guaranteed to load across library versions.

## Troubleshooting

- **`ModuleNotFoundError: src`**: `src/` (with `__init__.py`) must sit next to `app.py` at the
  Space root.
- **Pickle / `InconsistentVersionWarning` errors when loading `rf.joblib`**: the Space installed a
  different scikit-learn version. Pin the same version as `requirements.txt`.
- **Upload rejected for large files**: files over 10 MB need Git LFS (`git lfs track "*.pt"`).
  The current models are smaller than that.
