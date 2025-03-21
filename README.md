# Red-Teaming-Classifier

## Setting up the local environment

Create a virtual environment in the same directory as the cloned repository with the command:

```python -m venv venv```

You can now activate the virtual environment with:

- Windows: ```venv\Scripts\activate```
- MacOS: ```source venv/bin/activate```

## Generating Features (TF-IDF + N-gram)

Once preprocessing is complete and `preprocessed_data.csv` is available in the `data/` directory, run the feature engineering script:

```bash
python src/feature_engineering.py
