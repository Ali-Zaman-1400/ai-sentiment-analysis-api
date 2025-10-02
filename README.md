# AI Sentiment Analysis (End-to-End)

A production-style AI project for **text sentiment analysis** with:
- Modular preprocessing (stopwords removal, lemmatization)
- Model training with TF-IDF + Logistic Regression
- Evaluation with classification report and confusion matrix
- CLI prediction and **FastAPI** serving endpoint
- Config-driven paths and parameters
- Logging and tests

## Project Structure
```
ai-sentiment-analysis-api
├── data/
│   ├── raw/                # raw datasets (csv/json)
│   ├── processed/          # cleaned datasets
│   └── models/             # saved models and reports
├── notebooks/              # experiments (EDA)
├── src/
│   ├── utils/logger.py     # logging helper
│   ├── preprocessing.py    # text cleaning
│   ├── train.py            # training pipeline
│   ├── evaluate.py         # evaluation pipeline
│   ├── predict.py          # CLI predictions
│   └── api.py              # FastAPI app
├── logs/                   # runtime logs
├── tests/                  # unit tests
├── config.yaml             # configuration
├── requirements.txt        # dependencies
└── README.md
```

## Quickstart
1) Install dependencies
```bash
pip install -r requirements.txt
```
2) Train model
```bash
python src/train.py
```
3) Evaluate
```bash
python src/evaluate.py
```
4) Predict (CLI)
```bash
python src/predict.py "I absolutely love this product"
```
5) Serve API
```bash
uvicorn src.api:app --reload
```
Then POST:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text":"Great value!"}'
```

## Notes
- Adjust `config.yaml` to change paths or model parameters.
- Artifacts (model, vectorizer, reports) are saved under `data/models/`.
- Logs are written to `logs/app.log`.

## Author

Ali Zamanpour

Data Engineer & AI Specialist
