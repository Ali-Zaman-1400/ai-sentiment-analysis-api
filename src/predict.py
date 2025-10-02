import sys
import yaml
import joblib
from src.preprocessing import clean_text

def main(cfg_path: str = "config.yaml"):
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py "Your text here"")
        sys.exit(1)

    text = sys.argv[1]

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    vec = joblib.load(cfg["paths"]["vectorizer_file"])
    model = joblib.load(cfg["paths"]["model_file"])

    clean = clean_text(text)
    X = vec.transform([clean])
    pred = model.predict(X)[0]
    # Predict probability if available
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(max(model.predict_proba(X)[0]))
    print({ "input": text, "prediction": pred, "confidence": proba })

if __name__ == "__main__":
    main()
