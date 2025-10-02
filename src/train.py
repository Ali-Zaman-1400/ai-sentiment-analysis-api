import os
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.preprocessing import batch_clean_text
from src.utils.logger import get_logger

def main(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("train", cfg["paths"]["log_file"])

    raw_path = cfg["paths"]["raw_dataset"]
    processed_path = cfg["paths"]["processed_dataset"]
    model_dir = cfg["paths"]["model_dir"]
    model_file = cfg["paths"]["model_file"]
    vec_file = cfg["paths"]["vectorizer_file"]
    report_dir = cfg["paths"]["report_dir"]
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    logger.info(f"Loading dataset from {raw_path}")
    df = pd.read_csv(raw_path)
    df = df.dropna(subset=["text", "label"])

    logger.info("Cleaning text...")
    df["clean_text"] = batch_clean_text(
        df["text"],
        lowercase=cfg["preprocessing"]["lowercase"],
        remove_punct=cfg["preprocessing"]["remove_punct"],
        remove_stopwords=cfg["preprocessing"]["remove_stopwords"],
        lemmatize=cfg["preprocessing"]["lemmatize"],
    )
    df.to_csv(processed_path, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    logger.info("Vectorizing with TF-IDF...")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    logger.info("Training LogisticRegression...")
    model = LogisticRegression(C=float(cfg["model"]["params"]["C"]),
                               max_iter=int(cfg["model"]["params"]["max_iter"]))
    model.fit(X_train_vec, y_train)

    logger.info("Evaluating...")
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, digits=4, output_dict=False)
    logger.info("\n" + report)

    with open(os.path.join(report_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    logger.info(f"Saving vectorizer to {vec_file} and model to {model_file}")
    joblib.dump(vec, vec_file)
    joblib.dump(model, model_file)

    logger.info("Training complete.")

if __name__ == "__main__":
    main()
