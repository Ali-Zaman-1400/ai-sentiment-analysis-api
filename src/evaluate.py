import os
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from src.preprocessing import batch_clean_text
from src.utils.logger import get_logger

def main(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    logger = get_logger("evaluate", cfg["paths"]["log_file"])

    raw_path = cfg["paths"]["raw_dataset"]
    report_dir = cfg["paths"]["report_dir"]
    model_file = cfg["paths"]["model_file"]
    vec_file = cfg["paths"]["vectorizer_file"]
    os.makedirs(report_dir, exist_ok=True)

    logger.info("Loading artifacts and dataset...")
    model = joblib.load(model_file)
    vec = joblib.load(vec_file)
    df = pd.read_csv(raw_path).dropna(subset=["text", "label"])

    texts = batch_clean_text(df["text"])
    X = vec.transform(texts)
    y_true = df["label"]
    y_pred = model.predict(X)

    # Save classification report
    report_txt = classification_report(y_true, y_pred, digits=4)
    with open(os.path.join(report_dir, "classification_report_full.txt"), "w") as f:
        f.write(report_txt)

    # Confusion matrix
    labels = sorted(y_true.unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    plt.figure()
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    out_png = os.path.join(report_dir, "confusion_matrix.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved reports to {report_dir}")

if __name__ == "__main__":
    main()
