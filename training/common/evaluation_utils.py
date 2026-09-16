import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_model(model, data_loader, class_names, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in data_loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(videos)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_accuracy = {}
    for i, name in enumerate(class_names):
        total = int(cm[i].sum())
        per_class_accuracy[name] = (
            float(cm[i, i] / total) if total else 0.0
        )

    return {
        "accuracy": accuracy,
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_accuracy,
        "classification_report": report,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "class_names": list(class_names),
    }


def print_evaluation_results(results, model_name, class_names):
    print("\n" + "=" * 60)
    print(f"{model_name} - TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {results['accuracy'] * 100:.2f}%")
    print(f"Precision: {results['precision_weighted']:.4f}")
    print(f"Recall   : {results['recall_weighted']:.4f}")
    print(f"F1       : {results['f1_weighted']:.4f}")

    print("\nPer-class accuracy:")
    for name in class_names:
        print(
            f"  {name:15s}: "
            f"{results['per_class_accuracy'][name] * 100:.2f}%"
        )

    print("\nConfusion matrix:")
    for row in results["confusion_matrix"]:
        print(row)


def save_results(results, prefix, output_dir="results"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_path / f"{prefix}_results_{timestamp}.json"

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {file_path}")
    return str(file_path)
