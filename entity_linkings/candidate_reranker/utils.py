from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    scores = p.predictions
    preds = scores.argmax(axis=1).ravel()
    labels = p.label_ids.ravel()
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    num_corrects = (preds == labels).sum().item()
    num_golds = mask.sum().item()
    accuracy = num_corrects / num_golds if num_golds > 0 else float("nan")
    return {"accuracy": accuracy}
