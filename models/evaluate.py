import torch
from scipy.stats import mode
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)


def transformer_evaluate(model, testing_data, le, logger):
    WINDOW_SIZE = 100
    STRIDE = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_by_sliding_window(df, model, ingredient):
        segments = []
        for start in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
            window = df.iloc[start : start + WINDOW_SIZE].values
            segments.append(window)

        X = torch.tensor(segments, dtype=torch.double).to(
            device
        )  # (num_windows, 100, 12)

        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        # Return predictions for each window
        return preds.cpu().numpy(), [ingredient] * len(segments)

    # === Run on test_data ===
    all_preds = []
    all_true_labels = []

    model.eval()
    for ingredient, dfs in testing_data.items():
        for df in dfs:
            df = df.iloc[:512]  # clip to 512 time steps
            preds, labels = predict_by_sliding_window(df, model, ingredient)
            all_preds.extend(preds)
            all_true_labels.extend(labels)

    # === Evaluation ===
    true_labels_encoded = le.transform(all_true_labels)
    accuracy = accuracy_score(true_labels_encoded, all_preds) * 100
    decoded_preds = le.inverse_transform(all_preds)

    logger.info(f"✅ Window-Level Test Accuracy: {accuracy:.2f}%")

    for true, pred in zip(all_true_labels, decoded_preds):
        logger.info(f"True: {true:15s} | Predicted: {pred}")


def regular_evaluate(model, data_loader, le, logger=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, dtype=torch.double)
            labels = labels.to(device)

            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100

    if logger:
        logger.info(f"✅ Regular Evaluation Accuracy: {acc:.2f}%")
        decoded_preds = le.inverse_transform(all_preds)
        decoded_true = le.inverse_transform(all_labels)
        # for true, pred in zip(decoded_true, decoded_preds):
        #     logger.info(f"True: {true:15s} | Predicted: {pred}")
    else:
        print(f"✅ Accuracy: {acc:.2f}%")

    return acc


def evaluate_retrieval(
    test_smell_data,
    test_smell_label,
    gcms_encoder,
    sensor_encoder,
    logger,
    device="cpu",
):
    """
    Evaluate how well the model matches GCMS embeddings to sensor embeddings.
    We'll compute:
      - embeddings for all GCMS data
      - embeddings for all sensor data
    Then for each GCMS embedding, we find the most similar sensor embedding
    and check if it's the correct one (same sample index).

    This returns the "retrieval accuracy" (% of rows i where argmax similarity == i).

    Parameters:
      test_loader: a DataLoader that yields (x_gcms, x_sensor) for test samples.
                   We assume each batch is aligned so sample i in both is the "same" sample.
      gcms_encoder, sensor_encoder: your trained PyTorch encoders
      device: 'cpu' or 'cuda'
    """
    gcms_encoder.eval()
    sensor_encoder.eval()

    # We'll store all embeddings in lists, then concatenate.
    all_z_gcms = []
    all_z_sensor = []

    testing_gcms_data = torch.tensor(gcms_data, dtype=torch.float).to(device)
    gcms_embeddings = gcms_encoder(testing_gcms_data)
    z_gcms = F.normalize(gcms_embeddings, dim=1)

    test_smell_data = torch.tensor(test_smell_data, dtype=torch.float).to(device)
    smell_embeddings = sensor_encoder(test_smell_data)
    z_smell = F.normalize(smell_embeddings, dim=1)

    sim = torch.matmul(z_smell, z_gcms.T)

    logger.info(f"Similarity matrix shape: {sim.shape}")

    # For each row i, find the column j with the highest similarity
    # If j == i, it means we matched the correct sensor embedding
    predicted = sim.argmax(dim=1)  # [N]

    logger.info("------------------Predictions---------------------")
    logger.info(predicted)

    # Compare with the "ground truth" index = i
    correct = predicted == test_smell_label
    accuracy = correct.float().mean().item()

    precision = precision_score(test_smell_label, predicted, average="macro")
    recall = recall_score(test_smell_label, predicted, average="macro")
    f1 = f1_score(test_smell_label, predicted, average="macro")
    conf_matrix = confusion_matrix(test_smell_label, predicted)

    logger.info("------------------Test Statistics---------------------")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(conf_matrix)

    return accuracy, conf_matrix


def analyze_confusion_matrix(conf_matrix):
    num_classes = conf_matrix.shape[0]
    class_metrics = {}

    # Calculate metrics for each class
    for i in range(num_classes):
        # True Positives (TP): Correct predictions for class i
        num_predictions = np.sum(conf_matrix[i])
        TP = conf_matrix[i, i]

        # False Positives (FP): Sum of column i (excluding TP)
        FP = np.sum(conf_matrix[:, i]) - TP

        # False Negatives (FN): Sum of row i (excluding TP)
        FN = np.sum(conf_matrix[i, :]) - TP

        # True Negatives (TN): Sum of all elements except row i and column i
        TN = np.sum(conf_matrix) - (TP + FP + FN)

        # Calculate metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        precision = TP / num_predictions

        # Store metrics for the class
        class_metrics[available_food_names[i]] = {
            "Accuracy": precision,
        }

    return class_metrics
