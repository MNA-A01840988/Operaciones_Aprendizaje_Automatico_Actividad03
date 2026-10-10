#Imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X, y):
    # Classification report generated to get more information about the model performance
    y_pred = model.predict(X)

    model_metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, pos_label="Yes"),
        "recall": recall_score(y, y_pred, pos_label="Yes"),
        "f1_score": f1_score(y, y_pred, pos_label="Yes")
    }

    return model_metrics