import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ROC_filename = f"roc_curve{timestamp}.png"

# Esempio di dataset

y_train = [0, 1, 0, 1, 0, 0, 0, 0]
y_pred = [0, 1, 1, 1, 0, 1, 0, 0]

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_curve, auc
)

def evaluate_model(y_true, y_pred, roc_filename):
    """Valuta il modello e salva la curva ROC.

    Args:
        y_true (array-like): Valori veri.
        y_pred (array-like): Predizioni del modello.
        roc_filename (str): Nome file per salvare la curva ROC.

    Returns:
        dict: Dizionario contenente le metriche calcolate.
    """

    # Calcolo metriche
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)

    # Calcolo ROC curve e AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(roc_filename)
    plt.close()

    # Restituisce le metriche
    return {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'roc_auc': roc_auc
    }

# Esempio di utilizzo
metrics = evaluate_model(y_train, y_pred, ROC_filename)
# Stampa le metriche
print("Metriche del modello:")
for key, value in metrics.items():
    print(f"{key}: {value:.2f}")
# Visualizza la matrice di confusione
cm = confusion_matrix(y_train, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()
# Visualizza le metriche
print("\nMatrice di confusione:")
print(cm)
# Visualizza la curva ROC
print(f"\nCurva ROC salvata come {ROC_filename}")
# Visualizza le metriche