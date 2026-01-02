import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import pandas as pd
from typing import Dict, List
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """
    Класс для визуализации истории обучения и метрик нейронных сетей.
    Адаптирован для отображения расширенного набора метрик (AUC-PR, F1, Sensitivity, Specificity).
    """

    @staticmethod
    def plot_history(history: Dict, save_path: str = "training_plot.png"):
        """
        Строит графики Loss и ключевых метрик.
        Автоматически находит метрики в истории и строит для них графики.
        """
        logger.info("Построение графиков истории обучения...")
        
        loss = history.get('loss', [])
        val_loss = history.get('val_loss', [])
        epochs = range(1, len(loss) + 1)

        # Список ключевых метрик, которые мы хотим видеть на графиках
        # Порядок важен для красоты
        metrics_to_plot = ['auc_pr', 'f1_macro', 'sens_mel', 'spec_mel', 'accuracy']
        
        # Фильтруем те, что реально есть в истории
        available_metrics = [m for m in metrics_to_plot if m in history]

        # Настраиваем сетку графиков
        # 1 график для Loss + по одному для каждой найденной метрики
        num_plots = 1 + len(available_metrics)
        rows = (num_plots + 1) // 2 # Округляем вверх для 2 колонок
        
        plt.figure(figsize=(15, 5 * rows))

        # 1. График Loss (всегда первый)
        plt.subplot(rows, 2, 1)
        plt.plot(epochs, loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
        plt.title('Loss Function')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Графики остальных метрик
        for i, metric in enumerate(available_metrics):
            val_metric = f"val_{metric}"
            
            if metric in history and val_metric in history:
                plt.subplot(rows, 2, i + 2)
                plt.plot(epochs, history[metric], 'b-', label=f'Train {metric}')
                plt.plot(epochs, history[val_metric], 'r--', label=f'Val {metric}')
                plt.title(f'Metric: {metric}')
                plt.xlabel('Epochs')
                plt.ylabel('Score')
                plt.legend(loc='best') # 'best' сам найдет свободное место
                plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], save_path: str = "confusion_matrix.png"):
        """
        Строит и сохраняет НОРМАЛИЗОВАННУЮ матрицу ошибок.
        Показывает, как часто истинный класс i был предсказан как j.
        """
        logger.info("Построение нормализованной матрицы ошибок...")
        
        cm = confusion_matrix(y_true, y_pred)
        # Нормализация (Recall view: сумма по строке = 1)
        # Добавляем epsilon, чтобы не делить на 0, если класс отсутствует в батче
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, vmin=0, vmax=1)
        
        plt.title('Normalized Confusion Matrix (Recall)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()

    @staticmethod
    def plot_roc_curves(y_true: np.ndarray, y_pred_probs: np.ndarray, classes: List[str], save_path: str = "roc_curves.png"):
        """Строит ROC-кривые для каждого класса (One-vs-Rest)."""
        logger.info("Построение ROC-кривых...")
        n_classes = len(classes)
        # Бинаризация меток для OvR
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan'])
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=100)
        plt.close()

    @staticmethod
    def plot_pr_curves(y_true: np.ndarray, y_pred_probs: np.ndarray, classes: List[str], save_path: str = "pr_curves.png"):
        """
        [НОВОЕ] Строит Precision-Recall кривые.
        Это КРИТИЧЕСКИ ВАЖНО для несбалансированных данных (как HAM10000).
        """
        logger.info("Построение Precision-Recall кривых...")
        n_classes = len(classes)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
            average_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])

        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan'])
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                     label=f'{classes[i]} (AP = {average_precision[i]:.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=100)
        plt.close()