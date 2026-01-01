import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import pandas as pd
from typing import Dict, List
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """
    Класс для визуализации истории обучения и метрик нейронных сетей.
    """

    @staticmethod
    def plot_history(history: Dict, save_path: str = "training_plot.png"):
        """Строит графики Loss и Accuracy/AUC по эпохам."""
        logger.info("Построение графиков истории обучения...")
        
        loss = history.get('loss', [])
        val_loss = history.get('val_loss', [])
        
        if 'auc' in history:
            metric_name = 'AUC'
            metric = history['auc']
            val_metric = history['val_auc']
        elif 'accuracy' in history:
            metric_name = 'Accuracy'
            metric = history['accuracy']
            val_metric = history['val_accuracy']
        else:
            metric_name = None

        epochs = range(1, len(loss) + 1)
        plt.figure(figsize=(14, 6))

        # 1. График Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
        plt.title('Динамика функции потерь (Loss)')
        plt.xlabel('Эпохи')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. График Метрики
        if metric_name:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, metric, 'b-', label=f'Training {metric_name}')
            plt.plot(epochs, val_metric, 'r--', label=f'Validation {metric_name}')
            plt.title(f'Динамика {metric_name}')
            plt.xlabel('Эпохи')
            plt.ylabel(metric_name)
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        # ИЗМЕНЕНИЕ: dpi=100 для нормального отображения на экране
        plt.savefig(save_path, dpi=100)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], save_path: str = "confusion_matrix.png"):
        """
        Строит и сохраняет НОРМАЛИЗОВАННУЮ матрицу ошибок.
        """
        logger.info("Построение нормализованной матрицы ошибок...")
        
        # Считаем обычную матрицу
        cm = confusion_matrix(y_true, y_pred)
        
        # ИЗМЕНЕНИЕ: Нормализация (делим на сумму по строке + epsilon чтобы не делить на 0)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        plt.figure(figsize=(10, 8))
        
        # ИЗМЕНЕНИЕ: fmt='.2f' для отображения долей (0.95), annot=True выводит числа
        # Если хотите проценты (95%), можно умножить cm_normalized на 100 перед отрисовкой
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, vmin=0, vmax=1)
        
        plt.title('Нормализованная матрица ошибок (Recall)')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.tight_layout()
        
        # ИЗМЕНЕНИЕ: dpi=100
        plt.savefig(save_path, dpi=100)
        plt.close()

    @staticmethod
    def plot_roc_curves(y_true: np.ndarray, y_pred_probs: np.ndarray, classes: List[str], save_path: str = "roc_curves.png"):
        """Строит ROC-кривые для каждого класса."""
        logger.info("Построение ROC-кривых...")
        n_classes = len(classes)
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
        plt.title('ROC-кривые по классам')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # ИЗМЕНЕНИЕ: dpi=100
        plt.savefig(save_path, dpi=100)
        plt.close()

    @staticmethod
    def plot_class_distribution(df: pd.DataFrame, class_col: str, save_path: str = "class_distribution.png"):
        """Строит гистограмму распределения классов."""
        logger.info("Построение графика распределения данных...")
        plt.figure(figsize=(10, 6))
        sns.countplot(x=class_col, data=df, palette='viridis', order=df[class_col].value_counts().index)
        plt.title('Распределение классов в датасете')
        plt.xlabel('Класс')
        plt.ylabel('Количество изображений')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # ИЗМЕНЕНИЕ: dpi=100
        plt.savefig(save_path, dpi=100)
        plt.close()