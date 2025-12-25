import matplotlib.pyplot as plt
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """
    Класс для визуализации истории обучения нейронных сетей.
    """

    @staticmethod
    def plot_history(history: Dict, save_path: str = "training_plot.png"):
        """
        Строит графики Loss и Accuracy/AUC по эпохам.
        
        :param history: Словарь history.history из объекта обучения Keras.
        :param save_path: Путь для сохранения изображения.
        """
        logger.info("Начало построения графиков обучения...")
        
        loss = history.get('loss', [])
        val_loss = history.get('val_loss', [])
        auc = history.get('auc', [])
        val_auc = history.get('val_auc', [])
        
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(14, 6))

        # График функции потерь
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
        plt.title('Динамика функции потерь (Loss)')
        plt.xlabel('Эпохи')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # График AUC
        plt.subplot(1, 2, 2)
        if auc:
            plt.plot(epochs, auc, 'b-', label='Training AUC')
            plt.plot(epochs, val_auc, 'r--', label='Validation AUC')
            plt.title('Метрика AUC-ROC')
            plt.xlabel('Эпохи')
            plt.ylabel('AUC')
            plt.legend(loc='lower right')
            plt.grid(True)
        else:
            logger.warning("Метрика AUC отсутствует в истории обучения.")

        plt.tight_layout()
        try:
            plt.savefig(save_path, dpi=300)
            logger.info(f"График успешно сохранен по пути: {save_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении графика: {e}")
        finally:
            plt.close()

if __name__ == "__main__":
    # Тестовые данные для проверки
    dummy_history = {
        'loss': [0.9, 0.7, 0.5], 'val_loss': [0.8, 0.6, 0.55],
        'auc': [0.7, 0.8, 0.9], 'val_auc': [0.65, 0.75, 0.85]
    }
    TrainingVisualizer.plot_history(dummy_history)