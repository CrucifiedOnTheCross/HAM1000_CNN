import json
import logging
import numpy as np
import tensorflow as tf
from typing import Any

# Настройка логирования на уровне модуля не требуется, если она выполнена в main,
# но для автономного тестирования оставим базовую конфигурацию в блоке main.

class TFJSONEncoder(json.JSONEncoder):
    """
    Класс для сериализации объектов TensorFlow и NumPy в формат JSON.
    Наследуется от стандартного json.JSONEncoder.
    """
    
    def default(self, obj: Any) -> Any:
        """
        Переопределение метода конвертации типов.
        """
        # Обработка тензоров TensorFlow
        if isinstance(obj, (tf.Tensor, tf.Variable)):
            # Если скаляр (одиночное число)
            if hasattr(obj, 'shape') and (obj.shape.ndims == 0 or obj.shape == ()):
                return float(obj.numpy())
            # Если массив
            if hasattr(obj, 'numpy'):
                return obj.numpy().tolist()

        # Обработка типов NumPy
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)

if __name__ == "__main__":
    # Блок для тестирования модуля
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        data = tf.constant([1.5, 2.5, 3.5], dtype=tf.float32)
        encoded_data = json.dumps(data, cls=TFJSONEncoder)
        logging.info(f"Тестовая сериализация выполнена успешно: {encoded_data}")
    except Exception as e:
        logging.critical(f"Ошибка при тестировании сериализатора: {e}")