import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LossFactory:
    """
    Предоставляет функции потерь для обучения нейросетей.
    Включает стандартную CrossEntropy и Focal Loss для борьбы с дисбалансом.
    """

    @staticmethod
    def get_categorical_crossentropy() -> tf.keras.losses.Loss:
        """Возвращает стандартную категориальную кросс-энтропию."""
        logger.info("Выбрана функция потерь: Categorical Crossentropy.")
        return tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    @staticmethod
    def get_focal_loss(gamma: float = 2.0, alpha: float = 0.25) -> callable:
        """
        Реализует Focal Loss: -alpha_t * (1 - p_t)^gamma * log(p_t).
        Снижает влияние простых примеров (NV) и фокусируется на сложных (MEL, DF).
        
        :param gamma: Параметр фокусировки (стандарт: 2.0).
        :param alpha: Весовой коэффициент балансировки (стандарт: 0.25).
        """
        def focal_loss_fixed(y_true, y_pred):
            # Клиппинг значений во избежание log(0)
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

            # Расчет кросс-энтропии
            cross_entropy = -y_true * tf.math.log(y_pred)
            
            # Расчет весового множителя
            weight = alpha * y_true * tf.math.pow((1 - y_pred), gamma)
            
            # Итоговый лосс
            loss = weight * cross_entropy
            return tf.math.reduce_sum(loss, axis=1)

        logger.info(f"Инициализирована функция Focal Loss (gamma={gamma}, alpha={alpha}).")
        return focal_loss_fixed

if __name__ == "__main__":
    logger.info("Модуль функций потерь готов к работе.")