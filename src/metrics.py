import tensorflow as tf
import logging
from typing import List, Union

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@tf.keras.utils.register_keras_serializable(package="CustomMetrics")
class MacroF1Score(tf.keras.metrics.Metric):
    """
    Реализация метрики F1-Score с усреднением Macro-Average.
    """
    def __init__(self, num_classes: int, name: str = 'f1_macro', **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Инициализация переменных состояния
        self.tp = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_idx = tf.argmax(y_pred, axis=-1)
        y_true_idx = tf.argmax(y_true, axis=-1)

        y_true_oh = tf.one_hot(y_true_idx, self.num_classes)
        y_pred_oh = tf.one_hot(y_pred_idx, self.num_classes)

        y_true_oh = tf.cast(y_true_oh, tf.float32)
        y_pred_oh = tf.cast(y_pred_oh, tf.float32)

        tp = tf.reduce_sum(y_true_oh * y_pred_oh, axis=0)
        fp = tf.reduce_sum((1 - y_true_oh) * y_pred_oh, axis=0)
        fn = tf.reduce_sum(y_true_oh * (1 - y_pred_oh), axis=0)

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        return tf.reduce_mean(f1)

    def reset_state(self):
        self.tp.assign(tf.zeros(self.num_classes))
        self.fp.assign(tf.zeros(self.num_classes))
        self.fn.assign(tf.zeros(self.num_classes))

    def get_config(self):
        """Необходим для корректного сохранения модели в H5/JSON"""
        config = super(MacroF1Score, self).get_config()
        config.update({"num_classes": self.num_classes})
        return config


@tf.keras.utils.register_keras_serializable(package="CustomMetrics")
class MCCMetric(tf.keras.metrics.Metric):
    """
    Реализация Matthews Correlation Coefficient (MCC).
    """
    def __init__(self, num_classes: int, name: str = 'mcc', **kwargs):
        super(MCCMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name='confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_idx = tf.argmax(y_pred, axis=-1)
        y_true_idx = tf.argmax(y_true, axis=-1)
        
        current_cm = tf.math.confusion_matrix(
            y_true_idx, 
            y_pred_idx, 
            num_classes=self.num_classes, 
            dtype=tf.float32
        )
        self.confusion_matrix.assign_add(current_cm)

    def result(self):
        cm = self.confusion_matrix
        c = tf.reduce_sum(tf.linalg.tensor_diag_part(cm))
        s = tf.reduce_sum(cm)
        pk = tf.reduce_sum(cm, axis=1)
        tk = tf.reduce_sum(cm, axis=0)
        
        numerator = c * s - tf.reduce_sum(pk * tk)
        term1 = s**2 - tf.reduce_sum(tf.square(pk))
        term2 = s**2 - tf.reduce_sum(tf.square(tk))
        denominator = tf.sqrt(term1 * term2)
        
        return tf.math.divide_no_nan(numerator, denominator)

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        """Необходим для корректного сохранения модели в H5/JSON"""
        config = super(MCCMetric, self).get_config()
        config.update({"num_classes": self.num_classes})
        return config


class MetricsFactory:
    """
    Фабрика метрик.
    """
    @staticmethod
    def get_all_metrics(num_classes: int = 7) -> List[Union[tf.keras.metrics.Metric, str]]:
        return [
            "accuracy",
            MacroF1Score(num_classes=num_classes),
            MCCMetric(num_classes=num_classes),
            tf.keras.metrics.AUC(name='auc', multi_label=True),
            tf.keras.metrics.Recall(class_id=1, name='sensitivity_mel'),
            tf.keras.metrics.Precision(class_id=1, name='precision_mel')
        ]

if __name__ == "__main__":
    logger.info("Модуль метрик готов к использованию.")