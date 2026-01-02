import tensorflow as tf
import logging
from typing import List, Union, Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================================================================================
# CUSTOM METRIC CLASSES
# ==================================================================================

@tf.keras.utils.register_keras_serializable(package="CustomMetrics")
class MacroF1Score(tf.keras.metrics.Metric):
    """
    Реализация метрики F1-Score с усреднением Macro-Average.
    Полезна для оценки качества на несбалансированных данных (как HAM10000).
    """
    def __init__(self, num_classes: int, name: str = 'f1_macro', **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
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
        config = super(MacroF1Score, self).get_config()
        config.update({"num_classes": self.num_classes})
        return config


@tf.keras.utils.register_keras_serializable(package="CustomMetrics")
class MCCMetric(tf.keras.metrics.Metric):
    """
    Реализация Matthews Correlation Coefficient (MCC).
    Считается одной из лучших метрик для бинарной и мультиклассовой классификации 
    при дисбалансе классов.
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
        config = super(MCCMetric, self).get_config()
        config.update({"num_classes": self.num_classes})
        return config


@tf.keras.utils.register_keras_serializable(package="CustomMetrics")
class SpecificityMetric(tf.keras.metrics.Metric):
    """
    Реализация Specificity (True Negative Rate) для конкретного класса (One-vs-Rest).
    Specificity = TN / (TN + FP)
    Показывает способность модели НЕ присваивать диагноз здоровому пациенту (или пациенту с другим диагнозом).
    """
    def __init__(self, class_id: int, num_classes: int, name: str = None, **kwargs):
        # Если имя не задано, формируем его автоматически
        name = name if name else f'specificity_class_{class_id}'
        super(SpecificityMetric, self).__init__(name=name, **kwargs)
        self.class_id = class_id
        self.num_classes = num_classes
        # Инициализация счетчиков
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_idx = tf.argmax(y_pred, axis=-1)
        y_true_idx = tf.argmax(y_true, axis=-1)

        # Логические маски для текущего класса
        is_class_true = tf.equal(y_true_idx, self.class_id)
        is_class_pred = tf.equal(y_pred_idx, self.class_id)

        # True Negative: Реально НЕ этот класс и Предсказано НЕ этот класс
        tn = tf.logical_and(tf.logical_not(is_class_true), tf.logical_not(is_class_pred))
        
        # False Positive: Реально НЕ этот класс, но Предсказано, что ЭТОТ
        fp = tf.logical_and(tf.logical_not(is_class_true), is_class_pred)

        # Приводим к float и суммируем
        if sample_weight is not None:
             sample_weight = tf.cast(sample_weight, tf.float32)
             self.tn.assign_add(tf.reduce_sum(tf.cast(tn, tf.float32) * sample_weight))
             self.fp.assign_add(tf.reduce_sum(tf.cast(fp, tf.float32) * sample_weight))
        else:
             self.tn.assign_add(tf.reduce_sum(tf.cast(tn, tf.float32)))
             self.fp.assign_add(tf.reduce_sum(tf.cast(fp, tf.float32)))

    def result(self):
        # Specificity = TN / (TN + FP)
        return tf.math.divide_no_nan(self.tn, self.tn + self.fp)

    def reset_state(self):
        self.tn.assign(0.0)
        self.fp.assign(0.0)

    def get_config(self):
        config = super(SpecificityMetric, self).get_config()
        config.update({"class_id": self.class_id, "num_classes": self.num_classes})
        return config


# ==================================================================================
# METRICS FACTORY
# ==================================================================================

class MetricsFactory:
    """
    Фабрика метрик для набора данных HAM10000.
    Генерирует стандартные метрики, агрегированные метрики и 
    детальные метрики (Sens/Spec/Prec) для КАЖДОГО класса.
    """
    
    # Стандартный маппинг HAM10000 (в алфавитном порядке)
    # Если ваш Generator использует другой порядок, измените этот словарь.
    HAM10000_CLASSES: Dict[int, str] = {
        0: 'akiec', # Actinic keratoses (Pre-cancerous) -> ОПАСНЫЙ
        1: 'bcc',   # Basal cell carcinoma -> ОПАСНЫЙ
        2: 'bkl',   # Benign keratosis-like lesions
        3: 'df',    # Dermatofibroma
        4: 'mel',   # Melanoma -> ОПАСНЫЙ
        5: 'nv',    # Melanocytic nevi (Common mole)
        6: 'vasc'   # Vascular lesions
    }

    @staticmethod
    def get_all_metrics(num_classes: int = 7) -> List[Union[tf.keras.metrics.Metric, str]]:
        
        # 1. Глобальные агрегированные метрики
        metrics = [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_acc'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_acc'),
            MacroF1Score(num_classes=num_classes),
            MCCMetric(num_classes=num_classes),
            tf.keras.metrics.AUC(name='auc_roc', multi_label=True, curve='ROC'),
            tf.keras.metrics.AUC(name='auc_pr', multi_label=True, curve='PR'), # Precision-Recall curve
        ]

        # 2. Детализация по каждому классу (Per-class metrics)
        # Генерируем Sensitivity (Recall), Specificity и Precision для каждого класса
        for class_id, class_name in MetricsFactory.HAM10000_CLASSES.items():
            if class_id >= num_classes:
                break
                
            # Sensitivity / Recall (Полнота) - критична для mel, bcc, akiec
            metrics.append(
                tf.keras.metrics.Recall(
                    class_id=class_id, 
                    name=f'sens_{class_name}' # ex: sens_mel, sens_bcc
                )
            )
            
            # Specificity (Специфичность) - контроль ложных тревог
            metrics.append(
                SpecificityMetric(
                    class_id=class_id, 
                    num_classes=num_classes, 
                    name=f'spec_{class_name}' # ex: spec_mel, spec_bcc
                )
            )
            
            # Precision (Точность)
            metrics.append(
                tf.keras.metrics.Precision(
                    class_id=class_id, 
                    name=f'prec_{class_name}' # ex: prec_mel
                )
            )

        return metrics

if __name__ == "__main__":
    logger.info("Модуль метрик инициализирован.")
    
    # Пример проверки создания
    metrics_list = MetricsFactory.get_all_metrics()
    logger.info(f"Создано {len(metrics_list)} метрик.")
    
    # Вывод имен метрик для проверки
    metric_names = []
    for m in metrics_list:
        if isinstance(m, str):
            metric_names.append(m)
        else:
            metric_names.append(m.name)
    
    logger.info(f"Список метрик: {metric_names}")