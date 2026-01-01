import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_efficientnet_b0(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    learning_rate: float = 1e-4,
    use_focal_loss: bool = False
) -> models.Model:
    """
    Создает и компилирует модель на базе EfficientNetB0.
    """
    logger.info("Инициализация сборки модели EfficientNetB0...")
    
    # 1. Входной слой
    inputs = layers.Input(shape=input_shape)
    
    # 2. Базовая модель (EfficientNet имеет встроенный препроцессинг для 0-255 inputs)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    
    # Разморозка весов для Fine-tuning
    base_model.trainable = True 

    # 3. Голова классификатора (Head)
    # Используем выход base_model.output, так как base_model построена на inputs
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Skin")

    # 4. Импорт зависимостей (внутри функции во избежание циклических импортов)
    from src.metrics import MetricsFactory
    from src.losses import LossFactory

    # 5. Настройка функции потерь
    if use_focal_loss:
        loss_fn = LossFactory.get_focal_loss()
    else:
        loss_fn = LossFactory.get_categorical_crossentropy()

    # 6. Компиляция с полным набором метрик
    # MetricsFactory.get_all_metrics возвращает список: [Acc, F1, MCC, AUC, Sensitivity]
    metrics_list = MetricsFactory.get_all_metrics(num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=metrics_list
    )
    
    logger.info(f"Модель EfficientNetB0 успешно скомпилирована. Параметров: {model.count_params()}")
    return model

if __name__ == "__main__":
    # Тест сборки
    try:
        model = build_efficientnet_b0()
        logger.info("Тестовая сборка EfficientNet прошла успешно.")
    except Exception as e:
        logger.critical(f"Ошибка сборки модели: {e}")