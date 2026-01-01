import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_resnet50(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    learning_rate: float = 1e-4,
    use_focal_loss: bool = False
) -> models.Model:
    """
    Создает и компилирует модель на базе ResNet50.
    Включает слой препроцессинга внутри графа модели.
    """
    logger.info("Инициализация сборки модели ResNet50...")
    
    # 1. Входной слой
    inputs = layers.Input(shape=input_shape)

    # 2. Слой препроцессинга (ResNet ожидает данные не в 0-1, а специфично центрированные)
    # Используем Lambda слой, чтобы логика была частью сохраненной модели
    x = layers.Lambda(tf.keras.applications.resnet50.preprocess_input, name='resnet_preprocess')(inputs)
    
    # 3. Базовая модель
    # Важно: не передаем input_tensor=inputs в конструктор, так как мы подаем уже обработанный x
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    base_model.trainable = True

    # Пропускаем препроцессированные данные через базу
    x = base_model(x)

    # 4. Голова классификатора (Head)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x) # Добавлено для стабильности (как в EfficientNet)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50_Skin")

    # 5. Импорт зависимостей
    from src.metrics import MetricsFactory
    from src.losses import LossFactory

    # 6. Выбор функции потерь
    if use_focal_loss:
        loss_fn = LossFactory.get_focal_loss()
    else:
        loss_fn = LossFactory.get_categorical_crossentropy()

    # 7. Компиляция с полным набором метрик
    metrics_list = MetricsFactory.get_all_metrics(num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=metrics_list
    )
    
    logger.info(f"Модель ResNet50 успешно скомпилирована. Параметров: {model.count_params()}")
    return model

if __name__ == "__main__":
    # Тест сборки
    try:
        model = build_resnet50()
        logger.info("Тестовая сборка ResNet50 прошла успешно.")
    except Exception as e:
        logger.critical(f"Ошибка сборки модели: {e}")