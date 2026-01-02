import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_simple_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    learning_rate: float = 1e-4,
    use_focal_loss: bool = False,
    freeze_backbone: bool = False,      # Игнорируется, так как обучаем с нуля
    use_augmentation: bool = False      # Флаг для GPU-аугментации
) -> models.Model:
    """
    Создает ОЧЕНЬ ПРОСТУЮ CNN для быстрой проверки пайплайна обучения.
    Аргументы совпадают с build_efficientnet_b0 для совместимости.
    """
    
    # 1. Входной слой
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # 2. GPU-Аугментация (совместимость с вашим пайплайном)
    if use_augmentation:
        logger.info("SimpleCNN: Включена GPU-аугментация.")
        x = layers.RandomFlip("horizontal_and_vertical")(x)
        x = layers.RandomRotation(0.2)(x)
    
    # 3. Нормализация (Важно! EfficientNet делает это сам, а простая сеть - нет)
    # Приводим пиксели из [0, 255] в [0, 1]
    x = layers.Rescaling(1./255)(x)

    # 4. Архитектура "Simple CNN" (всего ~50k параметров против 5M у EffNet)
    # Блок 1
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Блок 2
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Блок 3
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x) # Сжимаем всё в вектор

    # 5. Обработка флага заморозки
    if freeze_backbone:
        # Для простой сети, которую учим с нуля, заморозка вредна.
        # Просто логируем предупреждение, но не морозим слои.
        logger.warning("SimpleCNN: Флаг freeze_backbone=True проигнорирован (обучение с нуля).")

    # 6. Голова классификатора
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    # Сборка
    model = models.Model(inputs=inputs, outputs=outputs, name="Simple_CNN_Fast")

    # 7. Компиляция (Copy-Paste логики из вашего EfficientNet для совместимости метрик)
    try:
        from src.metrics import MetricsFactory
        from src.losses import LossFactory
        
        metrics_list = MetricsFactory.get_all_metrics(num_classes=num_classes)
        
        if use_focal_loss:
            loss_fn = LossFactory.get_focal_loss()
        else:
            loss_fn = LossFactory.get_categorical_crossentropy()
            
    except ImportError:
        logger.warning("SimpleCNN: Модули src.metrics/losses не найдены. Используются стандарты.")
        metrics_list = ['accuracy']
        loss_fn = 'categorical_crossentropy'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=metrics_list
    )
    
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    logger.info(f"SimpleCNN готова. Параметров: {trainable_count:,} (Очень мало!)")
    
    return model

if __name__ == "__main__":
    # Тест сборки
    model = build_simple_cnn()
    model.summary()