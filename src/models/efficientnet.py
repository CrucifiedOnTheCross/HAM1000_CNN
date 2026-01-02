import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_efficientnet_b0(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    learning_rate: float = 1e-4,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,           # <--- НОВЫЙ АРГУМЕНТ
    focal_alpha: float = 0.25,          # <--- НОВЫЙ АРГУМЕНТ
    freeze_backbone: bool = True,       
    use_augmentation: bool = False      
) -> models.Model:
    """
    Создает модель EfficientNetB0 с поддержкой настройки Focal Loss.
    """
    
    # 1. Входной слой
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # 2. GPU-Аугментация
    if use_augmentation:
        logger.info("Включена расширенная GPU-аугментация для дерматоскопии.")
        x = layers.RandomRotation(factor=1.0, fill_mode='reflect')(x)
        x = layers.RandomFlip("horizontal_and_vertical")(x)
        x = layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='reflect')(x)
        x = layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='reflect')(x)
        x = layers.RandomBrightness(factor=0.2)(x)
        x = layers.RandomContrast(factor=0.2)(x)
        x = layers.GaussianNoise(stddev=0.05)(x)

    # 3. Базовая модель (Backbone)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=x 
    )
    
    # 4. Логика заморозки
    if freeze_backbone:
        base_model.trainable = False
        logger.info("Режим модели: WARMUP (Backbone заморожен).")
    else:
        base_model.trainable = True
        # Принудительно замораживаем BatchNorm
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        logger.info("Режим модели: FINE-TUNING (Backbone разморожен, BatchNorm заморожен).")

    # 5. Голова классификатора
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    # Сборка
    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Skin")

    # 6. Компиляция
    try:
        from src.metrics import MetricsFactory
        from src.losses import LossFactory
        
        metrics_list = MetricsFactory.get_all_metrics(num_classes=num_classes)
        
        # Функция потерь с параметрами
        if use_focal_loss:
            loss_fn = LossFactory.get_focal_loss(gamma=focal_gamma, alpha=focal_alpha)
            logger.info(f"Loss: Focal Loss (gamma={focal_gamma}, alpha={focal_alpha})")
        else:
            loss_fn = LossFactory.get_categorical_crossentropy()
            logger.info("Loss: Categorical Crossentropy")
            
    except ImportError:
        logger.warning("Модули src.metrics/losses не найдены. Используются стандарты Keras.")
        metrics_list = ['accuracy']
        loss_fn = 'categorical_crossentropy'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=metrics_list
    )
    
    return model

if __name__ == "__main__":
    try:
        model = build_efficientnet_b0(freeze_backbone=True, use_augmentation=True)
        print("Тестовая сборка успешна.")
    except Exception as e:
        logger.critical(f"Ошибка сборки: {e}")