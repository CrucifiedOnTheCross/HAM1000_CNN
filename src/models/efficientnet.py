import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, Optional
import logging

# Настройка логирования
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
    
    # Явно создаем входной слой, чтобы зафиксировать shape
    inputs = layers.Input(shape=input_shape)
    
    # Инициализируем базу, передавая input_tensor
    # Это самый надежный способ избежать shape mismatch
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    
    # Разморозка (или частичная разморозка)
    base_model.trainable = True 

    # Построение головы
    # Примечание: base_model(inputs) уже не нужно вызывать отдельно, 
    # т.к. base_model построен на inputs. Берем его выход.
    x = base_model.output
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Skin")

    # Импорт зависимостей
    from src.metrics import MetricsFactory
    from src.losses import LossFactory

    # Выбор функции потерь
    if use_focal_loss:
        loss_fn = LossFactory.get_focal_loss()
    else:
        loss_fn = LossFactory.get_categorical_crossentropy()

    # Компиляция
    mel_index = 1 
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            'accuracy',
            MetricsFactory.get_auc_roc(),
            MetricsFactory.get_sensitivity_for_class(mel_index)
        ]
    )
    
    logger.info(f"Модель EfficientNetB0 успешно скомпилирована. Параметров: {model.count_params()}")
    return model