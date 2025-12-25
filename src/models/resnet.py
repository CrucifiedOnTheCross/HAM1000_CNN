import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_resnet50(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    learning_rate: float = 1e-4,
    use_focal_loss: bool = False
) -> models.Model:
    """
    Создает и компилирует модель на базе ResNet50[cite: 64].
    """
    logger.info("Инициализация сборки модели ResNet50...")
    
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    base_model.trainable = True

    inputs = layers.Input(shape=input_shape)
    # ResNet требует специфичного препроцессинга, если не включен в модель
    x = tf.keras.applications.resnet50.preprocess_input(inputs) 
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x) # Дополнительный полносвязный слой
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="ResNet50_Skin")

    from src.metrics import MetricsFactory
    from src.losses import LossFactory

    if use_focal_loss:
        loss_fn = LossFactory.get_focal_loss()
    else:
        loss_fn = LossFactory.get_categorical_crossentropy()

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
    
    logger.info(f"Модель ResNet50 успешно скомпилирована. Параметров: {model.count_params()}")
    return model

if __name__ == "__main__":
    model = build_resnet50()