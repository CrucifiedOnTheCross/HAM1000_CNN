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
    freeze_backbone: bool = True,       # Флаг для этапа Warmup
    use_augmentation: bool = False      # Флаг для GPU-аугментации (Ускорение на K80)
) -> models.Model:
    """
    Создает модель EfficientNetB0 с поддержкой:
    1. Двухэтапного обучения (freeze_backbone).
    2. Встроенной GPU-аугментации (use_augmentation).
    3. Защиты BatchNormalization слоев.
    """
    
    # 1. Входной слой
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # 2. GPU-Аугментация (Опционально)
    # Эти слои активны ТОЛЬКО во время обучения (model.fit).
    # На валидации и тесте они отключаются автоматически.
    if use_augmentation:
        logger.info("Включена расширенная GPU-аугментация для дерматоскопии.")
        
        # --- Геометрические (безопасные и сильные) ---
        # Поворот на любой угол (фактор 1.0 = 2pi)
        x = layers.RandomRotation(factor=1.0, fill_mode='reflect')(x)
        # Отражение по всем осям
        x = layers.RandomFlip("horizontal_and_vertical")(x)
        # Сдвиг и зум (имитация разного кадрирования и дистанции съемки)
        x = layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='reflect')(x)
        x = layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='reflect')(x)
        
        # --- Цветовые (аккуратные) ---
        # Яркость: имитация разной мощности лампы дерматоскопа
        x = layers.RandomBrightness(factor=0.2)(x)
        # Контраст: разная четкость границ
        x = layers.RandomContrast(factor=0.2)(x)
        
        # --- Продвинутые (для устойчивости) ---
        # Добавляем Гауссов шум (имитация зернистости дешевых камер/телефонов)
        # Это помогает модели не "заучивать" конкретные пиксели
        x = layers.GaussianNoise(stddev=0.05)(x)

    # 3. Базовая модель (Backbone)
    # EfficientNet ожидает пиксели 0-255, препроцессинг встроен внутри
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=x 
    )
    
    # 4. Логика заморозки (Transfer Learning Strategy)
    if freeze_backbone:
        # ЭТАП 1: WARMUP
        # Замораживаем базу полностью. Учим только голову.
        base_model.trainable = False
        logger.info("Режим модели: WARMUP (Backbone заморожен).")
    else:
        # ЭТАП 2: FINE-TUNING
        # Размораживаем базу для тонкой настройки
        base_model.trainable = True
        
        # ВАЖНО: Принудительно замораживаем ВСЕ слои BatchNormalization.
        # Если их обучать на мелких батчах или новых данных, 
        # статистика mean/var испортится, и точность упадет.
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        
        logger.info("Режим модели: FINE-TUNING (Backbone разморожен, BatchNorm заморожен).")

    # 5. Голова классификатора (Head)
    # Используем output базовой модели
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x) # Этот новый BN можно и нужно обучать
    x = layers.Dropout(0.3)(x)         # Dropout для борьбы с переобучением
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    # Сборка модели
    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Skin")

    # 6. Компиляция
    # Импорт фабрик внутри функции, чтобы избежать циклических ссылок
    try:
        from src.metrics import MetricsFactory
        from src.losses import LossFactory
        
        # Метрики
        metrics_list = MetricsFactory.get_all_metrics(num_classes=num_classes)
        
        # Функция потерь
        if use_focal_loss:
            loss_fn = LossFactory.get_focal_loss()
            logger.info("Loss: Focal Loss")
        else:
            loss_fn = LossFactory.get_categorical_crossentropy()
            logger.info("Loss: Categorical Crossentropy")
            
    except ImportError:
        # Fallback, если модули src не найдены (для тестов)
        logger.warning("Модули src.metrics/losses не найдены. Используются стандарты Keras.")
        metrics_list = ['accuracy']
        loss_fn = 'categorical_crossentropy'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=metrics_list
    )
    
    # Лог количества параметров
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    logger.info(f"Модель скомпилирована. Обучаемых параметров: {trainable_count:,}")
    
    return model

if __name__ == "__main__":
    # Простой тест сборки при запуске файла напрямую
    try:
        model = build_efficientnet_b0(freeze_backbone=True, use_augmentation=True)
        print("Тестовая сборка успешна.")
    except Exception as e:
        logger.critical(f"Ошибка сборки: {e}")