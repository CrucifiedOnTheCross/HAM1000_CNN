import os
import multiprocessing

# Убираем лишний шум от TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import logging
import warnings
import argparse
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Отключение предупреждений
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.WARNING)

# Импорт локальных модулей
from src.data_loader import DataManager, BalancedDataGenerator
from src.models.efficientnet import build_efficientnet_b0
from src.models.resnet import build_resnet50
from src.visualization import TrainingVisualizer
from src.serializers import TFJSONEncoder

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_standard_generator(df: pd.DataFrame, input_shape: tuple, batch_size: int, class_order: list, augment: bool, is_training: bool) -> tf.keras.preprocessing.image.Iterator:
    """Создает стандартный ImageDataGenerator."""
    # Используем CPU аугментацию (быстрее на K80, чем GPU layers)
    if is_training and augment:
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
        logger.info("Создан генератор: СТАНДАРТНЫЙ + АУГМЕНТАЦИЯ (CPU).")
    else:
        datagen = ImageDataGenerator()
        if is_training:
            logger.info("Создан генератор: СТАНДАРТНЫЙ (БЕЗ АУГМЕНТАЦИИ).")

    return datagen.flow_from_dataframe(
        dataframe=df, x_col='path', y_col='dx', target_size=input_shape,
        class_mode='categorical', classes=class_order, batch_size=batch_size, shuffle=is_training
    )

def run_experiment(args: argparse.Namespace) -> None:
    logger.info(f"Старт: {args.model} | Bal={not args.no_balance} | Aug={not args.no_augment} | Focal={args.focal}")

    # --- КОНФИГУРАЦИЯ ОБОРУДОВАНИЯ ---
    # Определяем количество ядер для параллельной загрузки данных
    n_workers = multiprocessing.cpu_count()
    logger.info(f"Доступно ядер CPU для загрузки данных (workers): {n_workers}")

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Используется {strategy.num_replicas_in_sync} GPU (режим MirroredStrategy).")
    else:
        strategy = tf.distribute.get_strategy()

    # 1. Подготовка данных
    data_manager = DataManager(data_dir=args.data_dir)
    class_order = list(data_manager.class_map.keys())
    try:
        train_df, val_df, test_df = data_manager.prepare_data(test_size=0.2)
    except Exception as e:
        logger.critical(f"Ошибка данных: {e}"); return

    input_shape = (224, 224)
    batch_size = args.batch_size

    # 2. Создание генераторов
    # Включаем augment=True здесь, так как вернулись к CPU аугментации
    if args.no_balance:
        train_gen = create_standard_generator(
            train_df, input_shape, batch_size, class_order, 
            augment=not args.no_augment, # <--- ВКЛЮЧЕНО (CPU)
            is_training=True
        )
    else:
        logger.info("Инициализация BalancedDataGenerator (с кэшированием в RAM)...")
        train_gen = BalancedDataGenerator(
            train_df, 
            batch_size=batch_size, 
            input_shape=input_shape, 
            shuffle=True, 
            augment=not args.no_augment, # <--- ВКЛЮЧЕНО (CPU)
            cache_images=True
        )

    val_gen = create_standard_generator(val_df, input_shape, batch_size, class_order, False, False)
    test_gen = create_standard_generator(test_df, input_shape, batch_size, class_order, False, False)

    # 3. Настройка путей
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.model}_{'imbalanced' if args.no_balance else 'balanced'}_{'noaug' if args.no_augment else 'aug'}_{'focal' if args.focal else 'ce'}_{timestamp}"
    ckpt_dir = os.path.join("experiments", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    test_df.to_csv(os.path.join(ckpt_dir, "test_dataset.csv"), index=False)

    # ==========================================
    # 4. ДВУХЭТАПНОЕ ОБУЧЕНИЕ
    # ==========================================
    with strategy.scope():
        # --- ФАЗА 1: WARMUP ---
        logger.info("\n>>> ФАЗА 1: WARMUP (Обучение классификатора, база заморожена)...")
        
        if args.model == 'efficientnet':
            model = build_efficientnet_b0(
                num_classes=7, 
                learning_rate=args.lr, 
                use_focal_loss=args.focal, 
                freeze_backbone=True,
                use_augmentation=False # Отключили медленную GPU аугментацию
            )
        else:
            model = build_resnet50(num_classes=7, learning_rate=args.lr, use_focal_loss=args.focal)

        warmup_epochs = 3
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=warmup_epochs,
            verbose=1,
            workers=n_workers,          # <--- ИСПОЛЬЗУЕМ ВСЕ ЯДРА
            use_multiprocessing=True, 
            max_queue_size=10,
            callbacks=[CSVLogger(os.path.join(ckpt_dir, "log_warmup.csv"), append=True)]
        )

        # --- ФАЗА 2: FINE-TUNING ---
        if args.model == 'efficientnet':
            logger.info("\n>>> ФАЗА 2: FINE-TUNING (Разморозка Backbone)...")
            
            # Сохраняем веса головы
            temp_weights = os.path.join(ckpt_dir, "warmup_weights.h5")
            model.save_weights(temp_weights)
            
            # Пересобираем модель
            model = build_efficientnet_b0(
                num_classes=7, 
                learning_rate=args.lr / 10.0, 
                use_focal_loss=args.focal, 
                freeze_backbone=False,
                use_augmentation=False # Отключили медленную GPU аугментацию
            )
            
            # Загружаем веса
            model.load_weights(temp_weights, by_name=True, skip_mismatch=True)
            logger.info("Веса Warmup загружены. LR уменьшен.")

    # Коллбеки
    callbacks = [
        ModelCheckpoint(os.path.join(ckpt_dir, "best_weights.h5"), monitor='val_auc', mode='max', save_best_only=True, save_weights_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
        CSVLogger(os.path.join(ckpt_dir, "training_log.csv"), append=True)
    ]

    # Основное обучение
    try:
        history = model.fit(
            train_gen, 
            validation_data=val_gen, 
            epochs=args.epochs, 
            initial_epoch=warmup_epochs,
            callbacks=callbacks, 
            workers=n_workers,          # <--- ИСПОЛЬЗУЕМ ВСЕ ЯДРА
            use_multiprocessing=True, 
            max_queue_size=10,
            verbose=1
        )
    except KeyboardInterrupt:
        logger.warning("Обучение прервано пользователем."); return
    except Exception as e:
        logger.critical(f"Ошибка обучения: {e}"); return

    # Сохранение (без изменений)
    with open(os.path.join(ckpt_dir, "history.json"), 'w') as f: 
        json.dump(history.history, f, cls=TFJSONEncoder, indent=4)
    TrainingVisualizer.plot_history(history.history, save_path=os.path.join(ckpt_dir, "history_plot.png"))
    model.save_weights(os.path.join(ckpt_dir, "final_weights.h5"))

    # Оценка (без изменений)
    logger.info("--- ФИНАЛЬНЫЙ ТЕСТ ---")
    try:
        model.load_weights(os.path.join(ckpt_dir, "best_weights.h5"))
    except: pass
    
    # Также используем workers при оценке
    test_metrics = model.evaluate(
        test_gen, 
        verbose=1, 
        return_dict=True, 
        workers=n_workers, 
        use_multiprocessing=True
    )
    
    test_preds = model.predict(
        test_gen, 
        verbose=1, 
        workers=n_workers, 
        use_multiprocessing=True
    )
    
    y_true = test_gen.classes
    y_pred = np.argmax(test_preds, axis=1)
    
    TrainingVisualizer.plot_confusion_matrix(y_true, y_pred, class_order, os.path.join(ckpt_dir, "test_confusion_matrix.png"))
    TrainingVisualizer.plot_roc_curves(y_true, test_preds, class_order, os.path.join(ckpt_dir, "test_roc_curves.png"))
    
    with open(os.path.join(ckpt_dir, "test_metrics.json"), 'w') as f: json.dump(test_metrics, f, cls=TFJSONEncoder, indent=4)
    logger.info(f"УСПЕХ. Результаты: {ckpt_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model', type=str, default='efficientnet', choices=['efficientnet', 'resnet'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--focal', action='store_true')
    parser.add_argument('--no_balance', action='store_true')
    parser.add_argument('--no_augment', action='store_true')
    args = parser.parse_args()
    run_experiment(args)