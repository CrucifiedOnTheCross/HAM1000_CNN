import os

# --- БЛОК ПОДАВЛЕНИЯ ПРЕДУПРЕЖДЕНИЙ ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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

# Отключение питоновских предупреждений
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)

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
    if is_training and augment:
        datagen = ImageDataGenerator(horizontal_flip=True, brightness_range=[0.9, 1.1])
        logger.info("Создан стандартный генератор С АУГМЕНТАЦИЕЙ.")
    else:
        datagen = ImageDataGenerator()
        logger.info(f"Создан генератор ({'ВАЛИДАЦИЯ/ТЕСТ' if not is_training else 'ОБУЧЕНИЕ'}).")

    return datagen.flow_from_dataframe(
        dataframe=df, x_col='path', y_col='dx', target_size=input_shape,
        class_mode='categorical', classes=class_order, batch_size=batch_size, shuffle=is_training
    )

def run_experiment(args: argparse.Namespace) -> None:
    logger.info(f"Старт эксперимента: {args.model} | Balance={not args.no_balance} | Aug={not args.no_augment} | Focal={args.focal}")

    # --- GPU ---
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Используется {strategy.num_replicas_in_sync} GPU.")
    else:
        strategy = tf.distribute.get_strategy()

    # 1. Данные
    data_manager = DataManager(data_dir=args.data_dir)
    class_order = list(data_manager.class_map.keys())
    try:
        # Распаковываем 3 датафрейма
        train_df, val_df, test_df = data_manager.prepare_data(test_size=0.2)
    except Exception as e:
        logger.critical(f"Ошибка данных: {e}"); return

    input_shape = (224, 224)

    # Генераторы
    if args.no_balance:
        train_gen = create_standard_generator(train_df, input_shape, args.batch_size, class_order, not args.no_augment, True)
    else:
        train_gen = BalancedDataGenerator(train_df, args.batch_size, input_shape, not args.no_augment)

    val_gen = create_standard_generator(val_df, input_shape, args.batch_size, class_order, False, False)
    
    # Генератор для ТЕСТА (создаем сразу)
    test_gen = create_standard_generator(test_df, input_shape, args.batch_size, class_order, False, False)

    # 2. Модель
    with strategy.scope():
        if args.model == 'efficientnet':
            model = build_efficientnet_b0(num_classes=7, learning_rate=args.lr, use_focal_loss=args.focal)
        elif args.model == 'resnet':
            model = build_resnet50(num_classes=7, learning_rate=args.lr, use_focal_loss=args.focal)
        else: return

    # 3. Пути
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.model}_{'imbalanced' if args.no_balance else 'balanced'}_{'noaug' if args.no_augment else 'aug'}_{'focal' if args.focal else 'ce'}_{timestamp}"
    ckpt_dir = os.path.join("experiments", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Сохраняем test_df для истории
    test_df.to_csv(os.path.join(ckpt_dir, "test_dataset.csv"), index=False)

    callbacks = [
        # save_weights_only=True обязательно для мульти-GPU
        ModelCheckpoint(os.path.join(ckpt_dir, "best_weights.h5"), monitor='val_auc', mode='max', save_best_only=True, save_weights_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
        CSVLogger(os.path.join(ckpt_dir, "training_log.csv"), append=True)
    ]

    try:
        TrainingVisualizer.plot_class_distribution(train_df, 'dx', save_path=os.path.join(ckpt_dir, "data_distribution.png"))
    except: pass

    # 4. Обучение
    try:
        history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks, verbose=1)
    except KeyboardInterrupt:
        logger.warning("Обучение прервано."); return
    except Exception as e:
        logger.critical(f"Ошибка обучения: {e}"); return

    # 5. Сохранение истории
    with open(os.path.join(ckpt_dir, "history.json"), 'w') as f: json.dump(history.history, f, cls=TFJSONEncoder, indent=4)
    TrainingVisualizer.plot_history(history.history, save_path=os.path.join(ckpt_dir, "history_plot.png"))
    
    # Сохранение финальных весов
    model.save_weights(os.path.join(ckpt_dir, "final_weights.h5"))

    # ==========================================
    # 6. Оценка на VALIDATION (для контроля)
    # ==========================================
    logger.info("--- ОЦЕНКА НА VALIDATION SET ---")
    val_gen.reset()
    val_metrics = model.evaluate(val_gen, verbose=1, return_dict=True)
    
    val_preds = model.predict(val_gen, verbose=1)
    TrainingVisualizer.plot_confusion_matrix(val_gen.classes, np.argmax(val_preds, axis=1), class_order, os.path.join(ckpt_dir, "val_confusion_matrix.png"))
    TrainingVisualizer.plot_roc_curves(val_gen.classes, val_preds, class_order, os.path.join(ckpt_dir, "val_roc_curves.png"))
    
    with open(os.path.join(ckpt_dir, "val_metrics.json"), 'w') as f: json.dump(val_metrics, f, cls=TFJSONEncoder, indent=4)

    # ==========================================
    # 7. ФИНАЛЬНАЯ ОЦЕНКА НА TEST SET (Для диплома)
    # ==========================================
    logger.info("--- ФИНАЛЬНАЯ ОЦЕНКА НА TEST SET ---")
    
    # Загружаем ЛУЧШИЕ веса перед тестом
    logger.info("Загрузка лучших весов для тестирования...")
    try:
        model.load_weights(os.path.join(ckpt_dir, "best_weights.h5"))
    except Exception as e:
        logger.warning(f"Не удалось загрузить лучшие веса ({e}), используем текущие.")

    test_gen.reset()
    test_metrics = model.evaluate(test_gen, verbose=1, return_dict=True)
    logger.info(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    logger.info(f"Test AUC: {test_metrics.get('auc', 0):.4f}")

    # Предсказания для графиков
    test_preds = model.predict(test_gen, verbose=1)
    y_test_true = test_gen.classes
    y_test_pred = np.argmax(test_preds, axis=1)

    # 1. Матрица ошибок (TEST)
    TrainingVisualizer.plot_confusion_matrix(
        y_true=y_test_true, 
        y_pred=y_test_pred, 
        classes=class_order, 
        save_path=os.path.join(ckpt_dir, "test_confusion_matrix.png")
    )

    # 2. ROC-кривые (TEST)
    TrainingVisualizer.plot_roc_curves(
        y_true=y_test_true, 
        y_pred_probs=test_preds, 
        classes=class_order, 
        save_path=os.path.join(ckpt_dir, "test_roc_curves.png")
    )

    # 3. Сохранение метрик (TEST)
    with open(os.path.join(ckpt_dir, "test_metrics.json"), 'w') as f:
        json.dump(test_metrics, f, cls=TFJSONEncoder, indent=4)

    logger.info(f"Эксперимент завершен. Результаты в: {ckpt_dir}")

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