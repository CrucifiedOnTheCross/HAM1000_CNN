import os
import json
import logging
import warnings
import argparse
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# --- БЛОК ПОДАВЛЕНИЯ ПРЕДУПРЕЖДЕНИЙ ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)

# Импорт локальных модулей
from src.data_loader import DataManager, BalancedDataGenerator
from src.models.efficientnet import build_efficientnet_b0
from src.models.resnet import build_resnet50
from src.visualization import TrainingVisualizer
from src.serializers import TFJSONEncoder  # Импорт исправления для JSON

# Настройка основного логирования проекта
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_validation_generator(df: pd.DataFrame, input_shape: tuple, batch_size: int, class_order: list) -> tf.keras.preprocessing.image.Iterator:
    """
    Создает генератор для валидационной выборки.
    """
    val_datagen = ImageDataGenerator()
    generator = val_datagen.flow_from_dataframe(
        dataframe=df,
        x_col='path',
        y_col='dx',
        target_size=input_shape,
        class_mode='categorical',
        classes=class_order,
        batch_size=batch_size,
        shuffle=False  # Важно: для корректной оценки shuffle должен быть False
    )
    return generator

def run_experiment(args: argparse.Namespace) -> None:
    """
    Основная логика проведения эксперимента.
    """
    logger.info("Инициализация эксперимента с параметрами:")
    logger.info(f"Архитектура: {args.model}")
    logger.info(f"Функция потерь: {'Focal Loss' if args.focal else 'CrossEntropy'}")
    logger.info(f"Параметр Patience: {args.patience}")

    # Проверка доступности GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        logger.info(f"Вычисления будут выполнены на GPU: {len(physical_devices)} устройств(а).")
    else:
        logger.warning("GPU не обнаружен. Вычисления будут выполнены на CPU (возможна низкая производительность).")

    # 1. Подготовка данных
    data_manager = DataManager(data_dir=args.data_dir)
    try:
        train_df, val_df = data_manager.prepare_data(test_size=0.2)
    except Exception as e:
        logger.critical(f"Ошибка подготовки данных: {e}")
        return

    # Генератор для обучения (Сбалансированный)
    train_gen = BalancedDataGenerator(
        train_df, 
        batch_size=args.batch_size, 
        input_shape=(224, 224),
        augment=True
    )

    # Генератор для валидации
    class_order = list(data_manager.class_map.keys())
    val_gen = create_validation_generator(val_df, (224, 224), args.batch_size, class_order)

    # 2. Сборка модели
    if args.model == 'efficientnet':
        model = build_efficientnet_b0(num_classes=7, learning_rate=args.lr, use_focal_loss=args.focal)
    elif args.model == 'resnet':
        model = build_resnet50(num_classes=7, learning_rate=args.lr, use_focal_loss=args.focal)
    else:
        logger.error("Указана неизвестная архитектура модели.")
        return

    # 3. Настройка Callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{args.model}_{'focal' if args.focal else 'ce'}_{timestamp}"
    checkpoint_dir = os.path.join("experiments", experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.h5"),
            monitor='val_auc', 
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # Ранняя остановка (Early Stopping)
        EarlyStopping(
            monitor='val_loss',         # Отслеживаемая метрика
            patience=args.patience,     # Количество эпох без улучшений
            min_delta=0.001,            # Минимальное значимое изменение
            restore_best_weights=True,  # Восстановление весов лучшей эпохи
            verbose=1
        ),
        CSVLogger(
            filename=os.path.join(checkpoint_dir, "training_log.csv"),
            append=True
        )
    ]

    # 4. Запуск обучения
    logger.info("Запуск процесса обучения...")
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1
        )
        logger.info("Обучение завершено успешно.")
    except KeyboardInterrupt:
        logger.warning("Обучение прервано пользователем.")
        return
    except Exception as e:
        logger.critical(f"Критическая ошибка в процессе обучения: {e}")
        return

    # 5. Итоговая оценка на валидационном наборе (Evaluation)
    logger.info("Выполняется итоговая оценка модели на валидационной выборке...")
    try:
        # evaluate возвращает список значений метрик
        val_metrics = model.evaluate(val_gen, verbose=1, return_dict=True)
        
        logger.info("Получены итоговые метрики валидации:")
        for metric, value in val_metrics.items():
            logger.info(f" - {metric}: {value:.4f}")
            
        # Сохранение метрик валидации в отдельный файл
        metrics_path = os.path.join(checkpoint_dir, "final_validation_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(val_metrics, f, cls=TFJSONEncoder, indent=4)
            
    except Exception as e:
        logger.error(f"Ошибка при вычислении итоговых метрик: {e}")

    # 6. Сохранение истории и графиков
    logger.info("Генерация отчетов и графиков...")
    
    # Сохранение истории в JSON
    history_path = os.path.join(checkpoint_dir, "history.json")
    try:
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history.history, f, cls=TFJSONEncoder, indent=4)
        logger.info("Файл истории обучения успешно сохранен.")
    except Exception as e:
        logger.error(f"Не удалось сохранить историю обучения в JSON: {e}")

    # Построение графиков
    try:
        plot_path = os.path.join(checkpoint_dir, "history_plot.png")
        TrainingVisualizer.plot_history(history.history, save_path=plot_path)
    except Exception as e:
        logger.error(f"Не удалось построить график истории: {e}")
    
    # Сохранение финальной модели (фактически это best_model из-за restore_best_weights)
    final_model_path = os.path.join(checkpoint_dir, "final_model.h5")
    model.save(final_model_path)
    logger.info(f"Эксперимент завершен. Артефакты сохранены в: {checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск эксперимента по классификации HAM10000")
    
    parser.add_argument('--data_dir', type=str, default='./data', help='Путь к данным')
    parser.add_argument('--model', type=str, default='efficientnet', choices=['efficientnet', 'resnet'], help='Архитектура модели')
    parser.add_argument('--epochs', type=int, default=30, help='Максимальное количество эпох')
    parser.add_argument('--batch_size', type=int, default=28, help='Размер батча')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--focal', action='store_true', help='Использовать Focal Loss')
    parser.add_argument('--patience', type=int, default=8, help='Количество эпох без улучшений для остановки')
    
    args = parser.parse_args()
    run_experiment(args)