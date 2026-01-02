import os

# --- КРИТИЧНО: НАСТРОЙКА ЛОГОВ TF ДО ИМПОРТА ---
# '3' - убирает всё, кроме FATAL ошибок (самый тихий режим)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import multiprocessing
import logging
import warnings
import argparse
import datetime
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tqdm.auto import tqdm

# Импорт локальных модулей
from src.data_loader import DataManager, BalancedDataGenerator
from src.models.efficientnet import build_efficientnet_b0
from src.models.resnet import build_resnet50
from src.models.simple_cnn import build_simple_cnn 
from src.visualization import TrainingVisualizer
from src.serializers import TFJSONEncoder

# Настройка логирования Python (убираем лишнее от библиотек)
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================================================================================
# CUSTOM CALLBACK: COMPACT LOGGING (TRAIN + EVAL + PREDICT)
# ==================================================================================
class CompactTQDMCallback(Callback):
    """
    Универсальный TQDM коллбек.
    Работает для fit() (показывает эпохи), evaluate() и predict().
    """
    def __init__(self, total_epochs=1, prefix='Train'):
        super().__init__()
        self.total_epochs = total_epochs
        self.prefix = prefix
        self.pbar = None
        
        # Список метрик для отображения
        self.whitelist = [
            'accuracy', 
            'auc_pr', 
            'f1_macro', 
            'sens_mel', 
            'spec_mel'
        ]

    def _format_logs(self, logs):
        """Вспомогательная функция для форматирования метрик"""
        display_logs = {}
        if 'loss' in logs:
            display_logs['loss'] = f"{logs['loss']:.3f}"
            
        for k, v in logs.items():
            if k in self.whitelist:
                display_logs[k] = f"{v:.3f}"
        return display_logs

    # --- LOGIC FOR TRAINING (FIT) ---
    def on_epoch_begin(self, epoch, logs=None):
        real_epoch = epoch + 1
        print(f"Epoch {real_epoch}/{self.total_epochs}")
        self.pbar = tqdm(
            total=self.params['steps'], 
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]',
            leave=True,
            desc=self.prefix
        )

    def on_batch_end(self, batch, logs=None):
        if logs and self.pbar:
            self.pbar.set_postfix(self._format_logs(logs))
            self.pbar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        if self.pbar:
            self.pbar.close()
            
        if logs:
            val_items = []
            val_items.append(f"val_loss: {logs.get('val_loss', 0):.4f}")
            if 'val_accuracy' in logs:
                val_items.append(f"val_acc: {logs['val_accuracy']:.4f}")
            
            for k in self.whitelist:
                if k == 'accuracy': continue
                val_key = f"val_{k}"
                if val_key in logs:
                    val_items.append(f"{k}: {logs[val_key]:.4f}")
            
            print(f"   >>> RES: {' | '.join(val_items)}\n")

    # --- LOGIC FOR EVALUATION (TEST) ---
    def on_test_begin(self, logs=None):
        # При evaluate steps передается через self.params['steps']
        self.pbar = tqdm(
            total=self.params['steps'],
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]',
            leave=True,
            desc="Evaluating"
        )

    def on_test_batch_end(self, batch, logs=None):
        if logs and self.pbar:
            self.pbar.set_postfix(self._format_logs(logs))
            self.pbar.update(1)

    def on_test_end(self, logs=None):
        if self.pbar:
            self.pbar.close()

    # --- LOGIC FOR PREDICTION (PREDICT) ---
    def on_predict_begin(self, logs=None):
        self.pbar = tqdm(
            total=self.params['steps'],
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            leave=True,
            desc="Predicting"
        )

    def on_predict_batch_end(self, batch, logs=None):
        # При predict метрик нет, просто двигаем прогресс
        if self.pbar:
            self.pbar.update(1)

    def on_predict_end(self, logs=None):
        if self.pbar:
            self.pbar.close()


# ==================================================================================
# MAIN EXPERIMENT LOGIC
# ==================================================================================

def create_standard_generator(df: pd.DataFrame, input_shape: tuple, batch_size: int, class_order: list, augment: bool, is_training: bool) -> tf.keras.preprocessing.image.Iterator:
    """Создает стандартный ImageDataGenerator."""
    if is_training and augment:
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
        logger.info("Gen: Standard + Augmentation (CPU).")
    else:
        datagen = ImageDataGenerator()
    
    return datagen.flow_from_dataframe(
        dataframe=df, x_col='path', y_col='dx', target_size=input_shape,
        class_mode='categorical', classes=class_order, batch_size=batch_size, shuffle=is_training
    )

def run_experiment(args: argparse.Namespace) -> None:
    logger.info(f"START: {args.model.upper()} | Balance={not args.no_balance} | Aug={not args.no_augment} | Focal={args.focal}")
    
    # Логируем параметры Focal Loss, если он включен
    if args.focal:
        logger.info(f"Focal Params: Gamma={args.focal_gamma}, Alpha={args.focal_alpha}")

    # --- КОНФИГУРАЦИЯ ОБОРУДОВАНИЯ ---
    n_workers = multiprocessing.cpu_count()
    physical_devices = tf.config.list_physical_devices('GPU')
    
    if len(physical_devices) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Hardware: {len(physical_devices)} GPUs (MirroredStrategy). Workers: {n_workers}")
    else:
        strategy = tf.distribute.get_strategy()
        logger.info(f"Hardware: Single GPU/CPU. Workers: {n_workers}")

    # 1. Подготовка данных
    data_manager = DataManager(data_dir=args.data_dir)
    class_order = list(data_manager.class_map.keys())
    
    try:
        train_df, val_df, test_df = data_manager.prepare_data(test_size=0.2)
    except Exception as e:
        logger.critical(f"Data Error: {e}"); return

    input_shape = (224, 224)
    batch_size = args.batch_size

    # 2. Создание генераторов
    if args.no_balance:
        train_gen = create_standard_generator(
            train_df, input_shape, batch_size, class_order, 
            augment=not args.no_augment, 
            is_training=True
        )
    else:
        logger.info("Gen: BalancedDataGenerator (RAM Cached)...")
        train_gen = BalancedDataGenerator(
            train_df, 
            batch_size=batch_size, 
            input_shape=input_shape, 
            shuffle=True, 
            augment=not args.no_augment, 
            cache_images=True
        )

    val_gen = create_standard_generator(val_df, input_shape, batch_size, class_order, False, False)
    test_gen = create_standard_generator(test_df, input_shape, batch_size, class_order, False, False)

    # 3. Настройка путей
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.model}_{'imbal' if args.no_balance else 'bal'}_{'noaug' if args.no_augment else 'aug'}_{'focal' if args.focal else 'ce'}_{timestamp}"
    ckpt_dir = os.path.join("experiments", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    test_df.to_csv(os.path.join(ckpt_dir, "test_dataset.csv"), index=False)

    # ==========================================
    # 4. ДВУХЭТАПНОЕ ОБУЧЕНИЕ
    # ==========================================
    with strategy.scope():
        # --- ФАЗА 1: WARMUP ---
        logger.info("\n>>> PHASE 1: WARMUP...")
        
        # ЛОГИКА ВЫБОРА МОДЕЛИ
        if args.model == 'simple_cnn':
            model = build_simple_cnn(
                num_classes=7,
                learning_rate=args.lr,
                use_focal_loss=args.focal,
                # Передаем параметры Focal Loss
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                use_augmentation=not args.no_augment
            )
        elif args.model == 'efficientnet':
            model = build_efficientnet_b0(
                num_classes=7, 
                learning_rate=args.lr, 
                use_focal_loss=args.focal,
                # Передаем параметры Focal Loss
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                freeze_backbone=True,
                use_augmentation=False 
            )
        elif args.model == 'resnet':
            model = build_resnet50(
                num_classes=7, 
                learning_rate=args.lr, 
                use_focal_loss=args.focal,
                # Передаем параметры Focal Loss
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha
            )

        warmup_epochs = 3
        
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=warmup_epochs,
            verbose=0, 
            workers=n_workers,
            use_multiprocessing=False, 
            max_queue_size=10,
            callbacks=[
                CompactTQDMCallback(warmup_epochs, prefix='Warmup'),
                CSVLogger(os.path.join(ckpt_dir, "log_warmup.csv"), append=True)
            ]
        )

        # --- ФАЗА 2: FINE-TUNING (Только для EfficientNet) ---
        if args.model == 'efficientnet':
            logger.info("\n>>> PHASE 2: FINE-TUNING (Unfrozen)...")
            
            temp_weights = os.path.join(ckpt_dir, "warmup_weights.h5")
            model.save_weights(temp_weights)
            
            model = build_efficientnet_b0(
                num_classes=7, 
                learning_rate=args.lr / 10.0, 
                use_focal_loss=args.focal,
                # Передаем параметры Focal Loss
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                freeze_backbone=False,
                use_augmentation=False
            )
            
            model.load_weights(temp_weights, by_name=True, skip_mismatch=True)
            logger.info("Warmup weights loaded. LR reduced.")

    # Коллбеки для основной фазы
    callbacks = [
        ModelCheckpoint(os.path.join(ckpt_dir, "best_weights.h5"), monitor='val_auc_pr', mode='max', save_best_only=True, save_weights_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
        CSVLogger(os.path.join(ckpt_dir, "training_log.csv"), append=True),
        CompactTQDMCallback(args.epochs, prefix='Training') # TQDM для обучения
    ]

    # Основное обучение
    try:
        history = model.fit(
            train_gen, 
            validation_data=val_gen, 
            epochs=args.epochs, 
            initial_epoch=warmup_epochs,
            callbacks=callbacks, 
            workers=n_workers, 
            use_multiprocessing=False, 
            max_queue_size=10,
            verbose=0 
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user."); return
    except Exception as e:
        logger.critical(f"Training failed: {e}"); return

    # Сохранение и визуализация
    with open(os.path.join(ckpt_dir, "history.json"), 'w') as f: 
        json.dump(history.history, f, cls=TFJSONEncoder, indent=4)
    
    TrainingVisualizer.plot_history(history.history, save_path=os.path.join(ckpt_dir, "history_plot.png"))
    model.save_weights(os.path.join(ckpt_dir, "final_weights.h5"))

    # Оценка
    logger.info("\n--- FINAL EVALUATION ---")
    try:
        model.load_weights(os.path.join(ckpt_dir, "best_weights.h5"))
    except: pass
    
    # -------------------------------------------------------------
    # Оценка на тесте с TQDM
    # -------------------------------------------------------------
    # Используем CompactTQDMCallback и для evaluate, и для predict
    # Важно: ставим verbose=0, чтобы убрать стандартный бар Keras
    
    eval_callback = CompactTQDMCallback()
    test_metrics = model.evaluate(
        test_gen, 
        verbose=0, 
        workers=n_workers, 
        use_multiprocessing=False,
        callbacks=[eval_callback], # <--- Подключаем TQDM
        return_dict=True
    )
    
    predict_callback = CompactTQDMCallback()
    test_preds = model.predict(
        test_gen, 
        verbose=0, 
        workers=n_workers, 
        use_multiprocessing=False,
        callbacks=[predict_callback] # <--- Подключаем TQDM
    )
    
    y_true = test_gen.classes
    y_pred = np.argmax(test_preds, axis=1)
    
    # Визуализация результатов
    TrainingVisualizer.plot_confusion_matrix(y_true, y_pred, class_order, os.path.join(ckpt_dir, "test_confusion_matrix.png"))
    TrainingVisualizer.plot_roc_curves(y_true, test_preds, class_order, os.path.join(ckpt_dir, "test_roc_curves.png"))
    TrainingVisualizer.plot_pr_curves(y_true, test_preds, class_order, os.path.join(ckpt_dir, "test_pr_curves.png"))
    
    with open(os.path.join(ckpt_dir, "test_metrics.json"), 'w') as f: json.dump(test_metrics, f, cls=TFJSONEncoder, indent=4)
    logger.info(f"Done. Saved to: {ckpt_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model', type=str, default='efficientnet', choices=['efficientnet', 'resnet', 'simple_cnn'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--focal', action='store_true')
    parser.add_argument('--no_balance', action='store_true')
    parser.add_argument('--no_augment', action='store_true')
    
    # --- НОВЫЕ АРГУМЕНТЫ ДЛЯ FOCAL LOSS ---
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Alpha parameter for Focal Loss')
    
    args = parser.parse_args()
    run_experiment(args)