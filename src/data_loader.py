import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
from glob import glob
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Класс для управления метаданными и разделения выборки.
    Обеспечивает корректное сопоставление путей к файлам и меток.
    """
    
    def __init__(self, data_dir: str = "./data", csv_filename: str = "HAM10000_metadata.csv"):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, csv_filename)
        self.df: Optional[pd.DataFrame] = None
        self.image_paths_map: dict = {}
        
        # Словарь маппинга классов
        self.class_map = {
            'nv': 0,    # Melanocytic nevi
            'mel': 1,   # Melanoma
            'bkl': 2,   # Benign keratosis-like lesions
            'bcc': 3,   # Basal cell carcinoma
            'akiec': 4, # Actinic keratoses
            'vasc': 5,  # Vascular lesions
            'df': 6     # Dermatofibroma
        }
        self.num_classes = len(self.class_map)

    def _load_image_paths(self) -> None:
        search_pattern = os.path.join(self.data_dir, "**", "*.jpg")
        image_paths = glob(search_pattern, recursive=True)
        
        if not image_paths:
            logger.error("Файлы изображений (.jpg) не обнаружены в указанной директории.")
            raise FileNotFoundError("Изображения отсутствуют.")

        self.image_paths_map = {
            os.path.splitext(os.path.basename(path))[0]: path 
            for path in image_paths
        }
        logger.info(f"Индексация изображений завершена. Найдено файлов: {len(self.image_paths_map)}")

    def prepare_data(self, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not os.path.exists(self.csv_path):
            logger.critical(f"Файл метаданных не найден: {self.csv_path}")
            raise FileNotFoundError(f"Missing CSV: {self.csv_path}")

        self._load_image_paths()
        df = pd.read_csv(self.csv_path)
        df['path'] = df['image_id'].map(self.image_paths_map)
        
        # Фильтрация потерянных файлов
        df = df.dropna(subset=['path'])

        # Кодирование меток
        df['label_idx'] = df['dx'].map(self.class_map)
        
        logger.info("Выполняется стратифицированное разделение выборки на Train/Val/Test...")
        
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['label_idx']
        )
        
        relative_val_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=relative_val_size, random_state=random_state, stratify=train_val_df['label_idx']
        )
        
        logger.info(f"Данные подготовлены.")
        logger.info(f"Обучающая выборка (Train): {len(train_df)}")
        logger.info(f"Валидационная выборка (Val): {len(val_df)}")
        logger.info(f"Тестовая выборка (Test): {len(test_df)}")
        
        return train_df, val_df, test_df

class BalancedDataGenerator(tf.keras.utils.Sequence):
    """
    Генератор сбалансированных батчей.
    Использует чистый NumPy для аугментации, чтобы избежать конфликтов TF Graph в multiprocessing.
    """
    def __init__(self, 
                 df: pd.DataFrame, 
                 batch_size: int = 28, 
                 input_shape: Tuple[int, int] = (224, 224),
                 shuffle: bool = True,
                 augment: bool = False,
                 cache_images: bool = True):
        
        self.df = df
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.augment = augment
        self.cache_images = cache_images
        self.num_classes = 7
        
        # Корректировка размера батча под количество классов
        if batch_size % self.num_classes != 0:
            self.batch_size = (batch_size // self.num_classes) * self.num_classes
        else:
            self.batch_size = batch_size
            
        self.samples_per_class = self.batch_size // self.num_classes
        self.groups = [self.df[self.df['label_idx'] == i] for i in range(self.num_classes)]
        self.n_batches = len(self.df) // self.batch_size
        
        # Кэширование
        self.image_cache = {}
        if self.cache_images:
            logger.info(f"Загрузка {len(df)} изображений в RAM для ускорения...")
            unique_paths = self.df['path'].unique()
            for path in tqdm(unique_paths, desc="Caching images"):
                self.image_cache[path] = self._load_raw_image(path)
            logger.info(f"Загружено в память. Размер кэша: {len(self.image_cache)}")

    def __len__(self) -> int:
        return self.n_batches

    def _load_raw_image(self, path: str) -> np.ndarray:
        try:
            # Используем Keras load_img, возвращаем numpy array сразу
            img = tf.keras.utils.load_img(path, target_size=self.input_shape)
            img_array = tf.keras.utils.img_to_array(img) # float32, shape (H, W, 3)
            return img_array
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)

    def _augment_image_numpy(self, img: np.ndarray) -> np.ndarray:
        """
        Аугментация на чистом NumPy (Thread-safe).
        """
        # 1. Flip Left-Right
        if np.random.rand() < 0.5:
            img = np.fliplr(img)
            
        # 2. Flip Up-Down (для дерматоскопии это допустимо, т.к. ориентация не важна)
        if np.random.rand() < 0.5:
            img = np.flipud(img)

        # 3. Rotate 90/180/270
        k = np.random.randint(4)
        if k > 0:
            img = np.rot90(img, k=k)
            
        # 4. Brightness (простая реализация на numpy)
        # img в диапазоне 0-255
        if np.random.rand() < 0.5:
            delta = np.random.uniform(-30, 30) # Сдвиг яркости
            img = img + delta
            img = np.clip(img, 0, 255) # Обрезаем, чтобы не вылететь за границы
            
        return img

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_x = []
        batch_y = []

        for class_idx in range(self.num_classes):
            # Берем случайные образцы для каждого класса
            # replace=True гарантирует, что мы сможем набрать нужное кол-во даже для редких классов
            class_samples = self.groups[class_idx].sample(n=self.samples_per_class, replace=True)
            
            for _, row in class_samples.iterrows():
                path = row['path']
                
                # Получаем базовое изображение
                if self.cache_images and path in self.image_cache:
                    # ВАЖНО: .copy() обязателен, иначе аугментация испортит кэш
                    img_array = self.image_cache[path].copy()
                else:
                    img_array = self._load_raw_image(path)
                
                # Применяем аугментацию (если включена)
                if self.augment:
                    img_array = self._augment_image_numpy(img_array)

                batch_x.append(img_array)
                batch_y.append(class_idx)

        # Собираем батч
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y)
        
        # One-hot encoding
        batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)

        # Перемешиваем внутри батча, чтобы классы не шли по порядку (0,0,0...1,1,1...)
        if self.shuffle:
            indices = np.arange(self.batch_size)
            np.random.shuffle(indices)
            batch_x = batch_x[indices]
            batch_y = batch_y[indices]

        return batch_x, batch_y

if __name__ == "__main__":
    manager = DataManager(data_dir="./data")
    try:
        train, val, test = manager.prepare_data()
    except Exception as e:
        logger.error(f"Тест прерван: {e}")