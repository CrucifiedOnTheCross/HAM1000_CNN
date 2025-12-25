import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
from glob import glob

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
        """
        Инициализация менеджера данных.
        
        :param data_dir: Путь к корневой директории с данными (где лежат изображения и CSV).
        :param csv_filename: Имя файла метаданных.
        """
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, csv_filename)
        self.df: Optional[pd.DataFrame] = None
        self.image_paths_map: dict = {}
        
        # Словарь маппинга классов (согласно методологии)
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
        """Сканирует директорию данных для создания карты {image_id: full_path}."""
        # Ищем все jpg файлы рекурсивно или в плоской структуре внутри data_dir
        # Используем glob для поиска во всех подпапках
        search_pattern = os.path.join(self.data_dir, "**", "*.jpg")
        image_paths = glob(search_pattern, recursive=True)
        
        if not image_paths:
            logger.error("Файлы изображений (.jpg) не обнаружены в указанной директории.")
            raise FileNotFoundError("Изображения отсутствуют.")

        # Создаем словарь image_id -> path
        # filename example: ISIC_0027419.jpg -> key: ISIC_0027419
        self.image_paths_map = {
            os.path.splitext(os.path.basename(path))[0]: path 
            for path in image_paths
        }
        logger.info(f"Индексация изображений завершена. Найдено файлов: {len(self.image_paths_map)}")

    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Загружает CSV, добавляет полные пути и разделяет на train/test.
        
        :param test_size: Доля тестовой выборки.
        :param random_state: Сид для воспроизводимости.
        :return: Кортеж (train_df, test_df).
        """
        if not os.path.exists(self.csv_path):
            logger.critical(f"Файл метаданных не найден: {self.csv_path}")
            raise FileNotFoundError(f"Missing CSV: {self.csv_path}")

        # 1. Загрузка и маппинг
        self._load_image_paths()
        df = pd.read_csv(self.csv_path)
        
        # 2. Добавление колонки path
        df['path'] = df['image_id'].map(self.image_paths_map)
        
        # 3. Фильтрация потерянных изображений (если есть расхождения между CSV и файлами)
        missing_count = df['path'].isna().sum()
        if missing_count > 0:
            logger.warning(f"Обнаружено {missing_count} записей в CSV без соответствующих файлов изображений. Они будут исключены.")
            df = df.dropna(subset=['path'])

        # 4. Кодирование меток (Label Encoding)
        df['label_idx'] = df['dx'].map(self.class_map)
        
        # 5. Стратифицированное разделение
        logger.info("Выполняется стратифицированное разделение выборки...")
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df['label_idx']
        )
        
        logger.info(f"Данные подготовлены. Обучающая выборка: {len(train_df)}, Тестовая выборка: {len(test_df)}")
        return train_df, test_df


class BalancedDataGenerator(tf.keras.utils.Sequence):
    """
    Генератор данных, обеспечивающий сбалансированное присутствие всех классов в каждом батче.
    Использует технику Under-sampling мажоритарных классов и Over-sampling миноритарных 
    в рамках одного батча.
    """

    def __init__(self, 
                 df: pd.DataFrame, 
                 batch_size: int = 28, 
                 input_shape: Tuple[int, int] = (224, 224),
                 shuffle: bool = True,
                 augment: bool = False):
        """
        Инициализация сбалансированного генератора.
        
        :param df: DataFrame с колонками 'path' и 'label_idx'.
        :param batch_size: Размер батча (должен быть кратен количеству классов, иначе округляется).
        :param input_shape: Размер входного изображения (H, W).
        :param augment: Флаг применения аугментации (True для train).
        """
        self.df = df
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.augment = augment
        self.num_classes = 7
        
        # Проверка размера батча
        if batch_size % self.num_classes != 0:
            new_batch_size = (batch_size // self.num_classes) * self.num_classes
            logger.warning(f"Размер батча {batch_size} не кратен 7. Скорректирован до {new_batch_size} для идеального баланса.")
            self.batch_size = new_batch_size
        else:
            self.batch_size = batch_size
            
        self.samples_per_class = self.batch_size // self.num_classes
        
        # Группировка данных по классам для быстрого доступа
        self.groups = [self.df[self.df['label_idx'] == i] for i in range(self.num_classes)]
        
        # Расчет длины эпохи (кол-во батчей). 
        # Т.к. мы балансируем искусственно, эпоха определяется по мажоритарному классу или произвольно.
        # Определим эпоху как (Общее кол-во данных) / batch_size
        self.n_batches = len(self.df) // self.batch_size
        
        logger.info(f"Инициализирован BalancedDataGenerator. Образцов на класс в батче: {self.samples_per_class}")

    def __len__(self) -> int:
        """Возвращает количество батчей в эпохе."""
        return self.n_batches

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует один батч данных.
        
        :param index: Индекс батча (не используется напрямую для выборки при случайном сэмплировании).
        """
        batch_x = []
        batch_y = []

        # Формирование сбалансированного набора
        for class_idx in range(self.num_classes):
            # Случайная выборка N примеров конкретного класса
            # replace=True позволяет брать повторы, если примеров в классе (DF) меньше, чем нужно
            class_samples = self.groups[class_idx].sample(n=self.samples_per_class, replace=True)
            
            for _, row in class_samples.iterrows():
                img = self._load_and_process_image(row['path'])
                batch_x.append(img)
                batch_y.append(class_idx)

        # Конвертация в numpy массивы
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        
        # One-hot encoding для меток
        batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)

        # Перемешивание внутри батча, чтобы классы не шли по порядку [0,0,1,1...]
        if self.shuffle:
            indices = np.arange(self.batch_size)
            np.random.shuffle(indices)
            batch_x = batch_x[indices]
            batch_y = batch_y[indices]

        return batch_x, batch_y

    def _load_and_process_image(self, path: str) -> np.ndarray:
        """
        Загрузка и препроцессинг изображения.
        
        :param path: Путь к файлу.
        :return: Нормализованный массив изображения.
        """
        try:
            # Загрузка через Keras utils
            img = tf.keras.utils.load_img(path, target_size=self.input_shape)
            img_array = tf.keras.utils.img_to_array(img)
            
            # Нормализация [0, 1]
            # ResNet/EfficientNet могут требовать специфичного preprocess_input, 
            # но для начала используем универсальный rescaling 1./255. 
            # В файлах архитектур мы учли preprocess_input для ResNet, здесь отдаем "сырой" нормализованный
            # Если использовать tf.keras.applications, лучше отдавать 0-255 и preprocessing слой внутри модели.
            # Для универсальности здесь вернем 0-255 (float).
            
            # Примечание: В файлах моделей (resnet.py) добавлен слой preprocess_input.
            # Поэтому здесь мы просто возвращаем массив.
            
            if self.augment:
                # Здесь можно внедрить Albumentations или простые tf augmentation
                # Для базовой версии пока оставим без аугментации (или минимальный flip)
                img_array = tf.image.random_flip_left_right(img_array)
                img_array = tf.image.random_brightness(img_array, max_delta=0.1)
                
            return img_array
            
        except Exception as e:
            logger.error(f"Ошибка чтения файла {path}: {e}")
            # Возврат черного квадрата, чтобы не уронить батч (крайний случай)
            return np.zeros((self.input_shape[0], self.input_shape[1], 3))

if __name__ == "__main__":
    # Тест работоспособности
    manager = DataManager(data_dir="./data")
    try:
        train, test = manager.prepare_data()
        
        # Создаем генератор (батч 14 = по 2 примера на класс)
        gen = BalancedDataGenerator(train, batch_size=14)
        X, y = gen[0]
        
        logger.info(f"Тестовый батч сформирован. Shape X: {X.shape}, Shape y: {y.shape}")
        logger.info(f"Пример меток (должны быть разные классы): \n{np.argmax(y, axis=1)}")
        
    except Exception as e:
        logger.error(f"Тест прерван: {e}")