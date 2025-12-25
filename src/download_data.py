import os
import logging
import shutil
from pathlib import Path
from typing import Optional, Union

# Настройка логирования (Строгий запрет на print)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class HamDatasetManager:
    """
    Класс для управления загрузкой и размещением датасета HAM10000
    с использованием библиотеки kagglehub.
    """

    def __init__(self, target_dir: str = "./data", dataset_handle: str = "kmader/skin-cancer-mnist-ham10000"):
        """
        Инициализация менеджера.

        :param target_dir: Целевая директория проекта, куда необходимо поместить данные.
        :param dataset_handle: Идентификатор датасета в репозитории Kaggle.
        """
        self.target_dir = Path(target_dir)
        self.dataset_handle = dataset_handle
        self._ensure_target_dir_exists()

    def _ensure_target_dir_exists(self) -> None:
        """Создает целевую директорию, если она отсутствует."""
        try:
            self.target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.critical(f"Ошибка при создании целевой директории {self.target_dir}: {e}")
            raise

    def check_data_exists(self) -> bool:
        """
        Проверяет наличие файлов датасета в целевой директории.
        
        :return: True, если ключевой файл метаданных найден.
        """
        # Маркерный файл для проверки (основной CSV)
        marker_file = self.target_dir / "HAM10000_metadata.csv"
        
        if marker_file.exists():
            logger.info("Проверка целостности: Данные уже присутствуют в целевой директории.")
            return True
        return False

    def _move_files_from_cache(self, cache_path: str) -> None:
        """
        Перемещает файлы из кэша kagglehub в целевую папку проекта.

        :param cache_path: Путь к директории, куда kagglehub загрузил данные.
        """
        source = Path(cache_path)
        logger.info(f"Начато перемещение данных из кэша ({source}) в целевую директорию ({self.target_dir}).")

        try:
            # shutils.copytree с dirs_exist_ok=True позволяет копировать в существующую папку (начиная с Python 3.8)
            for item in source.iterdir():
                destination = self.target_dir / item.name
                if item.is_dir():
                    if destination.exists():
                        shutil.rmtree(destination)
                    shutil.copytree(item, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, destination)
            
            logger.info("Перемещение файлов выполнено успешно.")
            
        except Exception as e:
            logger.error(f"Ошибка при перемещении файлов: {e}")
            raise

    def download_data(self) -> None:
        """
        Выполняет загрузку через kagglehub и последующее перемещение файлов.
        """
        try:
            import kagglehub
            
            logger.info(f"Инициирована загрузка датасета: {self.dataset_handle}")
            
            # Загрузка в кэш системы
            cache_path = kagglehub.dataset_download(self.dataset_handle)
            logger.info(f"Загрузка в системный кэш завершена: {cache_path}")
            
            # Перемещение в папку проекта
            self._move_files_from_cache(cache_path)
            
        except ImportError:
            logger.critical("Библиотека 'kagglehub' не обнаружена. Требуется установка: pip install kagglehub")
            raise
        except Exception as e:
            logger.critical(f"Критический сбой в процессе загрузки: {e}")
            raise

    def run(self) -> None:
        """
        Точка входа для выполнения сценария.
        """
        if self.check_data_exists():
            logger.info("Загрузка пропущена: Актуальные данные найдены.")
            return

        self.download_data()
        
        # Финальная верификация
        if self.check_data_exists():
            logger.info("Модуль завершил работу: Данные готовы к использованию.")
        else:
            logger.warning("Модуль завершил работу, но файлы не обнаружены в целевой директории.")

if __name__ == "__main__":
    # Запуск процесса
    loader = HamDatasetManager(target_dir="./data")
    loader.run()