#!/bin/bash

# Скрипт автоматического запуска серии экспериментов для диплома
# Прерывание выполнения при ошибке любой команды
set -e

echo "========================================================"
echo "ЗАПУСК ЭКСПЕРИМЕНТОВ: HAM10000 (efficientnet)"
echo "Дата запуска: $(date)"
echo "========================================================"

# --- ГЛАВНЫЕ НАСТРОЙКИ ---
# Выберите модель: 'simple_cnn', 'efficientnet' или 'resnet'
MODEL_NAME="efficientnet"

EPOCHS=100      # Полноценное обучение
BATCH_SIZE=128   # Оптимально для EfficientNetB0 на K80/T4
PATIENCE=10      # Терпение EarlyStopping

echo "Используемая модель: $MODEL_NAME"
echo "Эпох: $EPOCHS, Батч: $BATCH_SIZE"

# # ========================================================
# # ЭТАП 1: BASELINE (Базовый уровень)
# # ========================================================
# echo ""
# echo "[1/3] ЗАПУСК: 1. Baseline (Imbalanced, No Augmentation)"
python3 train_experiment.py \
    --model $MODEL_NAME \
    --no_balance \
    --no_augment \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --patience $PATIENCE

# echo "СТАТУС: Этап 1 завершен."

# # ========================================================
# # ЭТАП 2: ВЛИЯНИЕ АУГМЕНТАЦИИ
# # ========================================================
# echo ""
# echo "[2/3] ЗАПУСК: 2. With Augmentation (Imbalanced, Augmentation)"
# python3 train_experiment.py \
#     --model $MODEL_NAME \
#     --no_balance \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --patience $PATIENCE

# echo "СТАТУС: Этап 2 завершен."

# ========================================================
# ЭТАП 3: ИТОГОВЫЙ МЕТОД (Proposed Method)
# ========================================================
# echo ""
# echo "[3/3] ЗАПУСК: 3. Proposed Method (Balanced, Augmentation, Focal Loss)"
# python3 train_experiment.py \
#     --model $MODEL_NAME \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --patience $PATIENCE
    
# echo "СТАТУС: Этап 3 завершен."

echo ""
echo "========================================================"
echo "ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ."
echo "========================================================"