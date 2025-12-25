#!/bin/bash

# Скрипт автоматического запуска серии экспериментов
# Прерывание выполнения при ошибке любой команды
set -e

echo "========================================================"
echo "НАЧАЛО СЕРИИ ЭКСПЕРИМЕНТОВ: КЛАССИФИКАЦИЯ HAM10000"
echo "Дата запуска: $(date)"
echo "========================================================"

# Константы для всех экспериментов
EPOCHS=100
BATCH_SIZE=64
PATIENCE=8

# --- ЭКСПЕРИМЕНТ 1: EfficientNetB0 + CrossEntropy (Baseline) ---
echo ""
echo "[1/4] Запуск: EfficientNetB0 + Categorical CrossEntropy"
python3 train_experiment.py \
    --model efficientnet \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --patience $PATIENCE

echo "Статус: Эксперимент 1 завершен успешно."

# --- ЭКСПЕРИМЕНТ 2: EfficientNetB0 + Focal Loss ---
echo ""
echo "[2/4] Запуск: EfficientNetB0 + Focal Loss"
python3 train_experiment.py \
    --model efficientnet \
    --focal \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --patience $PATIENCE

echo "Статус: Эксперимент 2 завершен успешно."

# --- ЭКСПЕРИМЕНТ 3: ResNet50 + CrossEntropy ---
echo ""
echo "[3/4] Запуск: ResNet50 + Categorical CrossEntropy"
python3 train_experiment.py \
    --model resnet \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --patience $PATIENCE

echo "Статус: Эксперимент 3 завершен успешно."

# --- ЭКСПЕРИМЕНТ 4: ResNet50 + Focal Loss ---
echo ""
echo "[4/4] Запуск: ResNet50 + Focal Loss"
python3 train_experiment.py \
    --model resnet \
    --focal \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --patience $PATIENCE

echo "Статус: Эксперимент 4 завершен успешно."

echo ""
echo "========================================================"
echo "ВСЕ ЭКСПЕРИМЕНТЫ ВЫПОЛНЕНЫ."
echo "Результаты сохранены в директории ./experiments/"
echo "========================================================"