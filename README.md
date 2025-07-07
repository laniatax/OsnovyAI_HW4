# Сравнение полносвязных и сверточных нейронных сетей

## Описание проекта

Цель проекта — провести сравнительный анализ различных архитектур нейронных сетей (полносвязные, сверточные, модели с остаточными связями) на задачах классификации изображений. Исследование проводится на датасетах MNIST и CIFAR-10.

Проект включает:
- сравнение точности и переобучения моделей;
- эксперименты с архитектурой CNN: глубина, ядра свертки, остаточные блоки;
- реализацию и тестирование пользовательских слоев: attention, кастомные активации и pooling;
- визуализацию активаций и feature-карт;
- замеры времени инференса и числа параметров моделей.

## Структура проекта

homework/

├── homework_cnn_vs_fc_comparison.py

├── homework_cnn_architecture_analysis.py

├── homework_custom_layers_experiments.py



├── models/

│ ├── fc_models.py

│ ├── cnn_models.py

│ └── custom_layers.py

├── utils/

│ ├── training_utils.py

│ ├── visualization_utils.py

│ └── comparison_utils.py


├── results/

│ ├── mnist_comparison/

│ ├── cifar_comparison/

│ └── architecture_analysis/


├── plots/
└── README.md


## Запуск

###1. Установить зависимости:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn
```

###2. Запустить нужный модуль:

python homework_cnn_vs_fc_comparison.py

python homework_cnn_architecture_analysis.py

python homework_custom_layers_experiments.py
