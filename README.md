ML in production. Homework 1.

Download dataset:
   1. Use Kaggle public API (https://www.kaggle.com/docs/api). 
   2. Type the following commands:
      - cd ml_project/data/raw
      - kaggle datasets download -d cherngs/heart-disease-cleveland-uci
      - unzip heart-disease-cleveland-uci.zip 
        - rm heart-disease-cleveland-uci.zip 
RUN train with default config:
   cd ml_project
   python -m src.pipelines.train_pipeline

RUN train with overrode config (all pipelines' configs are in configs/train_pipelines folder):
   cd ml_project
   python -m src.pipelines.train_pipeline [+train_pipelines=PIPELINE_CONFIG_NAME]

Run predict with default config:
    cd ml_project
    python -m src.pipelines.predict_pipeline

RUN tests:
   cd ml_project
   pytest


Архитектурные решения:
- Проект ml-project представляет собой пакет Python. Структура пакета (заимствована стуктура cookiecutter-data-science):
```
├── __init__.py
├── README.md         
├── data
│   └── raw            <- Исходный датасет
│
├── notebooks          <- Результаты анализа данных и прототипы.
│   └── EDA.ipynb
│
├── configs            <- Конфигурации проекта в формате .yaml
│   ├── train_pipelines      <- Измененные default конфигурации train пайплайнов
│   ├── features
│   ├── model
│   ├── preprocessing
│   ├── split
│   ├── default_train_pipeline.yaml
│   └── default_predict_pipeline.yaml
│   
├── requirements.txt  
│
├── outputs            <- Все артефакты создаваемые пайплайнами (модели, метрики, используемые конфигурационные файлы).  
│   ├── train
│   └── inference
└── src                <- Исходный код.
    ├── __init__.py    
    ├── tests
    └── pipelines
        ├── __init__.py
        ├── train_pipeline.py   
        ├── predict_pipeline.py 
        │
        ├── data           <- Модули, содержащие код для предобработки данных.
        │   ├── __init__.py
        │   ├── features_params.py   <- Класс для конфигурации (Выделяемые из датасета признаки).
        │   ├── split_params.py      <- Класс для конфигурации (Размеры тренировочного и тестого датасетов).
        │   └── make_dataset.py      <- Создание датафрейма, разбиение на тренировочные и тестовые данные.
        │
        ├── preprocessing       <-  Содержит трансформеры для подготовки данных.
        │   ├── __init__.py
        │   ├── preprocessing_params.py     <- Класс для конфигурации (Параметры трансформера).
        │   └── transformers.py
        │
        └── models      <- Обучение, использовние и конфигурирование модели.
            ├── __init__.py
            ├── model_params.py   <- Класс для конфигурации (Параметры модели).
            ├── predict_model.py  
            └── train_model.py    
```
- Для тренировки модели и использования обученной модели созданы соответствующие пайплайны (src/pipelines).
- Проект имеет модульную структуру (за каждую часть пайплайна отвечает определенный пакет).
- Для конфигурирования использована hydra.
- Данные являются частью проекта (так неправильно, но так проще).
- Артефакты сохраняются в папке outputs/train, outputs/inference.

1. В описании к пулл реквесту описаны основные "архитектурные" и тактические решения. (1/1)
2. В пулл-реквесте проведена самооценка (1/1)
3. Выполнено EDA и прототипирование. Нет скрипта (1/2)
4. Написана функция/класс для тренировки модели, вызов оформлен как утилита командной строки, записана в readme инструкция по запуску (3/3)
5. Написана функция/класс predict (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, 
   тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в readme (3/3)
6. Проект имеет модульную структуру. Описание структуры проекта в README.md (2/2)
7. Использованы логгеры (2/2)
   - Используется модуль logging в основных модулях (пайплайны, обучение, получение датасета).
   - Настройки модуля - измененнные настройки логгера hydra (вывод в файл и в консоль).
   - Логи подсвечиваются в консоли.
8. Написаны тесты на отдельные модули и на прогон обучения и predict (Частично, только для пайпланов и загрузки их конфигов: 2/3)
9. Для тестов генерируются синтетические данные, приближенные к реальным. (0.5/2)
   Только для тестирования inference (случайные значения features и target с помощью numpy.random)
10. Обучение модели конфигурируется с помощью конфигов yaml. (3/3) 
    Есть две конфигурации (отличаются модели, признаки и параметры разбиения на train, test):
    - configs/default_train_pipeline.yaml
    - configs/pipelines/forest_train_pipeline.yaml
11. Используются датаклассы для сущностей из конфига, а не голые dict (2/2)
    - Все датакслассы конфигов находятся в директориях с соответствующими модулями.
    - Имена классов модулей содержащих датаклассы конфигов с суффиксом '_params'.
12. Напишите кастомный трансформер и протестируйте его (0/3)
13. В проекте зафиксированы все зависимости (1/1)
14. Настроен CI для прогона тестов, линтера на основе github actions (3/3).

Дополнительные баллы:
1. Используется hydra для конфигурирования (3 балла)
2. Mlflow (Не используется: 0 баллов)
