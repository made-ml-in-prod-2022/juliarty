This is the second homework.

To build docker image run:

```
    docker build -t juliarty/online_inference .
```

To pull the latest official image run:

```
    docker pull juliarty/online_inference:latest
```

To run a container run:

```
    docker run -p 9999:9999 -e MODEL_GDRIVE_URL=https://drive.google.com/uc?id=1YY9Vo5wxwUOa_42t52zYL4to-wxJtqIY juliarty/online_inference
```

To test service run:

```
    python -m test_service --host 0.0.0.0 --port 9999 --req-num 20
```


**Основная часть**

1) Есть endpoint /predict (3/3)
2) Есть endpoint /health, возвращает 200, если ваша модель готова к работе, скачались артефакты. (1/1)
3) Есть unit тест для /predict (проверка отправки хороших параметров и плохих) (3/3)
4) Есть скрипт, который будет делает запросы к сервису в виде утилиты командной строки (test_service генерирует данные и запросы) (2/2)

5) Есть Dockrfile для сборки docker image, команда по сборке указана (4/4)
6) Опубликован образ в https://hub.docker.com/ (juliarty/online_inference) (2/2)
7) Приведены корректные команды docker pull/run, скрипт test_service взаимодействует с контейнером корректно (1/1)

8) проведена самооценка (1/1)


**Дополнительная часть**: 
1) Сервис скачивает модель с GoogleDrive, путь для скачивания передается через переменную окружения MODEL_GDRIVE_URL (2/2)
2) Не оптимизирован размер docker image (0/2) 
3) Сделана валидация входных данных (проверка типов, проверка диапазонов), при этом возвращается 400, если валидация не пройдена (2/2)

Всего баллов: 3/3 + 1/1 + 3/3 + 2/2 + 4/4 + 2/2 + 1/1 + 1/1 + 2/2 + 0/2 + 2/2 = 21/23

