ML in production. Homework 1.


RUN train:
   cd ml_project
   PYTHONPATH='.' python src/train_pipeline.py 

RUN tests:
   cd ml_project
   PYTHONPATH='src' pytest

1. To fetch data:
   1. Use Kaggle public API (https://www.kaggle.com/docs/api). 
   2. Type the following commands:
      - cd ml_project/data/raw
      - kaggle datasets download -d cherngs/heart-disease-cleveland-uci
      - unzip heart-disease-cleveland-uci.zip 
      - rm heart-disease-cleveland-uci.zip 
2. EDA was carried out, you can find the notebook with results in 'notebooks/'. 
   No scripts were used to generate a report.

3. There is a script to run training pipeline.
4. The project has a module structure.
5. Logging is used in the project.
6. There are tests for some modules.
7. All processes are configured using .yaml configs.
8. Dataclasses are used for configuration dictionaries.
9. All dependencies are mentioned in requirements.txt.
10. Hydra is used.