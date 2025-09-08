import os

# Paths
DATASETS_PATH = '/home/cvillalba/data/data3/VQA/Tesis/TesisVQA/TesisVQA/VQA-MULTI-ESP/Datasets/'
IMAGES_PATH =  DATASETS_PATH + 'Images/images/'
NEW_DATASETS_FOLDER = '/home/cvillalba/data/data3/VQA/Tesis/TesisVQA/TesisVQA/VQA-MULTI-ESP/modelos/'
METRICS_LOG = 'metrics_multimodal_log.csv'

print(f"DATASETS_PATH: {DATASETS_PATH}"
      f"\nIMAGES_PATH: {IMAGES_PATH}"
      f"\nNEW_DATASETS_FOLDER: {NEW_DATASETS_FOLDER}")
# Dataset files
DATA_TRAIN = 'VizWiz_train_new_sample_multiM_corregido.csv'
DATA_VALI = 'VizWiz_test_new_sample_multiM_corregido.csv'
DATA_ANS = 'answer_space_trad_corregido.txt'


