import os
from dataclasses import dataclass
from typing import List
from PIL import Image
import torch
from torchvision.transforms import Normalize
from transformers import AutoTokenizer, AutoFeatureExtractor

from .config import IMAGES_PATH

@dataclass
class MultimodalCollator:
    """
    Collator personalizado para datasets multimodales de VQA.
    Se encarga de tokenizar el texto y procesar las imágenes para que puedan ser usadas por el modelo.
    """
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor

    def tokenize_text(self, texts: List[str]):
        """
        Tokeniza una lista de preguntas usando el tokenizer de HuggingFace.

        Args:
            texts (List[str]): Lista de preguntas en texto.

        Returns:
            dict: Diccionario con input_ids, token_type_ids y attention_mask.
        """
        encoded_text = self.tokenizer(
            text=texts,
            padding='longest',
            max_length=24,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
        }

    def preprocess_images(self, images: List[str], partition):
        """
        Procesa una lista de imágenes, aplicando normalización y convirtiéndolas en tensores.

        Args:
            images (List[str]): Lista de nombres de archivos de imagen.
            partition (str): 'training' o 'validation', para aplicar diferentes transformaciones si se desea.

        Returns:
            dict: Diccionario con los tensores de imágenes bajo la clave 'pixel_values'.
        """
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        normalize = Normalize(mean=mean, std=std)
        processed_images = self.preprocessor(
            images=[Image.open(os.path.join(IMAGES_PATH, image_id)).convert('RGB') for image_id in images],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }

    def __call__(self, raw_batch_dict, partition):
        """
        Prepara un batch para el modelo, tokenizando el texto y procesando las imágenes.

        Args:
            raw_batch_dict (dict o list): Batch de datos crudos del dataset.
            partition (str): 'training' o 'validation'.

        Returns:
            dict: Diccionario con los tensores listos para el modelo (input_ids, token_type_ids, attention_mask, pixel_values, labels).
        """
        return {
            **self.tokenize_text(
                raw_batch_dict['question_trad'] if isinstance(raw_batch_dict, dict)
                else [i['question_trad'] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict['image_id'] if isinstance(raw_batch_dict, dict)
                else [i['image_id'] for i in raw_batch_dict], partition
            ),
            'labels': torch.tensor(
                raw_batch_dict['label'] if isinstance(raw_batch_dict, dict)
                else [i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),
        }