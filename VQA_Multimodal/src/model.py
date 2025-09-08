import torch
import torch.nn as nn
from transformers import AutoModel

class MultimodalVQAModel(nn.Module):
    """
        Modelo Multimodal para Visual Question Answering (VQA).
        Combina un encoder de texto y un encoder de imagen (ambos transformers),
        fusiona sus representaciones y las pasa por una capa lineal para clasificación multiclase.
    """
    def __init__(self, answer_space, intermediate_dim=512,
                 pretrained_text_name='dccuchile/bert-base-spanish-wwm-uncased',
                 pretrained_image_name='google/vit-base-patch16-224-in21k'):
        """
            Inicializa el modelo multimodal.

            Args:
                answer_space (list): Lista de posibles respuestas (espacio de respuestas).
                intermediate_dim (int): Dimensión de la capa intermedia de fusión.
                pretrained_text_name (str): Nombre del modelo transformer de texto.
                pretrained_image_name (str): Nombre del modelo transformer de imagen.
        """
        super().__init__()
        self.num_labels = len(answer_space)
        self.text_encoder = AutoModel.from_pretrained(pretrained_text_name)
        self.image_encoder = AutoModel.from_pretrained(pretrained_image_name)
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(intermediate_dim, self.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, pixel_values, attention_mask=None, token_type_ids=None, labels=None):
        """
        Realiza el forward pass del modelo.

        Args:
            input_ids (torch.LongTensor): IDs de los tokens de la pregunta.
            pixel_values (torch.FloatTensor): Tensor de imágenes procesadas.
            attention_mask (torch.LongTensor, opcional): Máscara de atención para el texto.
            token_type_ids (torch.LongTensor, opcional): IDs de tipo de token para el texto.
            labels (torch.LongTensor, opcional): Etiquetas verdaderas para calcular la loss.

        Returns:
            dict: Diccionario con 'logits' y, si labels está presente, 'loss'.
        """
        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        fused_output = self.fusion(
            torch.cat([encoded_text['pooler_output'], encoded_image['pooler_output']], dim=1)
        )
        logits = self.classifier(fused_output)
        out = {"logits": logits}
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        return out

def create_multimodal_collator_and_model(answer_space, text_model, image_model, device):
    """
    Crea el collator y el modelo multimodal VQA listos para usar.

    Args:
        answer_space (list): Lista de posibles respuestas.
        text_model (str): Nombre del modelo transformer de texto.
        image_model (str): Nombre del modelo transformer de imagen.
        device (torch.device): Dispositivo donde se cargará el modelo.

    Returns:
        tuple: (collator, modelo) listos para entrenamiento o inferencia.
    """
    from transformers import AutoTokenizer, AutoFeatureExtractor
    from .collator import MultimodalCollator

    tokenizer = AutoTokenizer.from_pretrained(text_model)
    preprocessor = AutoFeatureExtractor.from_pretrained(image_model)
    collator = MultimodalCollator(tokenizer=tokenizer, preprocessor=preprocessor)
    model = MultimodalVQAModel(answer_space, pretrained_text_name=text_model, pretrained_image_name=image_model).to(device)
    return collator, model