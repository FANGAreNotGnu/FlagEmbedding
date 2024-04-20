from sentence_transformers import SentenceTransformer, models
from transformers.trainer import *

from .trainer import BiTrainer


def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normlized: bool=True):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    if normlized:
        normlize_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normlize_layer], device='cpu')
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir)


class SoftBiTrainer(BiTrainer):
    def _get_train_sampler(self):
        if self.train_dataset is None or not len(self.train_dataset):
            return None
        else:
            return self.train_dataset.sampler

