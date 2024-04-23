from sentence_transformers import SentenceTransformer, models
from transformers.trainer import *

from ..trainer import BiTrainer


def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normlized: bool=True):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    if normlized:
        normlize_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normlize_layer], device='cpu')
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir)


class InfoBatchBiTrainer(BiTrainer):
    def _get_train_sampler(self):
        if self.train_dataset is None or not len(self.train_dataset):
            return None
        else:
            return self.train_dataset.sampler
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss
        #print(f"query_indices shape: {query_indices.shape}")
        #print(f"loss shape: {loss}")
        query_indices = outputs.query_indices
        loss = self.train_dataset.update(loss, query_indices)
        #loss = loss.mean()

        return (loss, outputs) if return_outputs else loss
