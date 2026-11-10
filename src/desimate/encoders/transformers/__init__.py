import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from desimate.encoders import Vectorizer


class TransformersVectorizers(Vectorizer):
    def __init__(self, model_path: str, show_progress=True):
        super().__init__(show_progress)
        self.model = model_path

    def run(self, text: list[str], batch_size=128) -> np.ndarray:
        model = SentenceTransformer(self.model)
        task = self.progress.add_task("Embedding", total=len(text) // batch_size + 1)

        for i in range(len(text) // batch_size + 1):
            batch = text[i * batch_size : (i + 1) * batch_size]
            with torch.no_grad():
                embeddings = model.encode(
                    batch, batch_size=batch_size, show_progress_bar=False
                )
                yield embeddings
                self.progress.update(task, advance=1)
