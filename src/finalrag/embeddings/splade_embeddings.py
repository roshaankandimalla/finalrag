import os

import torch
from pgvector import SparseVector
from transformers import AutoModelForMaskedLM, AutoTokenizer


DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 512
DEFAULT_TOP_K = 768


class SpladeEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = DEFAULT_MAX_LENGTH,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.top_k = top_k
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and os.getenv("SPLADE_DEVICE", "auto") != "cpu"
            else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.dimension = int(self.model.config.vocab_size)

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**encoded).logits
            activations = torch.log1p(torch.relu(logits))
            mask = encoded["attention_mask"].unsqueeze(-1)
            weights = (activations * mask).amax(dim=1)

        vectors = []
        for row in weights:
            nonzero = torch.nonzero(row > 0, as_tuple=False).squeeze(-1)
            if self.top_k and nonzero.numel() > self.top_k:
                _, positions = torch.topk(row[nonzero], self.top_k)
                nonzero = nonzero[positions]
            nonzero, order = torch.sort(nonzero)
            values = row[nonzero].float().cpu().tolist()
            indices = nonzero.cpu().tolist()
            vectors.append(SparseVector(dict(zip(indices, values)), self.dimension))
        return vectors

    def encode_query(self, query: str) -> SparseVector:
        value = query.strip()
        if not value:
            raise ValueError("Query cannot be empty")
        return self.encode_documents([value])[0]


def load_query_encoder():
    print("Loading SPLADE locally")
    return SpladeEncoder(
        os.getenv("SPLADE_MODEL", DEFAULT_MODEL),
        max_length=int(os.getenv("SPLADE_MAX_LENGTH", str(DEFAULT_MAX_LENGTH))),
        top_k=int(os.getenv("SPLADE_QUERY_TOP_K", "128")),
    )
