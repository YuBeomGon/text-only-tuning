import torch
from torch.utils.data import Dataset
from transformers import WhisperTokenizer


class DomainTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        model_name: str = "openai/whisper-base",
        language: str = "ko",
        task: str = "transcribe",
    ):
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name)
        self.texts = texts
        self.language = language
        self.task = task

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        # set_prefix_tokens configures the tokenizer so that encode()
        # automatically prepends [SOT, language, task, notimestamps]
        # and appends [EOT]. Do NOT manually prepend prefix_tokens.
        self.tokenizer.set_prefix_tokens(
            language=self.language, task=self.task
        )
        full_ids = self.tokenizer.encode(text)

        decoder_input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        labels = torch.tensor(full_ids[1:], dtype=torch.long)

        return {
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
