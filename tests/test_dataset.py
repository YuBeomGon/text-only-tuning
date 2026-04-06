import pytest
from src.train.dataset import DomainTextDataset


def test_dataset_returns_input_ids():
    texts = ["This is a test sentence.", "Another sentence."]
    ds = DomainTextDataset(texts, model_name="openai/whisper-tiny")
    item = ds[0]
    assert "decoder_input_ids" in item
    assert "labels" in item


def test_dataset_length():
    texts = ["First.", "Second.", "Third."]
    ds = DomainTextDataset(texts, model_name="openai/whisper-tiny")
    assert len(ds) == 3


def test_dataset_labels_shifted():
    texts = ["Hello world."]
    ds = DomainTextDataset(texts, model_name="openai/whisper-tiny")
    item = ds[0]
    assert item["labels"].shape[0] > 0
