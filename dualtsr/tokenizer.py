from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import torch

from dualtsr.registry import load_class


class BaseTokenizer(ABC):
    """Shared interface for swappable tokenizers.

    Subclasses define how text is split into / joined from atomic tokens. The
    rest of the pipeline only relies on ``encode``/``decode``/``batch_decode``,
    ``vocab_size`` and the ``pad_id``/``mask_id``/``unk_id`` ids.
    """

    #: Config key under ``tokenizer`` that holds an inline vocabulary list.
    VOCAB_KEY: str = "vocab"

    def __init__(
        self,
        vocab: Iterable[str],
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        unk_token: str = "<unk>",
        eos_token: str = "<eos>",
    ) -> None:
        specials = [pad_token, mask_token, unk_token, eos_token]
        ordered: list[str] = []
        for token in [*specials, *list(vocab)]:
            if token not in ordered:
                ordered.append(token)
        self.vocab = ordered
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.pad_id = self.stoi[pad_token]
        self.mask_id = self.stoi[mask_token]
        self.unk_id = self.stoi[unk_token]
        self.eos_id = self.stoi[eos_token]

    # --- splitting / joining (subclass-specific) ---------------------------
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Split ``text`` into atomic tokens."""

    @abstractmethod
    def detokenize(self, tokens: list[str]) -> str:
        """Join atomic tokens back into a string."""

    @staticmethod
    @abstractmethod
    def atomize(text: str) -> list[str]:
        """Vocab-independent split used when building a vocabulary from a corpus."""

    # --- encode / decode ---------------------------------------------------
    def encode(self, text: str, max_length: int) -> torch.Tensor:
        if max_length < 1:
            raise ValueError("max_length must be positive")
        ids = [self.stoi.get(tok, self.unk_id) for tok in self.tokenize(text)[: max_length - 1]]
        ids.append(self.eos_id)
        ids.extend([self.pad_id] * (max_length - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Iterable[int], stop_at_pad: bool = True) -> str:
        tokens: list[str] = []
        for idx in ids:
            token = self.itos.get(int(idx), self.unk_token)
            if token in {self.pad_token, self.eos_token} and stop_at_pad:
                break
            if token in {self.pad_token, self.mask_token, self.eos_token}:
                continue
            tokens.append(token)
        return self.detokenize(tokens)

    def batch_decode(self, ids: torch.Tensor) -> list[str]:
        return [self.decode(row.tolist()) for row in ids.detach().cpu()]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # --- (de)serialization -------------------------------------------------
    def _extra_state(self) -> dict:
        """Subclass-specific fields to persist beyond the shared ones."""
        return {}

    def state_dict(self) -> dict:
        state = {
            "class_path": f"{type(self).__module__}:{type(self).__name__}",
            "vocab": self.vocab,
            "pad_token": self.pad_token,
            "mask_token": self.mask_token,
            "unk_token": self.unk_token,
            "eos_token": self.eos_token,
        }
        state.update(self._extra_state())
        return state

    @classmethod
    def _extra_kwargs_from_config(cls, tokenizer_cfg: dict) -> dict:
        return {}

    @classmethod
    def _extra_kwargs_from_state(cls, state: dict) -> dict:
        return {}

    @classmethod
    def from_config(cls, cfg: dict) -> "BaseTokenizer":
        tokenizer_cfg = cfg.get("tokenizer", {})
        if cls.VOCAB_KEY in tokenizer_cfg:
            vocab = list(tokenizer_cfg[cls.VOCAB_KEY])
        elif tokenizer_cfg.get("vocab_path"):
            vocab = cls.read_vocab(tokenizer_cfg["vocab_path"])
        else:
            raise ValueError(f"tokenizer.{cls.VOCAB_KEY} or tokenizer.vocab_path is required.")
        return cls(
            vocab,
            pad_token=tokenizer_cfg.get("pad_token", "<pad>"),
            mask_token=tokenizer_cfg.get("mask_token", "<mask>"),
            unk_token=tokenizer_cfg.get("unk_token", "<unk>"),
            eos_token=tokenizer_cfg.get("eos_token", "<eos>"),
            **cls._extra_kwargs_from_config(tokenizer_cfg),
        )

    @classmethod
    def from_state(cls, state: dict) -> "BaseTokenizer":
        return cls(
            state["vocab"],
            pad_token=state.get("pad_token", "<pad>"),
            mask_token=state.get("mask_token", "<mask>"),
            unk_token=state.get("unk_token", "<unk>"),
            # Old checkpoints used PAD as the only terminator. Preserve their
            # vocabulary shape while new checkpoints receive an explicit EOS.
            eos_token=state.get("eos_token", state.get("pad_token", "<pad>")),
            **cls._extra_kwargs_from_state(state),
        )

    @staticmethod
    def read_vocab(path: str | Path) -> list[str]:
        tokens: list[str] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                token = line.rstrip("\n")
                if token:
                    tokens.append(token)
        return tokens

    @classmethod
    def write_vocab(cls, path: str | Path, texts: Iterable[str]) -> None:
        tokens = sorted({tok for text in texts for tok in cls.atomize(text)})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for tok in tokens:
                f.write(tok + "\n")


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer."""

    VOCAB_KEY = "alphabet"

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def detokenize(self, tokens: list[str]) -> str:
        return "".join(tokens)

    @staticmethod
    def atomize(text: str) -> list[str]:
        return list(text)


class WordTokenizer(BaseTokenizer):
    """Word-level tokenizer backed by a vocabulary file or inline word list.

    Splits on whitespace by default; pass ``separator`` to split on a fixed
    delimiter instead. Out-of-vocabulary words map to ``<unk>``.
    """

    VOCAB_KEY = "words"

    def __init__(
        self,
        vocab: Iterable[str],
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        unk_token: str = "<unk>",
        eos_token: str = "<eos>",
        separator: str | None = None,
    ) -> None:
        super().__init__(
            vocab,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            eos_token=eos_token,
        )
        self.separator = separator

    def tokenize(self, text: str) -> list[str]:
        return text.split(self.separator) if self.separator else text.split()

    def detokenize(self, tokens: list[str]) -> str:
        return (self.separator or " ").join(tokens)

    @staticmethod
    def atomize(text: str) -> list[str]:
        return text.split()

    def _extra_state(self) -> dict:
        return {"separator": self.separator}

    @classmethod
    def _extra_kwargs_from_config(cls, tokenizer_cfg: dict) -> dict:
        return {"separator": tokenizer_cfg.get("separator")}

    @classmethod
    def _extra_kwargs_from_state(cls, state: dict) -> dict:
        return {"separator": state.get("separator")}


def build_tokenizer(config: dict) -> BaseTokenizer:
    """Instantiate the tokenizer from ``tokenizer.class_path`` (default CharTokenizer)."""
    tokenizer_cfg = config.get("tokenizer", {})
    class_path = tokenizer_cfg.get("class_path", "dualtsr.tokenizer:CharTokenizer")
    cls = load_class(class_path)
    return cls.from_config(config)


def tokenizer_from_state(state: dict) -> BaseTokenizer:
    """Reconstruct a tokenizer from a saved ``state_dict`` using its ``class_path`` marker."""
    class_path = state.get("class_path")
    cls = load_class(class_path) if class_path else CharTokenizer
    return cls.from_state(state)
