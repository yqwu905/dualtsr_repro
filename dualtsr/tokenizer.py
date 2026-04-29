from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass
class TokenizerState:
    vocab: list[str]
    pad_token: str = "<pad>"
    mask_token: str = "<mask>"
    unk_token: str = "<unk>"


class CharTokenizer:
    def __init__(
        self,
        vocab: Iterable[str],
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        unk_token: str = "<unk>",
    ) -> None:
        specials = [pad_token, mask_token, unk_token]
        ordered: list[str] = []
        for token in [*specials, *list(vocab)]:
            if token not in ordered:
                ordered.append(token)
        self.vocab = ordered
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.pad_id = self.stoi[pad_token]
        self.mask_id = self.stoi[mask_token]
        self.unk_id = self.stoi[unk_token]

    @classmethod
    def from_config(cls, cfg: dict) -> "CharTokenizer":
        tokenizer_cfg = cfg.get("tokenizer", {})
        if "alphabet" in tokenizer_cfg:
            vocab = list(tokenizer_cfg["alphabet"])
        else:
            vocab_path = tokenizer_cfg.get("vocab_path")
            if not vocab_path:
                raise ValueError("tokenizer.vocab_path or tokenizer.alphabet is required.")
            vocab = cls.read_vocab(vocab_path)
        return cls(
            vocab,
            pad_token=tokenizer_cfg.get("pad_token", "<pad>"),
            mask_token=tokenizer_cfg.get("mask_token", "<mask>"),
            unk_token=tokenizer_cfg.get("unk_token", "<unk>"),
        )

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        return cls(
            state["vocab"],
            pad_token=state.get("pad_token", "<pad>"),
            mask_token=state.get("mask_token", "<mask>"),
            unk_token=state.get("unk_token", "<unk>"),
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

    @staticmethod
    def write_vocab(path: str | Path, texts: Iterable[str]) -> None:
        chars = sorted({ch for text in texts for ch in text})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for ch in chars:
                f.write(ch + "\n")

    def state_dict(self) -> dict:
        return {
            "vocab": self.vocab,
            "pad_token": self.pad_token,
            "mask_token": self.mask_token,
            "unk_token": self.unk_token,
        }

    def encode(self, text: str, max_length: int) -> torch.Tensor:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text[:max_length]]
        ids.extend([self.pad_id] * (max_length - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Iterable[int], stop_at_pad: bool = True) -> str:
        chars: list[str] = []
        for idx in ids:
            token = self.itos.get(int(idx), self.unk_token)
            if token == self.pad_token and stop_at_pad:
                break
            if token in {self.pad_token, self.mask_token}:
                continue
            chars.append(token)
        return "".join(chars)

    def batch_decode(self, ids: torch.Tensor) -> list[str]:
        return [self.decode(row.tolist()) for row in ids.detach().cpu()]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

