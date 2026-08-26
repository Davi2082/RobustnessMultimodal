"""HotFlip: first-order token substitution.

Re-implementation of the substitution rule of Ebrahimi et al. (ACL 2018): rank
a swap w -> v at position i by (e_v - e_w)^T grad_{e_i} L. The beam search of
the original is replaced by a greedy step, so one substitution can be taken per
shared backward pass.

Pure algorithm: no fusion, dataset or result-file knowledge.
"""

import re

import torch

BYTE_FALLBACK = re.compile(r"<0x[0-9A-Fa-f]{2}>")
WORD_PREFIX = "▁"  # SentencePiece marks a word start with U+2581


def whole_word_positions(tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    """Positions holding a complete word, i.e. safe to substitute wholesale.

    A piece qualifies when it opens a word and the next one does too, so it is
    not a fragment of a multi-piece word; replacing a fragment is unreadable.
    """
    pieces = tokenizer.convert_ids_to_tokens(input_ids.reshape(-1).tolist())
    specials = set(tokenizer.all_special_tokens)
    starts = [
        piece.startswith(WORD_PREFIX) and piece[len(WORD_PREFIX):].isalpha()
        for piece in pieces
    ]

    complete = []
    for position, piece in enumerate(pieces):
        following = position + 1
        boundary = (
            following >= len(pieces)
            or starts[following]
            or pieces[following] in specials
        )
        complete.append(starts[position] and boundary and piece not in specials)

    return torch.tensor(complete, dtype=torch.bool, device=input_ids.device)


def effective_embedding_table(
    embedding: torch.nn.Module, vocab_size: int, device: torch.device, chunk: int = 8192
) -> torch.Tensor:
    """Materialise the embedding actually used at inference (LoRA included).

    ``embedding.weight`` is the base table and would rank candidates wrongly;
    running ids through the module applies the active adapters.
    """
    rows = []
    with torch.no_grad():
        for start in range(0, vocab_size, chunk):
            ids = torch.arange(
                start, min(start + chunk, vocab_size), device=device
            ).unsqueeze(0)
            rows.append(embedding(ids).squeeze(0).detach())
    return torch.cat(rows, dim=0)


def candidate_mask(
    tokenizer, vocab_size: int, ascii_only: bool, word_level: bool
) -> torch.Tensor:
    """Vocabulary positions HotFlip may substitute in.

    ``word_level`` keeps only word-initial alphabetic pieces, so a flip swaps a
    whole word rather than rewriting a fragment.
    """
    allowed = torch.ones(vocab_size, dtype=torch.bool)
    for token_id in tokenizer.all_special_ids:
        if 0 <= token_id < vocab_size:
            allowed[token_id] = False

    if word_level:
        for token_id in range(vocab_size):
            piece = tokenizer.convert_ids_to_tokens(token_id)
            # The word marker itself is U+2581, so ASCII is checked on the
            # word body rather than on the raw piece.
            body = piece[len(WORD_PREFIX):] if piece else ""
            if (
                piece is None
                or not piece.startswith(WORD_PREFIX)
                or not body.isalpha()
                or not body.isascii()
            ):
                allowed[token_id] = False

    if ascii_only:
        for token_id in range(vocab_size):
            piece = tokenizer.convert_ids_to_tokens(token_id)
            if piece is None:
                allowed[token_id] = False
                continue
            # Byte-fallback pieces such as <0x0A> read as printable ASCII but
            # decode to raw control bytes, which are neither readable
            # substitutions nor safe to serialise.
            if BYTE_FALLBACK.fullmatch(piece):
                allowed[token_id] = False
                continue
            text = piece.replace("▁", "")
            if not text or not text.isascii() or not text.isprintable():
                allowed[token_id] = False
    return allowed


class EmbeddingGradient:
    """Capture the text embedding tensor so its gradient can be read back.

    The frozen table with integer input carries no graph, so the hook swaps in
    a leaf copy that does.
    """

    def __init__(self, embedding: torch.nn.Module):
        self.output: torch.Tensor | None = None
        self._handle = embedding.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        leaf = output.detach().clone().requires_grad_(True)
        self.output = leaf
        return leaf

    def remove(self):
        self._handle.remove()
