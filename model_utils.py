"""Model utilities for Next Word Predictor app.

Provides a simple bigram (n-gram) predictor built from an internal
toy corpus and a `transformer_predict_next` that will use Hugging
Face `transformers` if available; otherwise it falls back to the
bigram predictor. The bigram predictor is fast and deterministic and
will make the app respond to input such as "I am learning".
"""
from typing import List, Tuple, Dict
import re

# Toy English corpus (kept small for responsiveness)
CORPUS_TEXT = """
Language is a system of communication used by humans to express ideas and emotions.
Machine learning models learn patterns from data and make predictions based on probability.
Artificial intelligence is widely used in applications such as search engines, assistants,
and recommendation systems. Learning from examples allows models to generalize to new inputs.
People read books, write code, and communicate using natural language every day.
Technology continues to evolve as data and computational power increase.
"""


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    # simple tokenization: split on whitespace and strip punctuation
    tokens = [re.sub(r"(^\W+|\W+$)", "", t) for t in text.split()]
    return [t for t in tokens if t]


def _build_bigram_counts(corpus: str) -> Dict[Tuple[str, str], int]:
    toks = _tokenize(corpus)
    counts: Dict[Tuple[str, str], int] = {}
    for a, b in zip(toks, toks[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


_BIGRAM_COUNTS = _build_bigram_counts(CORPUS_TEXT)


def ngram_predict_next(text: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Predict next tokens using bigram frequencies from the toy corpus.

    Returns a list of (word, probability) sorted by probability.
    """
    if not text or not text.strip():
        return []
    toks = _tokenize(text)
    if not toks:
        return []
    last = toks[-1]
    # collect candidates following `last`
    candidates: Dict[str, int] = {}
    for (a, b), c in _BIGRAM_COUNTS.items():
        if a == last:
            candidates[b] = candidates.get(b, 0) + c

    if not candidates:
        return []

    total = sum(candidates.values())
    sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(w, cnt / total) for w, cnt in sorted_cands]


# Transformer state (lazy loaded)
_HF_AVAILABLE = False
_HF_TOKENIZER = None
_HF_MODEL = None
_HF_DEVICE = "cpu"

def _try_load_hf(model_name: str = "distilgpt2") -> None:
    global _HF_AVAILABLE, _HF_TOKENIZER, _HF_MODEL, _HF_DEVICE
    if _HF_AVAILABLE:
        return
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        _HF_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _HF_MODEL = AutoModelForCausalLM.from_pretrained(model_name)
        _HF_MODEL.eval()
        _HF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _HF_MODEL.to(_HF_DEVICE)
        except Exception:
            # best-effort move; continue on CPU if GPU not available
            _HF_MODEL.to("cpu")
            _HF_DEVICE = "cpu"
        _HF_AVAILABLE = True
    except Exception:
        _HF_AVAILABLE = False


def transformer_predict_next(
    text: str,
    top_k: int = 5,
    prefer_generation: bool = True,
    num_beams: int = 4,
    max_new_tokens: int = 10,
) -> List[Tuple[str, float]]:
    """Predict next tokens using a transformer model if available.

    If Hugging Face `transformers` and a model are not available, this
    function falls back to `ngram_predict_next`.
    """
    if not text or not text.strip():
        return []

    _try_load_hf()
    if not _HF_AVAILABLE:
        # fallback to n-gram
        return ngram_predict_next(text, top_k=top_k)

    import torch

    # Prefer generation to obtain whole-word continuations.
    # Use beam search to get `top_k` candidate continuations and
    # extract the first word after the prompt from each generated seq.
    try:
        tokenized = _HF_TOKENIZER(text, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(_HF_DEVICE)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(_HF_DEVICE)

        num_beams = min(max(num_beams, 2), 8)
        num_return = min(top_k, num_beams)

        gen = _HF_MODEL.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=getattr(_HF_TOKENIZER, "eos_token_id", None),
        )

        sequences = gen.sequences.tolist()
        seq_scores = None
        # sequences_scores is available for beam search outputs
        if hasattr(gen, "sequences_scores") and gen.sequences_scores is not None:
            seq_scores = gen.sequences_scores.tolist()

        prefix = _HF_TOKENIZER.decode(input_ids[0].tolist()).strip()

        candidates: List[Tuple[str, float]] = []
        if seq_scores is not None:
            # extract first word from each generated sequence and map scores
            for seq, score in zip(sequences, seq_scores):
                decoded = _HF_TOKENIZER.decode(seq).strip()
                # remove prompt prefix
                if decoded.startswith(prefix):
                    rem = decoded[len(prefix) :].strip()
                else:
                    rem = decoded
                if not rem:
                    continue
                first_word = re.split(r"\s+", rem)[0]
                if not re.search(r"[A-Za-z0-9]", first_word):
                    continue
                candidates.append((first_word, float(score)))

            if not candidates:
                return ngram_predict_next(text, top_k)

            # convert log-prob scores to normalized probabilities
            scores_t = torch.tensor([c for _, c in candidates], dtype=torch.float)
            probs = torch.softmax(scores_t, dim=0).tolist()
            # take top_k unique first words preserving order by prob
            seen = set()
            results: List[Tuple[str, float]] = []
            for (w, _), p in zip(candidates, probs):
                if w in seen:
                    continue
                seen.add(w)
                results.append((w, float(p)))
                if len(results) >= top_k:
                    break
            return results

    except Exception:
        # fall back to simpler token-level method on any failure
        pass

    # fallback: token-level scoring (previous approach)
    tokenized = _HF_TOKENIZER(text, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(_HF_DEVICE)
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(_HF_DEVICE)
    with torch.no_grad():
        outputs = _HF_MODEL(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    next_token_logits = logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=0)
    # request a larger candidate set then filter to meaningful tokens
    k_cand = min(max(top_k * 3, top_k), probs.size(0))
    top_probs, top_indices = torch.topk(probs, k=k_cand)

    results: List[Tuple[str, float]] = []
    for idx, p in zip(top_indices.tolist(), top_probs.tolist()):
        token_str = _HF_TOKENIZER.decode([int(idx)])
        token_str = token_str.strip()
        # drop tokens that are only punctuation or whitespace
        if not re.search(r"[A-Za-z0-9]", token_str):
            continue
        results.append((token_str, float(p)))
        if len(results) >= top_k:
            break

    if not results:
        return ngram_predict_next(text, top_k)

    # normalize probabilities over the filtered set
    total = sum(p for _, p in results)
    return [(w, p / total) for w, p in results]


__all__ = ["ngram_predict_next", "transformer_predict_next"]


def hf_status() -> dict:
    """Return Hugging Face availability and device info."""
    # report whether required packages are installed, and whether the
    # HF model has actually been loaded into memory. We avoid forcing
    # a model download at UI startup.
    try:
        import transformers  # type: ignore
        import torch as _torch  # type: ignore
        packages_installed = True
    except Exception:
        packages_installed = False

    model_loaded = bool(_HF_AVAILABLE and _HF_MODEL is not None)
    device = _HF_DEVICE if model_loaded else None
    return {
        "packages_installed": packages_installed,
        "model_loaded": model_loaded,
        "device": device,
    }


__all__.append("hf_status")
