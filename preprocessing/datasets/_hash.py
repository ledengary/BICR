"""Deterministic VLCB hash_id construction.

The hash_id is the join key between VLCB's local item table (questions, images,
ground-truth answers — reconstructed by the user from each source distributor)
and the model-outputs table shipped on Ledengary/VLCB. ANY drift in this
function silently breaks the join, so every per-source curator MUST import
`md5_hash_id` from this single module — never copy the implementation.

Inputs are concatenated verbatim with the literal `[SEP]` separator. No
normalization (no lowercasing, stripping, NFC/NFKC). Empty / missing fields
are coerced to the literal string `"N/A"`.
"""

import hashlib

SEP = "[SEP]"


def md5_hash_id(dataset: str, category: str, question: str, answer: str, image_key: str) -> str:
    """Compute the canonical VLCB hash_id.

    Args:
        dataset:   one of {"GQA", "POPE", "GMAI-MMBench", "MMMU_Pro_4",
                   "MMMU_Pro_10", "MME-Finance", "LLaVA-Wild"}
        category:  source-specific subcategory (GQA's `detailed`, POPE's category,
                   GMAI's `clinical_vqa_task`, etc.). Use `"N/A"` if missing.
        question:  verbatim question text from the source.
        answer:    verbatim ground-truth answer (single token for MC; free text
                   for open-ended).
        image_key: source-specific identifier that uniquely disambiguates
                   samples sharing question+answer text. See the per-source
                   curators for which field each dataset uses.

    Returns:
        32-character hex MD5 digest.
    """
    def _coerce(x):
        if x is None:
            return "N/A"
        s = str(x)
        return "N/A" if s.strip() == "" else s

    content = SEP.join([
        _coerce(dataset),
        _coerce(category),
        _coerce(question),
        _coerce(answer),
        _coerce(image_key),
    ])
    return hashlib.md5(content.encode()).hexdigest()
