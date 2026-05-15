"""VLCB per-source curators.

Each module in this package builds a HuggingFace `Dataset` for one source
benchmark, with columns:

    question, answer, image, category, dataset, hash_id

`hash_id` is always computed via `_hash.md5_hash_id` so the bytes that go into
the join key are byte-for-byte identical across users.
"""
