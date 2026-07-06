from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterator

import pandas as pd
from torch.utils.data import Sampler


class SubjectAwareBatchSampler(Sampler):
    def __init__(
        self,
        manifest: pd.DataFrame,
        batch_size: int,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 2:
            raise ValueError("SubjectAwareBatchSampler requires batch_size >= 2")
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

        subject_label_to_indices: dict[str, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
        for idx, row in manifest.reset_index(drop=True).iterrows():
            subject_id = str(row["subject_id"])
            label = int(row["label"])
            if label in (0, 1):
                subject_label_to_indices[subject_id][label].append(int(idx))

        self._paired_subjects = {
            subject_id: {
                0: list(label_map[0]),
                1: list(label_map[1]),
            }
            for subject_id, label_map in subject_label_to_indices.items()
            if label_map[0] and label_map[1]
        }
        self._all_indices = list(range(len(manifest)))

    def __len__(self) -> int:
        return math.ceil(len(self._all_indices) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        paired_pools = {
            subject_id: {
                0: list(label_map[0]),
                1: list(label_map[1]),
            }
            for subject_id, label_map in self._paired_subjects.items()
        }
        for subject_id in paired_pools:
            rng.shuffle(paired_pools[subject_id][0])
            rng.shuffle(paired_pools[subject_id][1])

        eligible_subjects = [subject_id for subject_id, pools in paired_pools.items() if pools[0] and pools[1]]
        subjects_per_batch = max(self.batch_size // 2, 1)
        yielded_indices: set[int] = set()
        batches: list[list[int]] = []

        while eligible_subjects:
            rng.shuffle(eligible_subjects)
            chosen_subjects = eligible_subjects[:subjects_per_batch]
            batch: list[int] = []
            next_eligible: list[str] = []

            for subject_id in eligible_subjects:
                pools = paired_pools[subject_id]
                if subject_id in chosen_subjects and pools[0] and pools[1]:
                    neg_idx = pools[0].pop(-1)
                    pos_idx = pools[1].pop(-1)
                    batch.extend([neg_idx, pos_idx])
                    yielded_indices.add(neg_idx)
                    yielded_indices.add(pos_idx)
                if pools[0] and pools[1]:
                    next_eligible.append(subject_id)

            eligible_subjects = next_eligible
            if batch:
                batches.append(batch)

        leftover = [idx for idx in self._all_indices if idx not in yielded_indices]
        rng.shuffle(leftover)

        for batch in batches:
            while len(batch) < self.batch_size and leftover:
                batch.append(leftover.pop(-1))
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

        while leftover:
            batch = leftover[: self.batch_size]
            leftover = leftover[self.batch_size :]
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch
