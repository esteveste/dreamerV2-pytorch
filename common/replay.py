import datetime
import io
import pathlib
import time
import uuid

import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

import common


class Replay:

    def __init__(self, directory, limit=None):
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory
        self._limit = limit
        self._step = sum(int(
            str(n).split('-')[-1][:-4]) - 1 for n in directory.glob('*.npz'))
        self._episodes = load_episodes(directory, limit)

    @property
    def total_steps(self):
        return self._step

    @property
    def num_episodes(self):
        return len(self._episodes)

    @property
    def num_transitions(self):
        return sum(self._length(ep) for ep in self._episodes.values())

    def add(self, episode):
        length = self._length(episode)
        self._step += length
        if self._limit:
            total = 0
            for key, ep in reversed(sorted(
                    self._episodes.items(), key=lambda x: x[0])):
                if total <= self._limit - length:
                    total += self._length(ep)
                else:
                    del self._episodes[key]
        filename = save_episodes(self._directory, [episode])[0]
        self._episodes[str(filename)] = episode

    def dataset(self, batch, length, oversample_ends, pin_memory=True, **kwargs):
        generator = lambda: sample_episodes(
            self._episodes, length, oversample_ends)

        class ReplayDataset(IterableDataset):
            def __iter__(self):
                return generator()

        dataset = ReplayDataset()

        # FIXME we cant use workers, since it forks, and the _episodes variable will be outdated
        dataset = DataLoader(dataset, batch, pin_memory=pin_memory, drop_last=True, worker_init_fn=seed_worker,
                             **kwargs)  # ,num_workers=2)
        return dataset

    def _length(self, episode):
        return len(episode['reward']) - 1


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    filenames = []
    for episode in episodes:
        identifier = str(uuid.uuid4().hex)
        length = len(episode['reward']) - 1
        filename = directory / f'{timestamp}-{identifier}-{length}.npz'
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open('wb') as f2:  # wonder why not directly
                f2.write(f1.read())
        filenames.append(filename)
    return filenames


def sample_episodes(episodes, length=None, balance=False, seed=0):
    random = np.random  # .RandomState(seed)
    while True:
        episode = random.choice(list(episodes.values()))
        if length:
            total = len(next(iter(episode.values())))
            available = total - length
            if available < 1:
                print(f'Skipped short episode of length {total}.')
                continue
            if balance:
                index = min(random.randint(0, total), available)
            else:
                index = int(random.randint(0, available + 1))
            # convert all directly to float and torch
            dtype = torch.float16 if common.ENABLE_FP16 else torch.float32
            episode = {k: torch.as_tensor(v[index: index + length], dtype=dtype) for k, v in episode.items()}
            # its T, H,W,C to T,C,H,W
            episode['image'] = episode['image'].permute(0, 3, 1, 2)
        yield episode


def load_episodes(directory, limit=None):
    directory = pathlib.Path(directory).expanduser()
    episodes = {}
    total = 0
    for filename in reversed(sorted(directory.glob('*.npz'))):
        try:
            with filename.open('rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f'Could not load episode: {e}')
            continue
        episodes[str(filename)] = episode
        total += len(episode['reward']) - 1
        if limit and total >= limit:
            break
    return episodes


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
