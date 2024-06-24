import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as TorchDataset
from tqdm import tqdm
from transformers import AutoTokenizer

UINT32_MAX = 2**32 - 1  # 4294967295


def load_datasets(
    data_dir: Path,
    block_size: int,
    seed: int = 42,
    sanity_tokenizer: Path | None = None,
    use_clipped_val: bool = False,
    mask_bos_loss: bool = False,
    ensure_bos_token: bool = True,
    ensure_eos_token: bool = False,
):
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    print("Loading datasets...")
    # tokenizer vocab prints too much clutter
    print({k: v for k, v in metadata.items() if k != "tokenizer_vocab"})

    # do some sanity checks to make sure the user actually loaded the correct dataset
    data_tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"])
    if sanity_tokenizer:
        # assert tokenizer == Path(metadata["tokenizer"])
        print("sanity tokenizer:", sanity_tokenizer)
        sanity_tokenizer = AutoTokenizer.from_pretrained(sanity_tokenizer)
        if not sanity_tokenizer.get_vocab() == data_tokenizer.get_vocab():
            print("Sanity tokenizer vocab does not match data tokenizer vocab. Using data tokenizer.")
            print(
                "token difference:",
                len(set(sanity_tokenizer.get_vocab()) ^ set(data_tokenizer.get_vocab())),
            )

        assert sanity_tokenizer.pad_token_id == data_tokenizer.pad_token_id
        assert sanity_tokenizer.bos_token_id == data_tokenizer.bos_token_id
        assert sanity_tokenizer.eos_token_id == data_tokenizer.eos_token_id
        assert sanity_tokenizer.unk_token_id == data_tokenizer.unk_token_id

        assert data_tokenizer.eos_token_id == metadata["eos_token_id"]
        assert data_tokenizer.bos_token_id == metadata["bos_token_id"]

    metadata_tokenizer_vocab = metadata["tokenizer_vocab"]
    assert len(data_tokenizer.get_vocab()) == len(metadata_tokenizer_vocab)
    assert metadata_tokenizer_vocab == data_tokenizer.get_vocab()

    train_path = data_dir / metadata["train_data_file"]
    val_path = data_dir / metadata["dev_data_file"]
    train_index_path = data_dir / metadata["train_index_file"]
    val_idx_path = data_dir / metadata["dev_index_file"]

    if use_clipped_val:
        val_path = data_dir / metadata["val_clipped_data_file"]
        val_idx_path = data_dir / metadata["val_clipped_index_file"]

    assert train_path.exists()
    assert val_path.exists()
    assert train_index_path.exists()
    assert val_idx_path.exists()

    if data_tokenizer.pad_token_id is None:
        print("WARNING: tokenizer does not have a pad token. We will use EOS if padding is necessary. This will cause issues if you are also using EOS tokens in the pretraining.")
    common_kwargs = dict(
        block_size=block_size,
        data_dtype=np.dtype(metadata["data_dtype"]),
        doc_offset_dtype=np.dtype(metadata["doc_offset_dtype"]),
        output_dtype=np.int64,  # expected by torch (int64 => long)
        bos_token=metadata["bos_token_id"],
        eos_token=metadata["eos_token_id"],
        mask_bos_loss=mask_bos_loss,
        ensure_bos_token=ensure_bos_token,  # Prepend BOS token if not present
        ensure_eos_token=ensure_eos_token,  # Append EOS token if not present
        pad_token=data_tokenizer.pad_token_id
        or data_tokenizer.eos_token_id,  # NOTE: [KD] this is a hack that's ok for llama2 b/c it doesnt have a pad token and we don't use EOS for training
    )
    print("loading datasets internal")
    train_data = VeryCoolDataset(
        train_path,
        doc_offsets_file=train_index_path,
        shuffle=True,
        access="docs-iid-packed",
        **common_kwargs,
    )
    val_data = VeryCoolDataset(
        val_path,
        doc_offsets_file=val_idx_path,
        shuffle=False,
        access="single-doc-padded",
        **common_kwargs,
    )
    return train_data, val_data


def get_dataloaders(
    data_dir: Path,
    block_size: int,
    batch_size: int,
    workers: int,
    tokenizer_path: Path | None = None,
    val_batch_size: int | None = None,
    use_clipped_val: bool = False,
    resume_from_sample_idx: int | None = None,
    resume_from_epoch: int | None = None,
):
    train_data, val_data = load_datasets(
        data_dir=data_dir,
        block_size=block_size,
        sanity_tokenizer=tokenizer_path,
        use_clipped_val=use_clipped_val,
    )
    if resume_from_sample_idx is not None:
        assert train_data.training_order is not None
        if resume_from_epoch is None:
            print(f"Resuming dataset from sample idx {resume_from_sample_idx}")
            train_data.training_order = train_data.training_order[resume_from_sample_idx:]
        else:
            train_data.training_order = train_data.get_reproducible_shuffled_training_order_for_epoch(resume_from_epoch)[
                resume_from_sample_idx:
            ]

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
        # persistent_workers=True, # Deactivate because we don't do more than one epoch anyways
        shuffle=False,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=val_batch_size or batch_size,
        num_workers=workers,
        pin_memory=True,
        # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
        # persistent_workers=True,
        shuffle=False,
        drop_last=False,
    )

    return train_dataloader, val_dataloader


class VeryCoolDataset(TorchDataset):
    """
    In `data_tokenization.py` we store the tokenized concatenated data (`data_file`) AND a file containing the indices in `data_file` where a new sample starts.
    We use this `index_file` to sample from the data so that the beginning of each sample aligns with the start of an actual sample in the data.

    This allows us to change the `block_size` with ZERO overhead at runtime (no expensive re-tokenization / chunking, looking at you HF example scripts).
    However, we discard the remainder of each sample after the first `block_size` tokens (this could be optimized).

    *adapted heavily from lit-gpt pretrain/openwebtext.py*
    """

    def __init__(
        self,
        data_file: Path,
        doc_offsets_file: Path,
        block_size: int,
        access: Literal[
            "contiguous",
            "docs-iid-packed",
            "docs-iid-packed-no-attn-fix",
            "single-doc-padded",
            "single-doc-full",
        ] = "single-doc-padded",
        unk_token: int = 0,
        bos_token: int = 1,
        eos_token: int = 2,
        pad_token: int = -1,  # by default, chunked_cross_entropy ignores -1. llama2 does not have pad token in vocab
        ignore_index: int = -1,  # for cross entropy loss
        mask_bos_loss: bool = False,
        ensure_bos_token: bool = False,
        mask_eos_loss: bool = False,
        ensure_eos_token: bool = False,
        shuffle: bool = False,
        data_dtype: np.dtype = np.uint16,  # supports vocab size up to 65k
        doc_offset_dtype: np.dtype = np.uint64,  # supports up to 2**64 = a lot of tokens
        output_dtype: np.dtype = np.int64,  # for safety
    ):
        super().__init__()
        self.data_file = data_file
        self.block_size = block_size
        self.index_file = doc_offsets_file
        self.access = access
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.ignore_index = ignore_index
        self.mask_bos_loss = mask_bos_loss
        self.ensure_bos_token = ensure_bos_token
        self.mask_eos_loss = mask_eos_loss
        self.ensure_eos_token = ensure_eos_token

        self.data_dtype = data_dtype  # needs to address all token_ids in the vocab
        self.doc_offset_dtype = doc_offset_dtype  # needs to address all tokens in the dataset
        # output_dtype needs to represent data_dtype losslessly
        self.output_dtype = output_dtype  # needs to fit the entire vocab range. torch.from_numpy wants intXX, not uintXX

        self.data = np.memmap(self.data_file, dtype=self.data_dtype, mode="r")
        self.doc_offsets = np.memmap(self.index_file, dtype=self.doc_offset_dtype, mode="r")

        if self.access in ["docs-iid-packed-no-attn-fix", "docs-iid-packed"]:
            self.build_chunks_for_doc_packing()  # sets appropriate self.samples
        if self.access == "docs-iid-packed":
            print("Using docs-iid-packed access. This si designed to be used in conjunction with FlashAttention. You probably also need to use the monkeypatches from `dlib/monkeypatch_fa_packing.py`.")
        elif self.access in [
            "single-doc-padded",
            "single-doc-full",
        ]:
            self.num_samples = self.doc_offsets.size
        elif self.access == "contiguous":
            self.num_samples = self.data.size // self.block_size

        if self.ensure_bos_token or self.ensure_eos_token:
            assert self.access in ["single-doc-padded", "single-doc-full", "docs-iid-packed"]

        self.training_order = None
        if shuffle:
            self.training_order = self.get_reproducible_shuffled_training_order_for_epoch()

    def build_chunks_for_doc_packing(self):
        """
        Build chunks for document packing. This is very naive, could use better packing algos (it's a bin packing problem).
        But we can easily just paste them here and the rest will work.

        Populates self.samples with list[list[int]] := a list of lists containing document idxs that belong to each chunk.
        For efficient storage & loading w/ memmap, we pad each chunk to the same length with uint32 max.
        """
        print("Building chunks for document packing...")

        sample_idx_dtype = np.uint32
        num_documents = self.doc_offsets.size

        cache_path = self.data_file.with_suffix(
            f"{self.data_file.suffix}.chunked_n{num_documents}_w_blocksize_{self.block_size}.npy"
        )

        if not cache_path.exists():
            iid_sampled_docs = np.arange(start=0, stop=num_documents, step=1, dtype=sample_idx_dtype)
            rng = np.random.default_rng(seed=42)
            rng.shuffle(iid_sampled_docs)
            samples = []
            max_docs_per_chunk = 0
            iter_bar = tqdm(total=num_documents, leave=False, desc="Building chunks...")
            while len(iid_sampled_docs) > 0:
                new_chunk = []
                current_sample_len = 0
                end = False
                while current_sample_len < self.block_size + 1:  # +1 for labels
                    if len(iid_sampled_docs) == 0:
                        end = True
                        break
                    random_doc_idx = iid_sampled_docs[0]
                    if random_doc_idx == num_documents - 1:
                        # skip last doc for more convenient implementation
                        # if we really needed ALLLL data, we could also do it
                        iid_sampled_docs = iid_sampled_docs[1:]
                        continue
                    random_doc_len = self.doc_offsets[random_doc_idx + 1] - self.doc_offsets[random_doc_idx]
                    if random_doc_len > self.block_size:
                        samples.append([random_doc_idx])
                        iid_sampled_docs = iid_sampled_docs[1:]
                    else:
                        iid_sampled_docs = iid_sampled_docs[1:]
                        current_sample_len += random_doc_len
                        new_chunk.append(random_doc_idx)
                    iter_bar.update(1)
                if not end:
                    samples.append(new_chunk)
                    max_docs_per_chunk = max(max_docs_per_chunk, len(new_chunk))

            # shuffle samples
            rng.shuffle(samples)
            self.samples = samples
            print(
                f"Built {len(self.samples)} samples for document packing from {num_documents} documents. First 5:",
                self.samples[:5],
            )
            # use uint32 max as padding for efficient storage so that we don't have to switch to int32 w/ half dynamic range in positive values
            # if we have uint32 max many *documents*, let's be serious this should probably all be straight C++ and CUDA
            samples_np = np.full((len(self.samples), max_docs_per_chunk), UINT32_MAX, dtype=sample_idx_dtype)
            for i, sample in enumerate(tqdm(self.samples, desc="Converting to numpy for efficient loading...")):
                samples_np[i, : len(sample)] = sample
            np.save(cache_path, samples_np)

        print(f"Loading cached chunked samples from {cache_path}...")
        self.samples: np.memmap = np.load(cache_path, mmap_mode="r")
        self.num_samples = self.samples.shape[0]

    def get_reproducible_shuffled_training_order_for_epoch(self, epoch: int = 0):
        """
        Write a .npy file containing the shuffled indices for reproducible and resumable training.
        """
        assert self.num_samples is not None
        seed = epoch

        cache_path = self.data_file.with_suffix(f".shuffled_idx_w_seed_{seed}_n_{self.num_samples}.npy")

        if not cache_path.exists():
            # needs to address number of *samples (documents)* in the dataset, which is < 2**32 ~ 4.3B
            sample_idx_dtype = np.uint32
            training_order = np.arange(start=0, stop=self.num_samples, step=1, dtype=sample_idx_dtype)
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(training_order)
            np.save(cache_path, training_order)

        print(f"Loading cached shuffled indices from {cache_path}")
        training_order = np.load(cache_path, mmap_mode="r")
        return training_order

    def __len__(self) -> int:
        return self.num_samples

    def _read_data(self, start_idx: int, end_idx: int) -> torch.Tensor:
        return torch.from_numpy((self.data[start_idx:end_idx]).astype(self.output_dtype))

    def _mask(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mask_bos_loss:
            y[x == self.bos_token] = self.ignore_index
        y[y == self.bos_token] = self.ignore_index  # never learn to pred BOS tokens
        if self.mask_eos_loss:
            y[x == self.eos_token] = self.ignore_index
        y[y == self.eos_token] = self.ignore_index  # never learn to pred EOS tokens
        y[x == self.pad_token] = self.ignore_index  # always ignore pad tokens - never learn loss *on* pad tokens
        y[y == self.pad_token] = self.ignore_index  # always ignore pad tokens - never learn to pred pad tokens
        return x, y

    def _maybe_ensure_bos_token(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ensure_bos_token:
            if x[0] != self.bos_token:
                y = x
                x = torch.cat([torch.tensor([self.bos_token], dtype=x.dtype), x[:-1]])
        return x, y

    def _read_x_y_data(
        self, sample_idx: int, single_doc: bool = True, truncate_to_block_size: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        doc_data_start = self.doc_offsets[sample_idx].item()
        if single_doc:
            doc_data_end = self.doc_offsets[sample_idx + 1].item()
        else:
            doc_data_end = doc_data_start + self.block_size + 1
        doc_data = self._read_data(doc_data_start, doc_data_end)
        if self.ensure_bos_token:
            if doc_data[0] != self.bos_token:
                doc_data = torch.cat([torch.tensor([self.bos_token], dtype=doc_data.dtype), doc_data])
        if truncate_to_block_size:
            doc_data = doc_data[: self.block_size + 1]
        if self.ensure_eos_token:
            if doc_data[-1] != self.eos_token:
                doc_data[-1] = self.eos_token
        x = doc_data[:-1]
        y = doc_data[1:].detach().clone()
        return x, y

    def _read_x_y_data_for_doc_packing(
        self, sample_idx: int, return_doc_boundaries: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        TODO: need to fix that the last token of a doc has the first token of the next doc as target.
        This is fine if it's a BOS since the loss will be masked but not if it's a real token and wastes a bit of computation.
        """
        assert self.ensure_bos_token, "No BOS token is currently bugged."
        docs_idxs = self.samples[sample_idx]
        # remove padding that was added for efficient storage, uint32 max used as padding
        docs_idxs = [doc_idx for doc_idx in docs_idxs if doc_idx != UINT32_MAX]  
        data = None
        data_document_boundaries = []
        for doc_idx in docs_idxs:
            doc_data_start = self.doc_offsets[doc_idx].item()
            doc_data_end = self.doc_offsets[doc_idx + 1].item()
            doc_data = self._read_data(doc_data_start, doc_data_end)
            if self.ensure_bos_token:
                if doc_data[0] != self.bos_token:
                    doc_data = torch.cat([torch.tensor([self.bos_token], dtype=doc_data.dtype), doc_data])
            if self.ensure_eos_token:
                if doc_data[-1] != self.eos_token:
                    doc_data = torch.cat([doc_data, torch.tensor([self.eos_token], dtype=doc_data.dtype)])
            if data is None:
                data = doc_data
            else:
                data = torch.cat([data, doc_data])
            data_document_boundaries.append(data.size(0))

        if y.size(0) < self.block_size:
            y = torch.cat([y, torch.full((self.block_size - y.size(0),), self.ignore_index, dtype=y.dtype)])
            print("ERROR: y.size(0) < block_size", y.size(0), self.block_size)
        if x.size(0) < self.block_size:
            x = torch.cat([x, torch.full((self.block_size - x.size(0),), self.pad_token, dtype=x.dtype)])
            print("ERROR: x.size(0) < block_size", x.size(0), self.block_size)

        if return_doc_boundaries:
            return x, y, data_document_boundaries
        else:
            return x, y

    def _docs_packed_iid_no_attn_fix_access(self, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a chunk of `block_size` containing packed documents. Do not return an adjusted attention mask to prevent cross-doc attention.
        """
        x, y = self._read_x_y_data_for_doc_packing(sample_idx)
        x, y = self._mask(x, y)
        return {"input_ids": x, "labels": y}

    def _docs_packed_iid_access(self, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a chunk of `block_size` containing packed documents. Construct and return an adjusted attention mask to prevent cross-doc attention.
        """
        x, y, doc_boundaries = self._read_x_y_data_for_doc_packing(sample_idx, return_doc_boundaries=True)

        # create attn_mask like [1, ..., 1, 2, ..., 2, ... n, ..., n], where 1s correspond to first doc, 2s to second doc etc.
        # see: https://github.com/MeetKai/functionary/tree/main/functionary/train/packing
        attention_mask = torch.ones_like(x)
        for packed_doc_offset in doc_boundaries:
            attention_mask[packed_doc_offset:] += 1
        attention_mask[x == self.pad_token] = 0  # 0s for pad tokens
        x, y = self._mask(x, y)
        return {"input_ids": x, "labels": y, "attention_mask": attention_mask}

    def _doc_start_single_doc_access(self, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract up to `block_size` tokens from the data file, always starting at the beginning of a document.
        When a document is shorter than `block_size`, the remainder is padded with the pad token.
        """
        x, y = self._read_x_y_data(sample_idx, single_doc=True, truncate_to_block_size=True)

        if x.size(0) < self.block_size:
            x = torch.cat([x, torch.full((self.block_size - x.size(0),), self.pad_token, dtype=x.dtype)])
            y = torch.cat([y, torch.full((self.block_size - y.size(0),), self.ignore_index, dtype=y.dtype)])
        x, y = self._mask(x, y)
        return {"input_ids": x, "labels": y}
    
    def _contiguous_access(self, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        **Deprecated unless you know what your are doing!**
        
        Just extract a contiguous block of tokens from the data file, not respecting any document boundaries.
        EOS tokens should have already been inserted after each document if using this mode.
        """
        data_idx = sample_idx * self.block_size
        x = self._read_data(data_idx, data_idx + self.block_size)
        y = self._read_data(data_idx + 1, data_idx + self.block_size + 1)
        x, y = self._maybe_ensure_bos_token(x, y)
        return {"input_ids": x, "labels": y}

    def _full_samples_access(self, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        NOTE: this does not respect the block_size, but returns the full sample.
        """
        x, y = self._read_x_y_data(sample_idx, single_doc=False, truncate_to_block_size=False)
        x, y = self._mask(x, y)
        return {"input_ids": x, "labels": y}

    def __getitem__(
        self, sample_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training_order is not None:
            sample_idx = self.training_order[sample_idx].item()

        if self.access == "contiguous":
            return self._contiguous_access(sample_idx)
        if self.access == "single-doc-full":
            return self._full_samples_access(sample_idx)
        if self.access == "single-doc-padded":
            return self._doc_start_single_doc_access(sample_idx)
        if self.access == "docs-iid-packed":
            return self._docs_packed_iid_access(sample_idx)
        if self.access == "docs-iid-packed-no-attn-fix":
            return self._docs_packed_iid_no_attn_fix_access(sample_idx)
        raise ValueError(f"Unknown access type: {self.access}")
