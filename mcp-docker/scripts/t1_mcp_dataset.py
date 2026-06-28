"""Dataset adapter for Mat-T1 MCP RL parquet files."""

from __future__ import annotations

import datasets

from verl.utils.dataset import RLHFDataset


class CustomRLHFDataset(RLHFDataset):
    """Load already-normalized Mat-T1 MCP parquet data.

    The parquet files are prepared by `convert_jsonl_to_verl_parquet.py` and
    already contain prompt, reward_model, extra_info, and agent_name columns.
    """

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe = datasets.concatenate_datasets(dataframes)
        print(f"dataset len: {len(self.dataframe)}")
