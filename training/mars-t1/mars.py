# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import re
from typing import Any

import datasets

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)





answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)



            # if data_source in ["Maxwell-Jia/AIME_2024", "yentinglin/aime_2025"]:
            #     dataframe = dataframe.map(
            #         self.map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
            #     )
            # else:
            #     dataframe = dataframe.map(self.map_fn2, num_proc=16)
            # dataframes.append(dataframe)
            # for data in
        #self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    # def map_fn(self, row: dict, *, data_source: str = None):
    #     if data_source == "Maxwell-Jia/AIME_2024":
    #         problem, answer = row["Problem"], row["Answer"]
    #     elif data_source == "yentinglin/aime_2025":
    #         problem, answer = row["problem"], row["answer"]
    #     print("yyyyy")
    #     print(row)
    #     print("xxxxx")
    #     prompt = problem + answer_format
    #     data = {
    #         "data_source": data_source.split("/")[1].lower(),  # aime_2024, aime_2025
    #         "prompt": [{"role": "user", "content": prompt}],
    #         "ability": "MATH",
    #         "reward_model": {"ground_truth": str(answer)},
    #         "agent_name": "tool_agent",
    #     }
    #     return data

    # def map_fn2(self, row: dict):
    #     content = row["prompt"][0]["content"]
    #     row["prompt"][0]["content"] = content + answer_format
    #     row["agent_name"] = "tool_agent"
    #     return row


def compute_score(data_source, solution_str, ground_truth, extra_info):
    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
    messages = extra_info.get("messages", [])
    num_turns = extra_info.get("num_turns")

    print("\u2b50"*20)
    print("len_messages:",len(messages),"num turns:",num_turns)

    for message in messages:
        print(message['role'],end=' ')
    print()
    # encourage model to call tools
    num_turns = extra_info["num_turns"]
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
