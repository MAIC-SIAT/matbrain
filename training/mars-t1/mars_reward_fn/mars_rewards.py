import re
import json
import os
import regex
import logging
import traceback
from verl.utils import hf_processor, hf_tokenizer
from verl.experimental.agent_loop.tool_parser import HermesToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.experimental.agent_loop.tool_parser import ToolParser, FunctionCall

logger = logging.getLogger(__file__)
# --- Global variables for caching ---
_TOKENIZER = None
_TOOL_PARSER = None
_TOOLS = None
_TOOLS_INITIALIZED = False


class HermesToolParser(ToolParser):
    """Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"""

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)


    def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        text = self.tokenizer.decode(responses_ids)
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []

        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
            except Exception as e:
                logger.error(f"Failed to decode tool call: {e}")

        # remaining text excluding tool call tokens
        content = self.tool_call_regex.sub("", text)

        return content, function_calls

def _initialize_dependencies():
    """
    Lazy initialization of tokenizer, tool parser, and tools.
    This function is called by reward functions that need these components.
    """
    global _TOKENIZER, _TOOL_PARSER, _TOOLS, _TOOLS_INITIALIZED
    if _TOOLS_INITIALIZED:
        return

    print("[DEBUG] Initializing dependencies for reward functions...")
    try:
        # Initialize tokenizer
        _TOKENIZER = hf_tokenizer(os.getenv("MODEL_PATH", "Qwen/Qwen3-14B"), trust_remote_code=True)
        print("[DEBUG] Tokenizer loaded successfully.")

        # Initialize tool parser
        _TOOL_PARSER = HermesToolParser(_TOKENIZER)
        print("[DEBUG] HermesToolParser initialized.")

        # Initialize tools from config
        # Assume the config is in a fixed relative path
        current_dir = os.path.dirname(__file__)
        default_tool_config = os.path.abspath(
            os.path.join(current_dir, "..", "mars_mcp_tools", "mars_tool_config.yaml")
        )
        tool_config_path = os.getenv("MARS_TOOL_CONFIG", default_tool_config)
        if not os.path.exists(tool_config_path):
            raise FileNotFoundError(f"Tool config file not found at: {tool_config_path}")

        tool_list = initialize_tools_from_config(tool_config_path)
        _TOOLS = {tool.name: tool for tool in tool_list}
        print(f"[DEBUG] Tools loaded successfully: {list(_TOOLS.keys())}")

        _TOOLS_INITIALIZED = True
        print("[DEBUG] All dependencies initialized.")

    except Exception as e:
        print(f"[ERROR] Failed to initialize dependencies: {e}")
        print(traceback.format_exc())
        # Reset state if initialization fails
        _TOKENIZER = None
        _TOOL_PARSER = None
        _TOOLS = None
        _TOOLS_INITIALIZED = False


def _validate_arguments(tool_name: str, provided_args: dict) -> bool:
    """
    Validates the arguments for a given tool call against its schema.
    """
    if not _TOOLS or tool_name not in _TOOLS:
        return False

    tool_schema = _TOOLS[tool_name].tool_schema

    # Pydantic models are available in tool_schema.function
    if not hasattr(tool_schema, 'function') or not hasattr(tool_schema.function, 'parameters'):
        # If no parameters are defined, any empty dict is valid, non-empty is not.
        return not provided_args

    schema_params = tool_schema.function.parameters
    required_params = set(schema_params.required or [])
    all_defined_params = set(schema_params.properties.keys())
    provided_params = set(provided_args.keys())

    # 1. Check for missing required parameters
    if not required_params.issubset(provided_params):
        return False

    # 2. Check for unknown parameters
    if not provided_params.issubset(all_defined_params):
        return False

    # 3. Check for empty required parameters
    for param in required_params:
        value = provided_args.get(param)
        if value is None or (isinstance(value, (str, list, dict)) and not value):
            return False

    # 4. Validate cif_string format if it's a required parameter
    if "cif_string" in required_params:
        try:
            from pymatgen.core import Structure
            cif_string = provided_args.get("cif_string")
            if not cif_string:
                return False
            # Attempt to parse the CIF string
            Structure.from_str(cif_string, fmt='cif')
        except Exception:
            # If parsing fails, the format is invalid
            return False

    return True


def compute_score_syntax(data_source, solution_str, ground_truth, extra_info):
    """
    Computes the reward score for tool call syntax correctness.
    Checks if the tool call can be parsed, if the tool name is valid,
    and if the arguments match the tool's schema.
    """
    _initialize_dependencies()

    if not _TOOL_PARSER or not _TOOLS:
        print("[DEBUG] Syntax scoring skipped: dependencies not initialized.")
        return 0.0

    try:
        messages = extra_info.get("messages", [])
        assistant_blocks = [msg['content'] for msg in messages if msg['role'] == 'assistant']

        if not assistant_blocks:
            return 0.0

        block_scores = []

        # Iterate over all assistant blocks that might contain tool calls (typically all but the last)
        for i, block_content in enumerate(assistant_blocks[:-1]):
            if '<tool_call>' not in block_content:
                continue

            # The parser expects token IDs, so we need to encode the text.
            # Note: This is an approximation. The original flow has access to raw token IDs.
            token_ids = _TOKENIZER.encode(block_content, add_special_tokens=False)

            # extract_tool_calls is now sync
            _, tool_calls = _TOOL_PARSER.extract_tool_calls(token_ids)

            if not tool_calls:
                # Penalize if tags are present but parsing fails
                block_scores.append(0.0)
                continue

            tool_successes = []
            for tool_call in tool_calls:
                is_valid = False
                try:
                    # Basic check: name exists and is in our list of tools
                    if tool_call and tool_call.name in _TOOLS:
                        # Parse arguments string into a dict
                        args_dict = json.loads(tool_call.arguments)
                        # Validate arguments against the schema
                        if _validate_arguments(tool_call.name, args_dict):
                            is_valid = True
                except (json.JSONDecodeError, TypeError):
                    # Invalid JSON or other parsing error in arguments
                    is_valid = False

                tool_successes.append(is_valid)

            # Calculate success rate for the current block
            if tool_successes:
                block_score = sum(tool_successes) / len(tool_successes)
                block_scores.append(block_score)

        # Return the average score across all blocks that had tool calls
        if not block_scores:
            return 0.0

        return sum(block_scores) / len(block_scores)

    except Exception as e:
        print(f"[DEBUG] Error in compute_score_syntax: {e}")
        import traceback
        print(traceback.format_exc())
        return 0.0

def compute_score_turns(data_source, solution_str, ground_truth, extra_info):
    """计算多轮工具调用的奖励分数

    根据有效工具调用轮数给出奖励分数：
    - 1轮：0.2分
    - 2轮（基本情况）：0.5分
    - 3轮：0.7分
    - 4轮：0.85分
    - 5轮及以上：1.0分

    Args:
        solution_str: 解决方案文本

    Returns:
        float: 轮数奖励分数（0.0-1.0）
    """

    try:
        messages = extra_info.get("messages", [])


        # 提取所有assistant块
        assistant_blocks = [message['content'] for message in messages if message['role']=='assistant']


        if not assistant_blocks:
            return 0.0

        # 统计包含工具调用的轮数
        tool_call_rounds = 0

        # 遍历除最后一个block外的所有assistant blocks（最后一个通常是最终答案）
        for i, assistant_block in enumerate(assistant_blocks[:-1]):
            # 检查是否包含工具调用
            if '<tool_call>' in assistant_block and '</tool_call>' in assistant_block:
                # 验证工具调用格式是否正确
                tool_call_count = assistant_block.count('<tool_call>')
                tool_call_end_count = assistant_block.count('</tool_call>')

                # 只有当工具调用标签配对正确时才计入轮数
                if tool_call_count > 0 and tool_call_count == tool_call_end_count:
                    tool_call_rounds += 1

        # 根据轮数映射到奖励分数
        if tool_call_rounds == 0:
            return 0.0
        elif tool_call_rounds == 1:
            return 0.5
        elif tool_call_rounds == 2:
            return 0.7
        elif tool_call_rounds >= 3:
            return 1


    except Exception as e:
        print(f"[DEBUG] Error in compute_score_turns: {e}")
        return 0.0



def _get_tokenizer():
    """获取tokenizer，使用懒加载模式"""
    global _TOKENIZER
    if _TOKENIZER is None:
        _initialize_dependencies()
    return _TOKENIZER

def compute_score_think_length(data_source, solution_str, ground_truth, extra_info):
    """
    Computes a reward score based on the length of <think> content.
    This version uses a smoother reward curve to encourage longer thought processes.
    """
    try:
        tokenizer = _get_tokenizer()
        if not tokenizer:
            print("[DEBUG] Tokenizer not available. Cannot score think length.")
            return 0.0

        messages = extra_info.get("messages", [])
        assistant_blocks = [message['content'] for message in messages if message['role'] == 'assistant']

        if not assistant_blocks:
            return 0.0

        think_contents = []
        for assistant_block in assistant_blocks:
            think_matches = re.findall(r'<think>(.*?)</think>', assistant_block, re.DOTALL)
            for think_content in think_matches:
                think_content = think_content.strip()
                if think_content:
                    think_contents.append(think_content)

        if not think_contents:
            return 0.0

        # --- New Scoring Logic (v2) ---
        import math

        total_tokens = 0
        for think_content in think_contents:
            tokens = tokenizer.encode(think_content, add_special_tokens=False)
            total_tokens += len(tokens)

        if total_tokens == 0:
            return 0.0

        # Average tokens per think block
        avg_tokens = total_tokens / len(think_contents)

        # Use a tanh function for a smooth reward curve.
        # The parameters are chosen to give reasonable rewards at different lengths.
        # - Around 100 tokens: ~0.25 score
        # - Around 250 tokens: ~0.5 score
        # - Around 500 tokens: ~0.76 score
        # - Around 1000 tokens: ~0.96 score
        # The score gracefully approaches 1.0 as length increases.
        scale_factor = 500  # Controls how quickly the score rises.
        score = math.tanh(avg_tokens / scale_factor)

        return score

    except Exception as e:
        print(f"[ERROR] Error in compute_score_think_length: {e}")
        return 0.0

def compute_score_format(data_source, solution_str, ground_truth, extra_info):
    """
    Computes a reward score based on the structural format of the assistant's responses.
    This version is more flexible and correctly handles multi-turn scenarios where
    the model can decide to answer at any step.
    """
    try:
        print("\n[DEBUG FORMAT] --- Starting compute_score_format (v2) ---")
        messages = extra_info.get("messages", [])
        assistant_blocks = [message['content'] for message in messages if message['role'] == 'assistant']

        if not assistant_blocks:
            print("[DEBUG FORMAT] No assistant blocks found. Returning 0.0")
            return 0.0

        # Find the index of the first block containing an <answer> tag
        final_answer_index = -1
        for i, block in enumerate(assistant_blocks):
            if '<answer>' in block:
                final_answer_index = i
                break

        # If no answer tag is found, all blocks are considered intermediate
        if final_answer_index == -1:
            intermediate_blocks = assistant_blocks
            last_block = None
        else:
            intermediate_blocks = assistant_blocks[:final_answer_index]
            last_block = assistant_blocks[final_answer_index]

        format_reward = 0.0

        # --- Part 1: Score intermediate tool-calling blocks (Max 0.4 points) ---
        # This part now rewards each valid intermediate block proportionally.
        if intermediate_blocks:
            valid_intermediate_count = 0
            print(f"[DEBUG FORMAT] Checking {len(intermediate_blocks)} intermediate blocks.")
            for i, block in enumerate(intermediate_blocks):
                is_valid_intermediate = False
                # A valid intermediate block must contain <think> and <tool_call> and NOT <answer>
                if '<think>' in block and '</think>' in block and \
                   '<tool_call>' in block and '</tool_call>' in block and \
                   '<answer>' not in block:
                    # Check for correct ordering: <think> must come before <tool_call>
                    if block.find('<think>') < block.find('<tool_call>'):
                        is_valid_intermediate = True

                if is_valid_intermediate:
                    valid_intermediate_count += 1
                    print(f"  - Intermediate block {i+1}: VALID")
                else:
                    print(f"  - Intermediate block {i+1}: INVALID")

            # Reward is proportional to the number of valid intermediate blocks
            intermediate_reward = (valid_intermediate_count / len(intermediate_blocks)) * 0.4
            format_reward += intermediate_reward
            print(f"[DEBUG FORMAT] Part 1 Score: {intermediate_reward:.2f} ({valid_intermediate_count}/{len(intermediate_blocks)} valid blocks)")
        else:
            print("[DEBUG FORMAT] Part 1 SKIPPED: No intermediate blocks.")


        # --- Part 2: Score the final answer block (Max 0.6 points) ---
        # This block must contain <think> and <answer>.
        if last_block:
            print("[DEBUG FORMAT] Checking final answer block.")
            # A valid final block must contain <think> and <answer> in the correct order.
            if '<think>' in last_block and '</think>' in last_block and \
               '<answer>' in last_block and '</answer>' in last_block and \
               last_block.find('<think>') < last_block.find('<answer>'):
                final_reward = 0.6
                # Bonus: If there were no intermediate steps (direct answer), give full marks.
                if not intermediate_blocks:
                    final_reward = 1.0
                format_reward += final_reward
                print(f"  - Final block: VALID. Score += {final_reward:.2f}")
            else:
                print("  - Final block: INVALID.")
        else:
            print("[DEBUG FORMAT] Part 2 SKIPPED: No final answer block found.")

        # Ensure the final score is capped at 1.0
        final_score = min(format_reward, 1.0)

        print(f"\n[DEBUG FORMAT] --- Final reward: {final_score:.2f} ---")
        return final_score

    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        import traceback
        print(traceback.format_exc())
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info):
    """
    Computes a weighted final score by combining multiple reward functions.
    """
    # Define weights for each reward component
    weights = {
        "syntax": 0.35,
        "format": 0.25,
        "turns": 0.1,
        "think_length": 0.3,
    }

    # --- Call individual reward functions ---

    # 1. Syntax Score
    syntax_score = compute_score_syntax(data_source, solution_str, ground_truth, extra_info)

    # 2. Format Score
    format_score = compute_score_format(data_source, solution_str, ground_truth, extra_info)

    # 3. Turns Score
    turns_score = compute_score_turns(data_source, solution_str, ground_truth, extra_info)

    # 4. Think Length Score
    think_length_score = compute_score_think_length(data_source, solution_str, ground_truth, extra_info)

    # --- Calculate weighted score ---
    final_score = (
        weights["syntax"] * syntax_score +
        weights["format"] * format_score +
        weights["turns"] * turns_score +
        weights["think_length"] * think_length_score
    )

    # You can print the partial scores for debugging if needed
    # print(f"[DEBUG] Scores -> Syntax: {syntax_score:.2f}, Format: {format_score:.2f}, Turns: {turns_score:.2f}, Think: {think_length_score:.2f} | Final: {final_score:.2f}")

    metrics_dict = {
        "score": final_score,  # Must include "score" key for the main reward
        "reward/syntax": syntax_score,
        "reward/format": format_score,
        "reward/turns": turns_score,
        "reward/think_length": think_length_score,
    }
    print('\u2b50'*10,metrics_dict)
    return metrics_dict
