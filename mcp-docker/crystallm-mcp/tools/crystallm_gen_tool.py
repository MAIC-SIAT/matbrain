"""
CrystaLLM Crystal Structure Generation Tools

This module provides tools for generating crystal structures using the CrystaLLM model.
It integrates prompt creation, sampling, and post-processing functionality.
"""

import os
import subprocess
import shutil
from typing import Optional
from core import llm_tool
from config.config import crystallm_config
from core.utils import remove_symmetry_equiv_xyz, validate_chemical_formula


def _subprocess_error_payload(stage: str, result: subprocess.CompletedProcess[str]) -> str:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    tail = stderr[-4000:] if stderr else ""
    out_tail = stdout[-1000:] if stdout else ""
    pieces = [
        f"{stage} failed",
        f"returncode={result.returncode}",
    ]
    if tail:
        pieces.append(f"stderr_tail=\n{tail}")
    if out_tail:
        pieces.append(f"stdout_tail=\n{out_tail}")
    return " | ".join(pieces)

@llm_tool(name="generate_crystal_structures_crystallm",
          description="Generate crystal structures using CrystaLLM model based on chemical formula and optional space group")
async def generate_crystal_structures_crystallm(
    formula: str,
    space_group: Optional[str] = None,
    num_samples: int = 2
) -> str:
    """
    Generate crystal structures using the CrystaLLM model.

    Args:
        formula: Chemical formula (e.g., "CsPbBr3", "Na2Cl2"). Elements must be sorted by electronegativity.
        space_group: Optional space group symbol (e.g., "P4/nmm", "Fd-3m")
        num_samples: Number of crystal structures to generate (default: 2)

    Returns:
        Formatted string containing the generated and post-processed CIF files
    """
    # 验证化学式格式
    validated_formula = validate_chemical_formula(formula)

    # 确保临时目录存在
    temp_dir = crystallm_config.TEMP_ROOT
    os.makedirs(temp_dir, exist_ok=True)

    # 创建工作目录
    work_dir = os.path.join(temp_dir, f"crystallm/{validated_formula}")
    os.makedirs(work_dir, exist_ok=True)

    # CrystaLLM相关路径
    crystallm_root = crystallm_config.CRYSTALLM_ROOT
    model_dir = crystallm_config.CRYSTALLM_MODEL_DIR

    # 从配置文件获取技术参数
    temperature = crystallm_config.DEFAULT_TEMPERATURE
    top_k = crystallm_config.DEFAULT_TOP_K
    max_new_tokens = crystallm_config.DEFAULT_MAX_NEW_TOKENS
    device = crystallm_config.DEFAULT_DEVICE

    # 步骤1: 创建prompt文件
    prompt_file = os.path.join(work_dir, "prompt.txt")
    make_prompt_cmd = [
        "python", os.path.join(crystallm_root, "bin/make_prompt_file.py"),
        validated_formula, prompt_file
    ]

    if space_group:
        make_prompt_cmd.extend(["--spacegroup", space_group])

    # 执行prompt创建命令
    result = subprocess.run(make_prompt_cmd, capture_output=True, text=True, cwd=crystallm_root)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create prompt: {result.stderr}")

    # 步骤2: 生成原始CIF文件
    raw_cif_dir = os.path.join(work_dir, "raw_cifs")
    os.makedirs(raw_cif_dir, exist_ok=True)

    sample_cmd = [
        "python", os.path.join(crystallm_root, "bin/sample.py"),
        f"out_dir={model_dir}",
        f"start=FILE:{prompt_file}",
        f"num_samples={num_samples}",
        f"temperature={temperature}",
        f"top_k={top_k}",
        f"max_new_tokens={max_new_tokens}",
        f"device={device}",
        "target=file"
    ]

    # 执行采样命令；失败时先做一次保守降采样重试，避免偶发 sampling 抖动直接杀掉流程
    result = subprocess.run(sample_cmd, capture_output=True, text=True, cwd=raw_cif_dir)
    if result.returncode != 0 and int(num_samples) > 1:
        shutil.rmtree(raw_cif_dir, ignore_errors=True)
        os.makedirs(raw_cif_dir, exist_ok=True)
        retry_cmd = list(sample_cmd)
        retry_cmd[retry_cmd.index(f"num_samples={num_samples}")] = "num_samples=1"
        result = subprocess.run(retry_cmd, capture_output=True, text=True, cwd=raw_cif_dir)
    if result.returncode != 0:
        raise RuntimeError(_subprocess_error_payload("sampling", result))

    # 步骤3: 后处理CIF文件
    processed_cif_dir = os.path.join(work_dir, "processed_cifs")
    os.makedirs(processed_cif_dir, exist_ok=True)

    postprocess_cmd = [
        "python", os.path.join(crystallm_root, "bin/postprocess.py"),
        raw_cif_dir, processed_cif_dir
    ]

    # 执行后处理命令
    result = subprocess.run(postprocess_cmd, capture_output=True, text=True, cwd=crystallm_root)
    if result.returncode != 0:
        raise RuntimeError(_subprocess_error_payload("post-processing", result))

    # 步骤4: 读取并格式化结果
    cif_files = [f for f in os.listdir(processed_cif_dir) if f.endswith('.cif')]
    if not cif_files:
        raise RuntimeError(f"No CIF files were generated successfully for formula: {validated_formula}")

    # 按文件名排序
    cif_files.sort()

    # 构建结果字符串
    result_parts = []
    result_parts.append(f"# CrystaLLM Generated Crystal Structures")
    result_parts.append(f"")
    result_parts.append(f"Generated {len(cif_files)} crystal structure(s) for formula: {validated_formula}")
    if space_group:
        result_parts.append(f"Space group constraint: {space_group}")
    result_parts.append(f"")

    # 读取每个CIF文件的内容
    for i, cif_file in enumerate(cif_files, 1):
        cif_path = os.path.join(processed_cif_dir, cif_file)
        with open(cif_path, 'r', encoding='utf-8') as f:
            cif_content = f.read()
            cif_content = remove_symmetry_equiv_xyz(cif_content)  # 移除对称等效位置
        result_parts.append(f"## Structure {i}: ")
        result_parts.append(f"```")
        result_parts.append(cif_content.strip())
        result_parts.append(f"```")
        result_parts.append(f"")

    # 清理临时文件
    shutil.rmtree(work_dir)

    return "\n".join(result_parts)
