"""统一工具模块

整合了mat-query-mcp和matgl-mcp项目中的通用工具函数，
包含化学式验证、CIF文件处理、结构处理、格式化和参数验证等功能。
"""

import json
import logging
import re
import os
from typing import Dict, Any, List, Tuple, Union
from pymatgen.core import Structure

logger = logging.getLogger(__name__)


# ============================================================================
# 化学式处理模块
# ============================================================================

# Unicode下标字符映射表
SUBSCRIPT_MAP = {
    '₀': '0',
    '₁': '1',
    '₂': '2',
    '₃': '3',
    '₄': '4',
    '₅': '5',
    '₆': '6',
    '₇': '7',
    '₈': '8',
    '₉': '9'
}


def convert_subscripts_to_normal(text):
    """将文本中的Unicode下标字符转换为普通数字

    Args:
        text: 包含可能的下标字符的文本

    Returns:
        str: 转换后的文本，下标字符被替换为普通数字
    """
    if not text:
        return text

    result = text
    for subscript, normal in SUBSCRIPT_MAP.items():
        result = result.replace(subscript, normal)
    return result


class ChemicalFormulaValidator:
    """化学式验证器"""

    def __init__(self):
        # 完整的元素周期表
        self.elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
            'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
        }

    def extract_elements(self, formula):
        """提取化学式中的所有元素符号"""
        # 匹配元素符号（大写字母开头，可能跟一个小写字母）
        elements = re.findall(r'[A-Z][a-z]?', formula)
        return elements

    def is_valid_chemical_formula(self, formula):
        """
        高级版本：验证是否为有效化学表达式
        - 检查元素是否存在于周期表中
        - 验证格式是否正确
        - 支持复杂结构
        """
        if not formula or not isinstance(formula, str):
            return False

        # 预处理
        original = formula
        formula = re.sub(r'\s+', '', formula)  # 移除空格
        formula = convert_subscripts_to_normal(formula)  # 转换下标字符

        # 如果为空，返回False
        if not formula:
            return False

        try:
            # 第一步：基础格式检查
            if not self._check_basic_format(formula):
                return False

            # 第二步：提取并验证元素
            elements = self._extract_all_elements(formula)
            if not elements:
                return False

            # 检查所有元素是否在周期表中
            for element in elements:
                if element not in self.elements:
                    return False

            # 第三步：结构验证
            return self._validate_structure(formula)

        except Exception as e:
            return False

    def _check_basic_format(self, formula):
        """检查基础格式"""
        # 不能以数字开头
        if formula[0].isdigit():
            return False

        # 不能包含非法字符（除了字母、数字、括号、点、加减号）
        allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]·•+-^')
        if not set(formula).issubset(allowed_chars):
            return False

        return True

    def _extract_all_elements(self, formula):
        """提取所有元素，包括括号内的"""
        # 先移除离子标记和水合物标记
        clean_formula = re.sub(r'[·•]\d*H2O', '', formula)  # 移除水合物
        clean_formula = re.sub(r'[\+\-\^]\d*[\+\-]?', '', clean_formula)  # 移除离子标记
        clean_formula = re.sub(r'[\[\]]', '', clean_formula)  # 移除方括号

        # 提取元素
        elements = re.findall(r'[A-Z][a-z]?', clean_formula)
        return list(set(elements))  # 去重

    def _validate_structure(self, formula):
        """验证化学式结构"""
        # 检查括号匹配
        if not self._check_parentheses(formula):
            return False

        # 检查基本的化学式模式
        patterns = [
            r'^([A-Z][a-z]?\d*)+$',  # 基础化学式
            r'^([A-Z][a-z]?\d*|\([A-Za-z0-9]+\)\d*)+$',  # 带括号
            r'^[A-Za-z0-9()]+[·•]\d*H2O$',  # 水合物
            r'^(\[?[A-Za-z0-9()]+\]?)[\+\-]?\d*[\+\-]?$',  # 离子
            r'^([A-Z][a-z]?\d*|\([A-Za-z0-9]+\)\d*)+([·•]\d*H2O)?$',  # 组合
        ]

        for pattern in patterns:
            if re.match(pattern, formula):
                return True

        return False

    def _check_parentheses(self, formula):
        """检查括号是否匹配"""
        stack = []
        pairs = {'(': ')', '[': ']'}

        for char in formula:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack or pairs.get(stack.pop()) != char:
                    return False

        return len(stack) == 0


# 全局验证器实例
formula_validator = ChemicalFormulaValidator()


def validate_chemical_formula(formula: str) -> str:
    """
    验证化学式并返回清理后的化学式

    Args:
        formula: 输入的化学式字符串

    Returns:
        str: 验证通过的清理后化学式

    Raises:
        ValueError: 当化学式格式无效时抛出异常，包含详细错误信息
    """
    if not formula or not isinstance(formula, str):
        raise ValueError("化学式不能为空且必须是字符串")

    # 清理化学式格式
    cleaned_formula = formula.replace(" ", "").replace("\n", "").replace("'", "").replace('"', "")

    # 转换Unicode下标字符为普通数字
    cleaned_formula = convert_subscripts_to_normal(cleaned_formula)

    # 处理可能的等号分割格式
    if "=" in cleaned_formula:
        name, id = cleaned_formula.split("=")
        target_formula = id.strip()
    else:
        target_formula = cleaned_formula

    if not target_formula:
        raise ValueError("清理后的化学式为空")

    # 使用现有的验证器进行详细验证
    validator = formula_validator

    # 基础格式检查
    if not validator._check_basic_format(target_formula):
        if target_formula[0].isdigit():
            raise ValueError(f"化学式格式错误: '{target_formula}' - 化学式不能以数字开头")

        # 检查非法字符
        allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]·•+-^')
        invalid_chars = set(target_formula) - allowed_chars
        if invalid_chars:
            raise ValueError(f"化学式格式错误: '{target_formula}' - 包含非法字符: {', '.join(invalid_chars)}")

    # 提取并验证元素
    elements = validator._extract_all_elements(target_formula)
    if not elements:
        raise ValueError(f"化学式格式错误: '{target_formula}' - 未找到有效的化学元素")

    # 检查元素是否在周期表中
    invalid_elements = []
    for element in elements:
        if element not in validator.elements:
            invalid_elements.append(element)

    if invalid_elements:
        raise ValueError(f"化学式包含无效元素: {', '.join(invalid_elements)} - 请检查元素符号是否正确")

    # 检查括号匹配
    if not validator._check_parentheses(target_formula):
        raise ValueError(f"化学式格式错误: '{target_formula}' - 括号不匹配")

    # 结构验证
    if not validator._validate_structure(target_formula):
        raise ValueError(f"化学式结构错误: '{target_formula}' - 不符合标准化学式格式")

    return target_formula


# ============================================================================
# CIF文件处理模块
# ============================================================================

def extract_cif_info(path: str, fields_name: List[str]) -> Dict[str, Any]:
    """
    从CIF描述JSON文件中提取特定字段

    Args:
        path: 包含CIF信息的JSON文件路径
        fields_name: 要提取的字段类别列表。使用'all_fields'提取所有字段。
                    其他选项包括'basic_fields', 'energy_electronic_fields', 'metal_magentic_fields'

    Returns:
        包含提取字段的字典
    """
    basic_fields = [
        'formula_pretty', 'chemsys', 'composition', 'elements',
        'symmetry', 'nsites', 'volume', 'density'
    ]
    energy_electronic_fields = [
        'formation_energy_per_atom', 'energy_above_hull', 'is_stable',
        'efermi', 'cbm', 'vbm', 'band_gap', 'is_gap_direct'
    ]
    metal_magentic_fields = [
        'is_metal', 'is_magnetic', "ordering",
        'total_magnetization', 'num_magnetic_sites'
    ]

    selected_fields = []
    if fields_name[0] == 'all_fields':
        selected_fields = basic_fields + energy_electronic_fields + metal_magentic_fields
    else:
        for field in fields_name:
            if field == 'basic_fields':
                selected_fields.extend(basic_fields)
            elif field == 'energy_electronic_fields':
                selected_fields.extend(energy_electronic_fields)
            elif field == 'metal_magentic_fields':
                selected_fields.extend(metal_magentic_fields)

    try:
        with open(path, 'r', encoding='utf-8') as f:
            docs = json.load(f)

        new_docs = {}
        for field_name in selected_fields:
            new_docs[field_name] = docs.get(field_name, '')

        return new_docs

    except Exception as e:
        logger.error(f"提取CIF信息时出错 {path}: {e}")
        return {}


def remove_symmetry_equiv_xyz(cif_content: str) -> str:
    """
    从CIF文件内容中移除对称操作部分

    这在某些可视化工具中或专注于基本结构而不需要对称操作时很有用。

    Args:
        cif_content: CIF文件内容字符串

    Returns:
        移除对称操作后的清理CIF内容字符串
    """
    lines = cif_content.split('\n')
    output_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 检测循环开始
        if line == 'loop_':
            # 查看下一行，检查是否是对称性循环
            next_lines = []
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith('_'):
                next_lines.append(lines[j].strip())
                j += 1

            # 检查是否包含对称性操作标签
            if any('_symmetry_equiv_pos_as_xyz' in tag for tag in next_lines):
                # 跳过整个循环块
                while i < len(lines):
                    if i + 1 >= len(lines):
                        break

                    next_line = lines[i + 1].strip()
                    # 检查是否到达下一个循环或数据块
                    if next_line == 'loop_' or next_line.startswith('data_'):
                        break

                    # 检查是否到达原子位置部分
                    if next_line.startswith('_atom_site_'):
                        break

                    i += 1
            else:
                # 不是对称性循环，保留loop_行
                output_lines.append(lines[i])
        else:
            # 非循环开始行，直接保留
            output_lines.append(lines[i])

        i += 1

    return '\n'.join(output_lines)


# ============================================================================
# 结构处理模块
# ============================================================================

def read_structure_from_cif_string(cif_string: str) -> Tuple[str, str]:
    """
    从CIF字符串或文件名读取结构内容

    Args:
        cif_string: CIF格式的结构字符串或文件名

    Returns:
        Tuple[结构内容字符串, 格式类型]

    Raises:
        FileNotFoundError: 当文件不存在时
        ValueError: 当无法解析结构时
    """
    # 检查是否为文件路径
    if os.path.isfile(cif_string):
        # 从文件读取
        with open(cif_string, 'r', encoding='utf-8') as f:
            content = f.read()

        # 根据文件扩展名确定格式
        _, ext = os.path.splitext(cif_string.lower())
        if ext == '.cif':
            format_type = 'cif'
        elif ext in ['.vasp', '.poscar', '.contcar']:
            format_type = 'poscar'
        else:
            # 默认尝试CIF格式
            format_type = 'cif'

        return content, format_type
    else:
        # 直接作为字符串内容处理
        # 简单判断格式类型
        if 'data_' in cif_string or '_cell_length_a' in cif_string:
            format_type = 'cif'
        else:
            # 默认尝试POSCAR格式
            format_type = 'poscar'

        return cif_string, format_type


def load_structure_from_cif_string(cif_string: str) -> Structure:
    """
    从CIF字符串加载pymatgen结构对象

    Args:
        cif_string: CIF格式的结构字符串或文件名

    Returns:
        pymatgen Structure对象

    Raises:
        ValueError: 当无法解析结构时
    """
    cif_content, format_type = read_structure_from_cif_string(cif_string)

    try:
        # 预处理
        processed_cif = cif_content.replace('\\n', '\n')
        processed_cif = processed_cif.replace('\\t', '\t')
        processed_cif = processed_cif.replace('\\"', '"')
        processed_cif = processed_cif.strip().strip('"\'')

        structure = Structure.from_str(processed_cif, fmt=format_type)
        if structure is None:
            raise ValueError("无法解析结构数据")
        return structure
    except Exception as e:
        raise ValueError(f"结构解析失败: {str(e)}")


# ============================================================================
# 格式化工具模块
# ============================================================================

def format_basic_structure_info(structure: Structure) -> str:
    """
    格式化基本结构信息

    Args:
        structure: pymatgen Structure对象

    Returns:
        格式化的结构信息字符串
    """
    reduced_formula = structure.composition.reduced_formula
    lattice_info = structure.lattice
    volume = structure.volume
    density = structure.density
    symmetry = structure.get_space_group_info()

    return (f"### 结构信息\n\n"
            f"- **分子式**: `{reduced_formula}`\n"
            f"- **空间群**: `{symmetry[0]} (#{symmetry[1]})`\n"
            f"- **体积**: `{volume:.2f} Å³`\n"
            f"- **密度**: `{density:.2f} g/cm³`\n"
            f"- **晶格参数**:\n"
            f"  - a = `{lattice_info.a:.6f} Å`, b = `{lattice_info.b:.6f} Å`, c = `{lattice_info.c:.6f} Å`\n"
            f"  - α = `{lattice_info.alpha:.6f}°`, β = `{lattice_info.beta:.6f}°`, γ = `{lattice_info.gamma:.6f}°`\n")


def format_optimization_status(optimized: bool) -> str:
    """
    格式化优化状态信息

    Args:
        optimized: 是否经过优化

    Returns:
        优化状态字符串
    """
    return "已优化" if optimized else "未优化"


# ============================================================================
# 参数验证模块
# ============================================================================

def validate_numeric_parameter(value: Union[str, int, float], param_name: str, min_val: float = None, max_val: float = None) -> float:
    """
    验证和转换数值参数

    Args:
        value: 参数值
        param_name: 参数名称
        min_val: 最小值限制
        max_val: 最大值限制

    Returns:
        转换后的浮点数值

    Raises:
        ValueError: 当参数无效时
    """
    try:
        num_val = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"参数 {param_name} 必须是数值类型")

    if min_val is not None and num_val < min_val:
        raise ValueError(f"参数 {param_name} 不能小于 {min_val}")

    if max_val is not None and num_val > max_val:
        raise ValueError(f"参数 {param_name} 不能大于 {max_val}")

    return num_val


#======================通过sse远程调用MCP
import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession
async def _call_mcp_tool(sse_url,tool_name: str, args: Dict,max_retries=1,timeout=3600,
                         sse_read_timeout = 60*10,  # 增加到10分钟
                            initialize_timeout = 30 , # 增加初始化超时
                            call_timeout = 60*8 , # 增加到8分钟
                            retry_delay=1
                         ) -> Dict[str, Any]:
        """
        调用MCP工具的通用方法，处理连接、重试和错误
        """
        import logging
        logger = logging.getLogger(__name__)



        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting {tool_name} call, attempt {attempt + 1}/{max_retries}")

                async with sse_client(
                    sse_url,
                    timeout=timeout,
                    sse_read_timeout=sse_read_timeout
                ) as streams:
                    async with ClientSession(*streams) as session:
                        await asyncio.wait_for(session.initialize(), timeout=initialize_timeout)
                        logger.info("SSE session initialized successfully")

                        res = await asyncio.wait_for(
                            session.call_tool(tool_name, args),
                            timeout=call_timeout
                        )
                        results = res.content[0].text
                        logger.info(f"{tool_name} call completed successfully")
                        return {"content": results, "success": True}
            except asyncio.TimeoutError as e:
                logger.warning(f"Timeout error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return {"content": f"Request timed out after {max_retries} attempts. This may be due to network issues or server overload.", "success": False}

            except ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return {"content": f"Connection failed after {max_retries} attempts: {str(e)}", "success": False}

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")

                # 特殊处理 TaskGroup 错误
                if "TaskGroup" in str(e) or "unhandled errors" in str(e):
                    logger.warning("TaskGroup error detected, treating as connection issue")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return {
                            "content": f"TaskGroup error after {max_retries} attempts. This may be due to asyncio event loop conflicts. Error: {str(e)}",
                            "success": False
                        }

                if "sse_reader" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.info(f"SSE reader error detected, retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return {
                            "content": f"SSE connection failed after {max_retries} attempts. The remote server may be experiencing issues. Error: {str(e)}",
                            "success": False
                        }

                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return {"content": f"Error after {max_retries} attempts: {str(e)}", "success": False}

        # 如果所有重试都失败了
        return {"content": "All retry attempts failed", "success": False}

async def list_all_mcp_tools(sse_url,timeout=3600,
                    sse_read_timeout=60*10,
                    initialize_timeout=30,  # 增加初始化超时
                    call_timeout=60*8,  # 增加到8分钟
                ):
    async with sse_client(
                    sse_url,
                    timeout=timeout,
                    sse_read_timeout=sse_read_timeout
                ) as streams:
                    async with ClientSession(*streams) as session:
                        await asyncio.wait_for(session.initialize(), timeout=initialize_timeout)

                        res = await asyncio.wait_for(
                            session.list_tools(),
                            timeout=call_timeout
                        )

                        results = res.tools#res.content[0].text
    return results
