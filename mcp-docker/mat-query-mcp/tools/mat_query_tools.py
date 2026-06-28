"""Materials Query Tools - 统一的材料查询工具模块

包含Materials Project和OQMD数据库的查询工具函数
"""

import glob
import json
import asyncio
import httpx
import pandas as pd
from io import StringIO
from typing import Dict, Any, Union, List, Annotated
from bs4 import BeautifulSoup

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from mp_api.client import MPRester

from core import llm_tool
from core.utils import extract_cif_info, remove_symmetry_equiv_xyz, validate_chemical_formula
from config import mp_config, server_config


async def get_mpid_from_formula(formula: str) -> Union[List[str], str]:
    """
    根据化学式从Materials Project数据库获取材料ID (mpid)
    返回最低能量结构的mpids

    Args:
        formula: 化学式 (例如: "Fe2O3")

    Returns:
        材料ID列表或错误信息字符串
    """
    try:
        # 输入验证
        if not formula or not isinstance(formula, str):
            return "Error: 化学式不能为空且必须是字符串"

        # 清理化学式格式
        cleaned_formula = formula.replace(" ", "").replace("\n", "").replace("'", "").replace('"', "")

        # 处理可能的等号分割格式
        if "=" in cleaned_formula:
            name, id = cleaned_formula.split("=")
            target_formula = id.strip()
        else:
            target_formula = cleaned_formula

        # 验证化学式格式
        try:
            target_formula = validate_chemical_formula(target_formula)
        except ValueError as e:
            return f"Error: {str(e)}"

        # 检查API密钥
        if not mp_config.MP_API_KEY:
            return "Error: 未设置 MP_API_KEY"

        # 使用Materials Project API查询
        formula_list = [target_formula]
        id_list = []

        with MPRester(mp_config.MP_API_KEY) as mpr:
            docs = mpr.materials.summary.search(formula=formula_list)
            if not docs:
                return f"Error:未找到化学式 '{target_formula}' 对应的材料"
            else:
                for doc in docs:
                    id_list.append(doc.material_id)
                return id_list

    except Exception as e:
        return f"Error: get_mpid_from_formula: {str(e)}"


@llm_tool(name="search_crystal_structures_from_materials_project",
          description="从Materials Project数据库检索和优化晶体结构，使用化学式查询")
async def search_crystal_structures_from_materials_project(
    formula: str,
    conventional_unit_cell: bool = True,
    symprec: float = 0.1
) -> str:
    """
    根据给定化学式从Materials Project数据库检索晶体结构并应用对称性优化

    Args:
        formula: 要搜索的化学式 (例如: "Fe2O3")
        conventional_unit_cell: 如果为True，返回常规单胞；如果为False，返回原胞
        symprec: 结构细化的对称性精度参数 (默认: 0.1)

    Returns:
        检索到的晶体结构的格式化CIF数据，包含对称性分析
    """
    try:
        structures = {}
        mp_id_list = await get_mpid_from_formula(formula=formula)
        if isinstance(mp_id_list, str):
            return mp_id_list  # 直接返回错误信息

        for i, mp_id in enumerate(mp_id_list):
            try:
                # 文件操作可能引发异常
                cif_files = glob.glob(mp_config.LOCAL_MP_CIF_ROOT + f"/{mp_id}.cif")
                if not cif_files:
                    continue  # 如果没有找到文件，跳过这个mp_id

                cif_file = cif_files[0]
                structure = Structure.from_file(cif_file)

                # 结构处理可能引发异常
                if conventional_unit_cell:
                    structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

                # 对结构进行对称化处理
                sga = SpacegroupAnalyzer(structure, symprec=symprec)
                symmetrized_structure = sga.get_refined_structure()

                # 使用CifWriter生成CIF数据
                cif_writer = CifWriter(symmetrized_structure, symprec=symprec, refine_struct=True)
                cif_data = str(cif_writer)

                # 删除CIF文件中的对称性操作部分
                cif_data = remove_symmetry_equiv_xyz(cif_data)
                cif_data = cif_data.replace('# generated using pymatgen', "")

                # 生成一个唯一的键
                formula_key = structure.composition.reduced_formula
                key = f"{formula_key}_{i}"

                structures[key] = cif_data

                # 只保留前MP_TOPK个结果
                if len(structures) >= mp_config.MP_TOPK:
                    break

            except (FileNotFoundError, IndexError) as file_error:
                # 处理文件相关错误
                continue  # 跳过这个mp_id，继续处理下一个
            except ValueError as value_error:
                # 处理结构处理中的值错误
                continue  # 跳过这个mp_id，继续处理下一个
            except Exception as process_error:
                # 记录处理特定结构时的错误，但继续处理其他结构
                print(f"Error: 处理结构 {mp_id} 时出错: {str(process_error)}")
                continue

        # 如果没有成功处理任何结构
        if not structures:
            return f"Error:未找到化学式 {formula} 对应的有效晶体结构"

        # 格式化结果为可读字符串
        prompt = f"# Materials Project 晶体结构数据\n\n"

        for i, (key, cif_data) in enumerate(structures.items(), 1):
            prompt += f"--- Structure {i} ({key}) ---\n"
            prompt += cif_data
            prompt += f"\n\n"

        return prompt

    except Exception as e:
        # 捕获整个函数执行过程中的任何未处理异常
        return f"Error: 处理晶体结构时发生意外错误: {str(e)}"


@llm_tool(name="search_material_property_from_materials_project",
          description="使用化学式从Materials Project数据库查询材料属性")
async def search_material_property_from_materials_project(
        formula: str,
    ) -> str:
    """
    根据化学式从Materials Project数据库检索匹配材料的详细属性数据

    Args:
        formula: 要搜索的材料化学式 (例如: 'Fe2O3', 'LiFePO4')

    Returns:
        包含材料属性的格式化字符串，包括结构、电子、热力学和机械数据
    """
    # 获取MP ID列表
    mp_id_list = await get_mpid_from_formula(formula=formula)

    # 检查get_mpid_from_formula的返回值类型
    # 如果返回的是字符串，说明发生了错误或没有找到材料
    if isinstance(mp_id_list, str):
        return mp_id_list  # 直接返回错误信息

    # 如果代码执行到这里，说明mp_id_list是一个有效的ID列表
    try:
        # 获取材料属性
        properties = []
        for mp_id in mp_id_list:
            try:
                file_path = mp_config.LOCAL_MP_PROPS_ROOT + f"/{mp_id}.json"
                crystal_props = extract_cif_info(file_path, ['all_fields'])
                properties.append(crystal_props)
            except Exception as file_error:
                # 记录单个文件处理错误但继续处理其他ID
                continue

        # 检查是否有结果
        if len(properties) == 0:
            return "未找到给定化学式的材料属性，请重试。"

        # 只保留前MP_TOPK个结果
        properties = properties[:mp_config.MP_TOPK]

        # 格式化结果
        formatted_results = []
        for i, item in enumerate(properties, 1):
            formatted_result = f"--- Material {i} Properties ---\n"
            formatted_result += json.dumps(item, indent=2, ensure_ascii=False)
            formatted_result += f"\n\n"
            formatted_results.append(formatted_result)

        # 将所有结果合并为一个字符串
        res_chunk = "".join(formatted_results)
        res_template = f"# Materials Project 材料属性数据\n\n{res_chunk}"
        return res_template

    except Exception as e:
        return f"Error: 处理材料属性时出错: {str(e)}"


@llm_tool(name="query_material_from_OQMD", description="Query material properties by chemical formula from OQMD database")
async def query_material_from_OQMD(
    formula: Annotated[str, "Chemical formula (e.g., Fe2O3, LiFePO4)"]
) -> str:
    """
    Query material information by chemical formula from OQMD database.

    Args:
        formula: Chemical formula of the material (e.g., Fe2O3, LiFePO4)

    Returns:
        Formatted text with material information and property tables
    """
    # Fetch data from OQMD
    url = f"https://www.oqmd.org/materials/composition/{formula}"
    try:
        async with httpx.AsyncClient(timeout=server_config.TOOL_CALL_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Validate response content
            if not response.text or len(response.text) < 100:
                raise ValueError("Invalid response content from OQMD API")

            # Parse HTML data
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            # Parse basic data
            basic_data = []
            h1_element = soup.find('h1')
            if h1_element:
                basic_data.append(h1_element.text.strip())
            else:
                basic_data.append(f"Material: {formula}")

            for script in soup.find_all('p'):
                if script:
                    combined_text = ""
                    for element in script.contents:
                        if hasattr(element, 'name') and element.name == 'a':
                            # 只保留链接文本，移除URL
                            combined_text += element.text.strip() + " "
                        elif hasattr(element, 'text'):
                            combined_text += element.text.strip() + " "
                        else:
                            combined_text += str(element).strip() + " "
                    basic_data.append(combined_text.strip())

            # Parse table data
            table_data = ""
            table = soup.find('table')
            if table:
                try:
                    df = pd.read_html(StringIO(str(table)))[0]
                    df = df.fillna('')
                    df = df.replace([float('inf'), float('-inf')], '')
                    table_data = df.to_markdown(index=False)
                except Exception as e:
                    table_data = "Error: parsing OQMD table data"

            # Integrate data into a single text
            result = f"# OQMD 材料数据\n\n"

            # 只保留有用的基础信息，过滤空白内容
            useful_data = [data for data in basic_data if data.strip() and len(data.strip()) > 5]
            if useful_data:
                result += "\n\n".join(useful_data) + "\n\n"

            # 添加属性表格
            if table_data:
                result += "## 材料属性\n\n" + table_data

            return result

    except httpx.HTTPStatusError as e:
        return f"Error: OQMD API request failed - {str(e)}"
    except httpx.TimeoutException:
        return "Error: OQMD API request timed out"
    except httpx.NetworkError as e:
        return f"Error: Network error occurred - {str(e)}"
    except ValueError as e:
        return f"Error: Invalid response content - {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error occurred - {str(e)}"
