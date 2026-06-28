from verl.tools.base_tool import BaseTool
from typing import Dict, List, Any, Tuple
import asyncio
import time
from ipaddress import ip_address
from urllib.parse import urlparse
from mcp.client.sse import sse_client
from mcp import ClientSession
from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.schemas import ToolResponse
class MCPSSETool(BaseTool):
    """
    基于SSE连接的MCP工具基类，处理通用的连接、重试和错误处理逻辑
    """

    def __init__(self, config,tool_schema):
        super().__init__(config,tool_schema)
        self.sse_url = config.get('sse_url')
        self.toolcall_timeout = config.get('default_timeout',180)
        self.max_retries = 1
        self.retry_delay = 5
        self.parameters = tool_schema.function
        #print("参数",self.parameters)
    async def _call_mcp_tool(self, tool_name: str, args: Dict) -> Dict[str, Any]:
        """
        调用MCP工具的通用方法，处理连接、重试和错误
        """
        import logging
        import os
        logger = logging.getLogger(__name__)

        # 记录开始时间和打印工具名称
        start_time = time.perf_counter()


        retry_delay = self.retry_delay

        # 如果连接本地或私有网络服务器，临时禁用代理
        original_proxy_env = {}
        parsed_host = urlparse(self.sse_url).hostname or ""
        try:
            is_private_host = ip_address(parsed_host).is_private
        except ValueError:
            is_private_host = False
        is_local_connection = parsed_host in {"localhost", "127.0.0.1", "::1"} or is_private_host

        if is_local_connection:
            # 保存原始代理设置
            proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
            for var in proxy_vars:
                if var in os.environ:
                    original_proxy_env[var] = os.environ[var]
                    del os.environ[var]
            #print(f"[MCP工具调用] 检测到本地连接，已临时禁用代理")

        try:
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Attempting {tool_name} call, attempt {attempt + 1}/{self.max_retries}")

                    async with sse_client(
                        self.sse_url,
                        timeout=30,
                        sse_read_timeout=self.toolcall_timeout+10
                    ) as streams:
                        async with ClientSession(*streams) as session:
                            await asyncio.wait_for(session.initialize(), timeout=30)
                            #logger.info("SSE session initialized successfully")
                            print(f"[MCP工具调用] 开始调用工具: {tool_name}")
                            print(f"[MCP工具调用] 调用参数: {args}")
                            res = await asyncio.wait_for(
                                session.call_tool(tool_name, args),
                                timeout=self.toolcall_timeout
                            )
                            results = res.content[0].text

                            end_time = time.perf_counter()
                            elapsed_time = end_time - start_time
                            print(f"[MCP工具调用] 工具 {tool_name} 调用成功并已成功返回结果，耗时: {elapsed_time:.3f}秒")

                            return {"content": results, "success": True}

                except asyncio.TimeoutError as e:
                    logger.warning(f"Timeout error on attempt {attempt + 1}: {str(e)}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        # 计算并打印耗时
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"[MCP工具调用] 工具 {tool_name} 调用超时失败，耗时: {elapsed_time:.3f}秒")
                        return {"content": f"Request timed out after {self.max_retries} attempts. This may be due to network issues or server overload.", "success": False}

                except ConnectionError as e:
                    logger.warning(f"Connection error on attempt {attempt + 1}: {str(e)}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        # 计算并打印耗时
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"[MCP工具调用] 工具 {tool_name} 连接失败，耗时: {elapsed_time:.3f}秒")
                        return {"content": f"Connection failed after {self.max_retries} attempts: {str(e)}", "success": False}

                except Exception as e:
                    logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                    print(f"[DEBUG] 详细错误信息: {type(e).__name__}: {str(e)}")
                    import traceback
                    print(f"[DEBUG] 错误堆栈: {traceback.format_exc()}")

                    # 特殊处理 TaskGroup 错误
                    if "TaskGroup" in str(e) or "unhandled errors" in str(e):
                        logger.warning("TaskGroup error detected, treating as connection issue")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                        else:
                            # 计算并打印耗时
                            end_time = time.perf_counter()
                            elapsed_time = end_time - start_time
                            error =f'[MCP工具调用] 工具 {tool_name}，请求已发送，但由于工具端服务压力过大，无法及时返回结果。'
                            #error = f"[MCP工具调用] 工具 {tool_name} 由于TaskGroup调用失败，TaskGroup error after {self.max_retries} attempts. This may be due to asyncio event loop conflicts. Error: {str(e)}"
                            logger.warning(error+f"耗时: {elapsed_time:.3f}秒")
                            return {
                                "content":error,
                                "success": False
                            }

                    if "sse_reader" in str(e).lower():
                        if attempt < self.max_retries - 1:
                            logger.info(f"SSE reader error detected, retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                        else:
                            # 计算并打印耗时
                            end_time = time.perf_counter()
                            elapsed_time = end_time - start_time
                            error = f"[MCP工具调用] 工具 {tool_name} SSE连接失败."
                            logger.warning(error+f"耗时: {elapsed_time:.3f}秒")
                            return {
                                "content": error+ f"The remote server may be experiencing issues. Error: {str(e)}",
                                "success": False
                            }

                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        # 计算并打印耗时
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        error=f"[MCP工具调用] 工具 {tool_name} 调用失败."
                        logger.warning(error+f"耗时: {elapsed_time:.3f}秒")

                        return {"content": error+f"Due to unknown error: {type(e).__name__}: {str(e)}", "success": False}

            # 如果所有重试都失败了
            # 计算并打印耗时
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            error = f"[MCP工具调用] 工具 {tool_name} 所有重试失败."
            logger.warning(error+"耗时: {elapsed_time:.3f}秒")
            return {"content": error, "success": False}

        finally:
            # 恢复原始代理设置
            if is_local_connection and original_proxy_env:
                for var, value in original_proxy_env.items():
                    os.environ[var] = value
                #print(f"[MCP工具调用] 已恢复代理设置")

    def _validate_args(self, args: Dict) -> Dict[str, Any]:
        """
        验证参数是否合法：
        1. 检查参数是否为字典
        2. 检查所有提供的参数是否都在工具定义的参数列表中
        3. 检查所有必需的参数是否都已提供
        返回: {"valid": bool, "error": str or None}
        """
        # 检查 args 是否为 None
        if args is None:
            return {"valid": False, "error": "Error: Arguments cannot be None"}

        # 检查 args 是否为字典
        if not isinstance(args, dict):
            return {"valid": False, "error": f"Error: Arguments must be a dictionary, got {type(args).__name__}"}

        # 检查工具是否定义了参数规范
        if not hasattr(self, 'parameters') or not hasattr(self.parameters, 'parameters'):
            # 如果没有定义参数规范，则认为验证通过
            return {"valid": True, "error": None}

        # 获取工具定义的参数信息 - 使用属性访问而不是字典访问
        valid_params = set(self.parameters.parameters.properties.keys())
        provided_params = set(args.keys())
        required_params = set(self.parameters.parameters.required)

        # 检查是否提供了无效的参数
        invalid_params = provided_params - valid_params
        if invalid_params:
            return {
                "valid": False,
                "error": f"Error: Invalid parameters provided: {', '.join(sorted(invalid_params))}. Valid parameters are: {', '.join(sorted(valid_params))}"
            }

        # 检查是否缺少必需的参数
        missing_params = required_params - provided_params
        if missing_params:
            return {
                "valid": False,
                "error": f"Error: Missing required parameters: {', '.join(sorted(missing_params))}."
            }

        # 检查必需参数的值是否为空
        empty_required_params = []
        for param in required_params:
            if param in args:
                value = args[param]
                # 检查值是否为空（None、空字符串、空列表、空字典等）
                if value is None or (isinstance(value, (str, list, dict)) and len(value) == 0):
                    empty_required_params.append(param)

        if empty_required_params:
            return {
                "valid": False,
                "error": f"Error: Required parameters cannot be empty: {', '.join(sorted(empty_required_params))}"
            }
        if "cif_string" in required_params:
            try:
                # 尝试解析结构源参数为 pymatgen 结构
                from pymatgen.core import Structure
                cif_string = args["cif_string"]
                structure=Structure.from_str(cif_string, fmt='cif')
            except Exception as e:
                return {
                "valid": False,
                "error": f"Error: Invalid cif_string format. Please provide a valid and complete CIF string. Error: {str(e)}"
            }
        return {"valid": True, "error": None}

    async def execute(self, instance_id: str, args: Dict) -> Dict[str, Any]:
        """
        子类需要实现此方法，调用_call_mcp_tool并传入适当的工具名称和参数
        """
        raise NotImplementedError("Subclasses must implement execute method")


class SearchCrystalStructuresFromMaterialsProjectTool(MCPSSETool):
    name = 'search_crystal_structures_from_materials_project'
    description = 'Retrieve and optimize crystal structures from Materials Project database using a chemical formula'
    parameters = {
        'type': 'object',
        'properties': {
            'formula': {
                'type': 'string',
                'description': 'Chemical formula to search for (e.g., "Fe2O3")',
                'title': 'Formula'
            },
            'conventional_unit_cell': {
                'type': 'boolean',
                'description': 'If True, returns conventional unit cell; if False, returns primitive cell',
                'title': 'Conventional Unit Cell',
                'default': True
            },
            'symprec': {
                'type': 'number',
                'description': 'Symmetry precision parameter for structure refinement (default: 0.1)',
                'title': 'Symprec',
                'default': 0.1
            }
        },
        'required': ['formula']
    }
    @rollout_trace_op
    async def execute(self, instance_id: str, args: Dict) -> Dict[str, Any]:
        # 验证参数
        validation_result = self._validate_args(args)
        if not validation_result["valid"]:
            return {"content": validation_result["error"], "success": False}

        formula = args["formula"]
        conventional_unit_cell = args.get("conventional_unit_cell", True)
        symprec = args.get("symprec", 0.1)
        tool_results = await self._call_mcp_tool('search_crystal_structures_from_materials_project', {
            'formula': formula,
            'conventional_unit_cell': conventional_unit_cell,
            'symprec': symprec
        })
        tool_content = tool_results['content']
        toolcall_success = tool_results['success']

        # TODO:LZY 后面考虑是否要返回tool_reward_score 20251015
        # Returns: tool_response, tool_reward_score, tool_metrics
        #     tool_response: The ToolResponse object containing text, image, and/or video content.
        #     tool_reward_score: The step reward score of the tool.
        #     tool_metrics: The metrics of the tool.
        return  ToolResponse(text = tool_content),None,None


class SearchMaterialPropertyFromMaterialsProjectTool(MCPSSETool):
    name = 'search_material_property_from_materials_project'
    description = 'Query material properties from Materials Project database using chemical formula'
    parameters = {
        'type': 'object',
        'properties': {
            'formula': {
                'type': 'string',
                'description': "Chemical formula of the material(s) to search for (e.g. 'Fe2O3', 'LiFePO4')",
                'title': 'Formula'
            }
        },
        'required': ['formula']
    }

    @rollout_trace_op
    async def execute(self, instance_id: str, args: Dict) -> Dict[str, Any]:
        validation_result = self._validate_args(args)
        if not validation_result["valid"]:
            return ToolResponse(text=validation_result["error"]), None, None

        formula = args["formula"]
        tool_results = await self._call_mcp_tool(self.name, {'formula': formula})
        return ToolResponse(text=tool_results['content']), None, None

class GenerateCrystalStructuresCrystallmTool(MCPSSETool):
    name = 'generate_crystal_structures_crystallm'
    description = "Generate crystal structures using CrystaLLM model based on chemical formula and optional space group"
    parameters = {
        'type': 'object',
        'properties': {
            'formula': {
                'type': 'string',
                'description': "Chemical formula (e.g., 'CsPbBr3', 'Na2Cl2'). Elements must be sorted by electronegativity.",
                'title': 'Formula'
            },
            'space_group': {
                'type': 'string',
                'description': "Optional space group symbol (e.g., 'P4/nmm', 'Fd-3m')",
                'title': 'Space Group'
            },
            'num_samples': {
                'type': 'integer',
                'description': "Number of crystal structures to generate (default: 2)",
                'title': 'Number of Samples',
                'default': 2,
            }
        },
        'required': ['formula']
    }

    @rollout_trace_op
    async def execute(self, instance_id: str, args: Dict) -> Dict[str, Any]:
        validation_result = self._validate_args(args)
        if not validation_result["valid"]:
            return ToolResponse(text=validation_result["error"]), None, None

        formula = args["formula"]
        space_group = args.get("space_group")
        num_samples = args.get("num_samples", 2)
        tool_results = await self._call_mcp_tool(self.name, {
            'formula': formula,
            'space_group': space_group,
            'num_samples': num_samples
        })
        return ToolResponse(text=tool_results['content']), None, None

class PredictFormationEnergyMatGLTOOL(MCPSSETool):
    name = 'predict_formation_energy_MatGL'
    description = "Predict formation energy of crystal structures using MatGL M3GNet model"
    parameters = {
        'type': 'object',
        'properties': {
            'cif_string': {
                'type': 'string',
                'description': "Complete CIF format crystal structure string or file content",
                'title': 'CIF String'
            },
            'optimize_structure': {
                'type': 'boolean',
                'description': "Whether to optimize structure before prediction (default: True)",
                'title': 'Optimize Structure',
                'default': True
            },
            'fmax': {
                'type': 'number',
                'description': "Force convergence threshold for structure optimization",
                'title': 'Force Convergence Threshold',
                'default': 0.01,
            }
        },
        'required': ['cif_string']
    }

    @rollout_trace_op
    async def execute(self, instance_id: str, args: Dict) -> Dict[str, Any]:
        validation_result = self._validate_args(args)
        if not validation_result["valid"]:
            return ToolResponse(text=validation_result["error"]), None, None

        cif_string = args["cif_string"]
        optimize_structure = args.get("optimize_structure", True)
        fmax = args.get("fmax", 0.01)
        tool_results = await self._call_mcp_tool(self.name, {
            'cif_string': cif_string,
            'optimize_structure': optimize_structure,
            'fmax': fmax
        })
        return ToolResponse(text=tool_results['content']), None, None

class PredictMultiFidelityBandGapMatGLTOOL(MCPSSETool):
    name = 'predict_multi_fidelity_band_gap_MatGL'
    description = "Predict band gap of crystal structures using MatGL MEGNet multi-fidelity model"
    parameters = {
        'type': 'object',
        'properties': {
            'cif_string': {
                'type': 'string',
                'description': "CIF format crystal structure string or file content",
                'title': 'CIF String'
            },
            'optimize_structure': {
                'type': 'boolean',
                'description': "Whether to optimize structure before prediction (default: True)",
                'title': 'Optimize Structure',
                'default': True
            },
            'fmax': {
                'type': 'number',
                'description': "Force convergence threshold for structure optimization",
                'title': 'Force Convergence Threshold',
                'default': 0.01,
            }
        },
        'required': ['cif_string']
    }

    @rollout_trace_op
    async def execute(self, instance_id: str, args: Dict) -> Dict[str, Any]:
        validation_result = self._validate_args(args)
        if not validation_result["valid"]:
            return ToolResponse(text=validation_result["error"]), None, None

        cif_string = args["cif_string"]
        optimize_structure = args.get("optimize_structure", True)
        fmax = args.get("fmax", 0.01)
        tool_results = await self._call_mcp_tool(self.name, {
            'cif_string': cif_string,
            'optimize_structure': optimize_structure,
            'fmax': fmax
        })
        return ToolResponse(text=tool_results['content']), None, None

class AnalyzeThermodynamicStabilityPymatgenTOOL(MCPSSETool):
    name = 'analyze_thermodynamic_stability_pymatgen'
    description = "Analyze thermodynamic stability of materials"
    parameters = {
        'type': 'object',
        'properties': {
            'cif_string': {
                'type': 'string',
                'description': "CIF format crystal structure content string",
                'title': 'CIF String'
            },
            'energy_threshold': {
                'type': 'number',
                'description': "Energy threshold (eV/atom), default uses 0.025 from configuration",
                'title': 'Energy Threshold',
                'default': 0.025,
                'minimum': 0.0,
            },
            'optimize_structure': {
                'type': 'boolean',
                'description': "Whether to optimize structure before energy prediction, default False",
                'title': 'Optimize Structure',
                'default': False
            }
        },
        'required': ['cif_string']
    }

    @rollout_trace_op
    async def execute(self, instance_id: str, args: Dict) -> Dict[str, Any]:
        validation_result = self._validate_args(args)
        if not validation_result["valid"]:
            return ToolResponse(text=validation_result["error"]), None, None

        cif_string = args["cif_string"]
        energy_threshold = args.get("energy_threshold", 0.025)
        optimize_structure = args.get("optimize_structure", False)
        tool_results = await self._call_mcp_tool(self.name, {
            'cif_string': cif_string,
            'energy_threshold': energy_threshold,
            'optimize_structure': optimize_structure,
        })
        return ToolResponse(text=tool_results['content']), None, None


class CheckStructureAtomicGeometryPymatgenTool(MCPSSETool):
    name = 'check_structure_atomic_geometry_pymatgen'
    description = "使用PyMatGen检查晶体结构的原子几何合理性，包括原子重叠和键长分析"
    parameters = {
        'type': 'object',
        'properties': {
            'cif_string': {
                'type': 'string',
                'description': 'CIF格式的晶体结构内容字符串'
            },
            'min_distance_factor': {
                'type': 'number',
                'description': '最小距离因子，相对于原子半径和 (默认: 0.5)',
                'default': 0.5
            },
            'bond_tolerance': {
                'type': 'number',
                'description': '键长容差比例 (默认: 0.3，即±30%)',
                'default': 0.3
            },
            'max_neighbors_distance': {
                'type': 'number',
                'description': '搜索近邻的最大距离 (埃) (默认: 5.0)',
                'default': 5.0
            }
        },
        'required': ['cif_string']
    }

    @rollout_trace_op
    async def execute(self, instance_id: str, args: Dict) -> Dict[str, Any]:
        validation_result = self._validate_args(args)
        if not validation_result["valid"]:
            return ToolResponse(text=validation_result["error"]), None, None

        cif_string = args["cif_string"]
        min_distance_factor = args.get("min_distance_factor", 0.5)
        bond_tolerance = args.get("bond_tolerance", 0.3)
        max_neighbors_distance = args.get("max_neighbors_distance", 5.0)

        tool_results = await self._call_mcp_tool(self.name, {
            'cif_string': cif_string,
            'min_distance_factor': min_distance_factor,
            'bond_tolerance': bond_tolerance,
            'max_neighbors_distance': max_neighbors_distance,
        })
        return ToolResponse(text=tool_results['content']), None, None


class SimulateXrdPatternPymatgenTool(MCPSSETool):
    name = 'simulate_xrd_pattern_pymatgen'
    description = "使用PyMatGen模拟晶体结构的XRD衍射图谱，提供完整的衍射峰信息"
    parameters = {
        'type': 'object',
        'properties': {
            'cif_string': {
                'type': 'string',
                'description': 'CIF格式的晶体结构内容字符串'
            },
            'wavelength': {
                'type': 'string',
                'description': 'X射线波长类型，支持CuKa, MoKa, CrKa, FeKa, CoKa, AgKa等 (默认: "CuKa")',
                'default': 'CuKa'
            },
            'two_theta_range': {
                'type': 'array',
                'items': {'type': 'number'},
                'description': '2θ角度扫描范围，格式为(min_angle, max_angle) (默认: (10.0, 80.0))',
                'default': [10.0, 80.0]
            },
            'min_intensity_threshold': {
                'type': 'number',
                'description': '最小强度阈值，相对于最强峰的百分比 (默认: 0.5)',
                'default': 0.5
            }
        },
        'required': ['cif_string']
    }

    @rollout_trace_op
    async def execute(self, instance_id: str, args: Dict) -> Dict[str, Any]:
        validation_result = self._validate_args(args)
        if not validation_result["valid"]:
            return ToolResponse(text=validation_result["error"]), None, None

        cif_string = args["cif_string"]
        wavelength = args.get("wavelength", "CuKa")
        two_theta_range = args.get("two_theta_range", (10.0, 80.0))
        min_intensity_threshold = args.get("min_intensity_threshold", 0.5)

        tool_results = await self._call_mcp_tool(self.name, {
            'cif_string': cif_string,
            'wavelength': wavelength,
            'two_theta_range': two_theta_range,
            'min_intensity_threshold': min_intensity_threshold,
        })
        return ToolResponse(text=tool_results['content']), None, None
