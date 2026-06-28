## From Agent-R1,and modified by lzy,add MCPToolBase

from abc import ABC, abstractmethod
import time
from typing import Dict, List, Any, Tuple
from jsonschema import validate, ValidationError
from . import default_sse_timeout,default_timeout

import jsonschema
import asyncio
import time
from mcp.client.sse import sse_client
from mcp import ClientSession

def is_tool_schema(obj: dict) -> bool:
    """
    Check if obj is a valid JSON schema describing a tool compatible with OpenAI's tool calling.
    Example valid schema:
    {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
    """
    try:
        assert set(obj.keys()) == {'name', 'description', 'parameters'}
        assert isinstance(obj['name'], str)
        assert obj['name'].strip()
        assert isinstance(obj['description'], str)
        assert isinstance(obj['parameters'], dict)

        assert set(obj['parameters'].keys()) == {'type', 'properties', 'required'}
        assert obj['parameters']['type'] == 'object'
        assert isinstance(obj['parameters']['properties'], dict)
        assert isinstance(obj['parameters']['required'], list)
        assert set(obj['parameters']['required']).issubset(set(obj['parameters']['properties'].keys()))
    except AssertionError:
        return False
    try:
        jsonschema.validate(instance={}, schema=obj['parameters'])
    except jsonschema.exceptions.SchemaError:
        return False
    except jsonschema.exceptions.ValidationError:
        pass
    return True

class BaseTool(ABC):
    name: str = ''
    description: str = ''
    parameters: dict = {}

    def __init__(self):
        if not self.name:
            raise ValueError('Tool name must be provided')
        if not is_tool_schema({'name': self.name, 'description': self.description, 'parameters': self.parameters}):
            raise ValueError(
                'The parameters, when provided as a dict, must confirm to a valid openai-compatible JSON schema.')

    @abstractmethod
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        pass

    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        return [self.execute(args, **kwargs) for args in args_list]

    @property
    def tool_info(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters
        }

    @property
    def tool_description(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def validate_args(self, args: Dict) -> bool:
        try:
            validate(instance=args, schema=self.parameters)
            return True
        except ValidationError:
            return False


class MCPToolBase(BaseTool):
    """
    MCP工具基类，不需要在初始化时提供URL
    配合工厂模式使用，URL等配置由工厂注入
    """
    def __init__(self):
        super().__init__()
        # 这些属性将由工厂设置
        self.sse_url = None
        self.sse_timeout = None
        self.timeout = None

    def _configure(self, sse_url: str, sse_timeout: int = default_sse_timeout, timeout: int = default_timeout):
        """
        由工厂调用来配置工具实例

        Args:
            sse_url: SSE服务器URL
            sse_timeout: SSE超时时间
            timeout: 一般超时时间
        """
        self.sse_url = sse_url
        self.sse_timeout = sse_timeout
        self.timeout = timeout

    def _ensure_configured(self):
        """确保工具已经被正确配置"""
        if self.sse_url is None:
            raise RuntimeError(f"工具 {self.__class__.__name__} 未正确配置，请使用工厂创建实例")

    async def _call_mcp_tool(self, tool_name: str, args: Dict) -> Dict[str, Any]:
        """调用MCP工具，确保工具已配置"""
        self._ensure_configured()
        import logging
        import os
        logger = logging.getLogger(__name__)

        is_local_connection = 'localhost' in self.sse_url or '127.0.0.1' in self.sse_url or '10.' in self.sse_url
         # 如果连接内网服务器，临时禁用代理
        original_proxy_env = {}
        if is_local_connection:
            # 保存原始代理设置
            proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
            for var in proxy_vars:
                if var in os.environ:
                    original_proxy_env[var] = os.environ[var]
                    del os.environ[var]
            #print(f"[MCP工具调用] 检测到本地连接，已临时禁用代理")
        # 记录开始时间和打印工具名称
        start_time = time.perf_counter()
        try:
            async with sse_client(
                            self.sse_url,
                            timeout=self.timeout,
                            sse_read_timeout=self.sse_timeout+5
                        ) as streams:
                            async with ClientSession(*streams) as session:
                                await asyncio.wait_for(session.initialize(), timeout=self.timeout)
                                #logger.info("SSE session initialized successfully")
                                print(f"[MCP工具调用] 开始调用工具: {tool_name}")
                                print(f"[MCP工具调用] 调用参数: {args}")
                                res = await asyncio.wait_for(
                                    session.call_tool(tool_name, args),
                                    timeout=self.sse_timeout
                                )
                                results = res.content[0].text

                                end_time = time.perf_counter()
                                elapsed_time = end_time - start_time
                                if 'error' in results.lower():

                                    print(f"[MCP工具调用] 工具 {tool_name} 调用失败，耗时: {elapsed_time:.3f}秒，错误信息: {results}")
                                    return {"content": results, "success": False}
                                else:
                                    result_without_space= '\n'.join([line for line in results.splitlines() if line.strip()])
                                    print(f"[MCP工具调用] 工具 {tool_name} 调用成功，耗时: {elapsed_time:.3f}秒，返回结果: {result_without_space[:300]}...")
                                # 如果能通过参数检测，出现error其实时MCP调用太频繁或者超时的问题 ，其实不算是模型调用失败。。。
                                    return {"content": results, "success": True}
        except Exception as e:
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    logger.error(f"Error occurred while calling MCP tool {tool_name}: {e}")
                    print(f"[MCP工具调用] 工具 {tool_name} 调用异常，耗时: {elapsed_time:.3f}秒，异常信息: {e}")
                    return {"content": str(e), "success": False}

        finally:
            # 恢复原始代理设置
            if is_local_connection and original_proxy_env:
                for var, value in original_proxy_env.items():
                    os.environ[var] = value

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

        # # 检查工具是否定义了参数规范
        # if not hasattr(self, 'parameters') or 'properties' not in self.parameters:
        #     # 如果没有定义参数规范，则认为验证通过
        #     return {"valid": True, "error": None}

        # 获取工具定义的参数信息
        valid_params = set(self.parameters['properties'].keys())
        provided_params = set(args.keys())
        required_params = set(self.parameters.get('required', []))

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
                structure = Structure.from_str(cif_string, fmt='cif')
            except Exception as e:
                return {
                "valid": False,
                "error": f"Error: Invalid cif_string format. Please provide a valid CIF string or file content. Error: {str(e)}"
            }
        return {"valid": True, "error": None}

    async def execute(self, args: Dict) -> Dict[str, Any]:
        """
        子类需要实现此方法，调用_call_mcp_tool并传入适当的工具名称和参数
        """
        raise NotImplementedError("Subclasses must implement execute method")
