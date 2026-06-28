# MCP tools for materials  ,by lzy

from typing import Any, Dict
from .base import MCPToolBase


class SearchCrystalStructuresFromMaterialsProjectTool(MCPToolBase):
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

    async def execute(self, args: Dict) -> Dict[str, Any]:

        formula = args["formula"]
        conventional_unit_cell = args.get("conventional_unit_cell", True)
        symprec = args.get("symprec", 0.1)

        return await self._call_mcp_tool('search_crystal_structures_from_materials_project', {
            'formula': formula,
            'conventional_unit_cell': conventional_unit_cell,
            'symprec': symprec
        })

class SearchMaterialPropertyFromMaterialsProjectTool(MCPToolBase):
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

    async def execute(self, args: Dict) -> Dict[str, Any]:

        formula = args["formula"]

        return await self._call_mcp_tool('search_material_property_from_materials_project', {
            'formula': formula
        })

class GenerateCrystalStructuresCrystallmTool(MCPToolBase):
    name ='generate_crystal_structures_crystallm'
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

    async def execute(self, args: Dict) -> Dict[str, Any]:

        formula = args["formula"]
        space_group = args.get("space_group")
        num_samples = args.get("num_samples", 2)

        return await self._call_mcp_tool('generate_crystal_structures_crystallm', {
            'formula': formula,
            'space_group': space_group,
            'num_samples': num_samples
        })

class PredictFormationEnergyMatGLTOOL(MCPToolBase):
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

    async def execute(self, args: Dict) -> Dict[str, Any]:
        cif_string = args["cif_string"]
        optimize_structure = args.get("optimize_structure", True)
        fmax = args.get("fmax", 0.01)

        return await self._call_mcp_tool('predict_formation_energy_MatGL', {
            'cif_string': cif_string,
            'optimize_structure': optimize_structure,
            'fmax': fmax
        })

class PredictMultiFidelityBandGapMatGLTOOL(MCPToolBase):
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
    async def execute(self, args: Dict) -> Dict[str, Any]:
        cif_string = args["cif_string"]
        optimize_structure = args.get("optimize_structure", True)
        fmax = args.get("fmax", 0.01)

        return await self._call_mcp_tool('predict_multi_fidelity_band_gap_MatGL', {
            'cif_string': cif_string,
            'optimize_structure': optimize_structure,
            'fmax': fmax
        })

class AnalyzeThermodynamicStabilityPymatgenTOOL(MCPToolBase):
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
    async def execute(self, args: Dict) -> Dict[str, Any]:
        cif_string = args["cif_string"]
        energy_threshold = args.get("energy_threshold",0.025)
        optimize_structure = args.get("optimize_structure", True)


        return await self._call_mcp_tool('analyze_thermodynamic_stability_pymatgen', {
            'cif_string': cif_string,
            'energy_threshold':energy_threshold,
            'optimize_structure': optimize_structure,

        })
