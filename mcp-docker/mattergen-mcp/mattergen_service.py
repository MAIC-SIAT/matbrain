"""
Simplified MatterGen service for crystal structure generation
"""

import datetime
import os
import logging
import json
import re
import threading
import torch
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, Union

from mattergen_wrapper import *
from config.config import mattergen_config

logger = logging.getLogger(__name__)

def format_cif_content(content):
    """
    Improved CIF content formatting that handles both clean CIF content and ZIP artifacts.

    Args:
        content: String containing CIF content

    Returns:
        Formatted string with each CIF file properly labeled
    """
    if not content or content.strip() == '':
        return ''

    # Remove ZIP file artifacts and PK headers
    content = re.sub(r'PK.*?(?=data_|_chemical_formula_structural)', '', content, flags=re.DOTALL)
    content = re.sub(r'PK[^_]*$', '', content, flags=re.DOTALL)

    # Clean up any remaining binary artifacts
    content = re.sub(r'[^\x20-\x7E\n\r\t]', '', content)  # Remove non-printable characters

    # Split by data_ blocks (standard CIF format)
    data_blocks = re.split(r'(?=data_)', content)
    data_blocks = [block.strip() for block in data_blocks if block.strip()]

    if not data_blocks:
        # Fallback: try to split by _chemical_formula_structural
        formula_positions = [m.start() for m in re.finditer(r'_chemical_formula_structural', content)]
        if not formula_positions:
            return content.strip()  # Return as-is if no recognizable structure

        data_blocks = []
        for i in range(len(formula_positions)):
            start_pos = formula_positions[i]
            end_pos = formula_positions[i+1] if i < len(formula_positions)-1 else len(content)
            block = content[start_pos:end_pos].strip()
            if block:
                data_blocks.append(block)

    # Format output
    result = []
    for i, cif_block in enumerate(data_blocks, 1):
        # Extract formula for labeling
        formula_match = re.search(r'(?:data_(\w+)|_chemical_formula_structural\s+(\S+))', cif_block)
        if formula_match:
            formula = formula_match.group(1) or formula_match.group(2)
        else:
            formula = f"structure_{i}"

        # Ensure proper data_ header
        if not cif_block.startswith('data_'):
            cif_block = f"data_{formula}\n{cif_block}"

        formatted = f"[cif {i} begin]\n{cif_block}\n[cif {i} end]\n"
        result.append(formatted)

    return "\n".join(result)

class MatterGenService:
    """
    Simplified service for generating crystal structures using MatterGen.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of MatterGenService."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the MatterGenService."""
        self._generators = {}
        self._output_dir = mattergen_config.TEMP_ROOT

        # Ensure output directory exists
        os.makedirs(self._output_dir, exist_ok=True)

        # Initialize base generator
        self._init_base_generator()

    def _init_base_generator(self):
        """Initialize the base generator for unconditional generation."""
        model_path = os.path.join(mattergen_config.MATTERGENMODEL_ROOT, "mattergen_base")

        if not os.path.exists(model_path):
            logger.warning(f"Base model directory not found at {model_path}")
            return

        try:
            checkpoint_info = MatterGenCheckpointInfo(
                model_path=Path(model_path).resolve(),
                load_epoch="last",
                config_overrides=[],
                strict_checkpoint_loading=True,
            )

            generator = CrystalGenerator(
                checkpoint_info=checkpoint_info,
                properties_to_condition_on=None,
                batch_size=2,
                num_batches=1,
                sampling_config_name="default",
                sampling_config_path=None,
                sampling_config_overrides=[],
                record_trajectories=True,
                diffusion_guidance_factor=0.0,
                target_compositions_dict=[],
            )

            self._generators["base"] = generator
            logger.info("Base MatterGen generator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize base MatterGen generator: {e}")

    def _get_or_create_generator(self, properties=None, batch_size=2, num_batches=1, diffusion_guidance_factor=2.0):
        """
        Get or create a generator for the specified properties.

        Returns:
            tuple: (generator, generator_key, properties_to_condition_on, gpu_id)
        """
        # Use base generator if no properties
        if not properties:
            if "base" not in self._generators:
                self._init_base_generator()
            gpu_id = mattergen_config.MODEL_TO_GPU.get("mattergen_base", "0")
            return self._generators.get("base"), "base", None, gpu_id

        # Handle property constraints
        properties_to_condition_on = dict(properties)

        # Determine model directory
        if len(properties) == 1:
            # Single property condition
            property_name = list(properties.keys())[0]
            property_to_model = {
                "dft_mag_density": "dft_mag_density",
                "space_group": "space_group",
                "chemical_system": "chemical_system",
                "dft_band_gap": "dft_band_gap"
            }
            model_dir = property_to_model.get(property_name, property_name)
            generator_key = f"single_{property_name}"
        else:
            # Multi-property condition
            property_keys = set(properties.keys())
            if property_keys == {"dft_mag_density", "hhi_score"}:
                model_dir = "dft_mag_density_hhi_score"
                generator_key = "multi_dft_mag_density_hhi_score"
            elif property_keys == {"chemical_system", "energy_above_hull"}:
                model_dir = "chemical_system_energy_above_hull"
                generator_key = "multi_chemical_system_energy_above_hull"
            else:
                # If no specific multi-property model, use first property's model
                first_property = list(properties.keys())[0]
                property_to_model = {
                    "dft_mag_density": "dft_mag_density",
                    "space_group": "space_group",
                    "chemical_system": "chemical_system",
                    "dft_band_gap": "dft_band_gap"
                }
                model_dir = property_to_model.get(first_property, first_property)
                generator_key = f"multi_{first_property}_etc"

        # Get GPU ID
        gpu_id = mattergen_config.MODEL_TO_GPU.get(model_dir, "0")

        # Build model path
        model_path = os.path.join(mattergen_config.MATTERGENMODEL_ROOT, model_dir)

        # Check if model exists, fallback to base if not
        if not os.path.exists(model_path):
            logger.warning(f"Model directory for {model_dir} not found. Using base model.")
            model_path = os.path.join(mattergen_config.MATTERGENMODEL_ROOT, "mattergen_base")
            generator_key = "base"
            # Clear property conditions when falling back to base model
            properties_to_condition_on = None

        # Check if generator already exists
        if generator_key in self._generators:
            generator = self._generators[generator_key]
            generator.batch_size = batch_size
            generator.num_batches = num_batches
            generator.diffusion_guidance_factor = diffusion_guidance_factor if properties_to_condition_on else 0.0
            generator.properties_to_condition_on = properties_to_condition_on
            return generator, generator_key, properties_to_condition_on, gpu_id

        # Create new generator
        try:
            checkpoint_info = MatterGenCheckpointInfo(
                model_path=Path(model_path).resolve(),
                load_epoch="last",
                config_overrides=[],
                strict_checkpoint_loading=True,
            )

            generator = CrystalGenerator(
                checkpoint_info=checkpoint_info,
                properties_to_condition_on=properties_to_condition_on,
                batch_size=batch_size,
                num_batches=num_batches,
                sampling_config_name="default",
                sampling_config_path=None,
                sampling_config_overrides=[],
                record_trajectories=True,
                diffusion_guidance_factor=diffusion_guidance_factor if properties_to_condition_on else 0.0,
                target_compositions_dict=[],
            )

            self._generators[generator_key] = generator
            logger.info(f"MatterGen generator for {generator_key} initialized successfully")
            return generator, generator_key, properties_to_condition_on, gpu_id
        except Exception as e:
            logger.error(f"Failed to initialize MatterGen generator for {generator_key}: {e}")
            # Fallback to base generator
            if "base" not in self._generators:
                self._init_base_generator()
            base_gpu_id = mattergen_config.MODEL_TO_GPU.get("mattergen_base", "0")
            return self._generators.get("base"), "base", None, base_gpu_id

    def generate(self, properties=None, batch_size=2, num_batches=1, diffusion_guidance_factor=2.0):
        """
        Generate crystal structures with optional property constraints.

        Returns:
            str: Descriptive text with generated crystal structures in CIF format
        """
        # Handle string input
        if isinstance(properties, str):
            try:
                properties = json.loads(properties)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid properties JSON string: {properties}")

        properties = properties or {}

        # Get generator and GPU ID
        generator, generator_key, properties_to_condition_on, gpu_id = self._get_or_create_generator(
            properties, batch_size, num_batches, diffusion_guidance_factor
        )

        if generator is None:
            return "Error: Failed to initialize MatterGen generator"

        # Set GPU device
        try:
            cuda_device_id = int(gpu_id)
            torch.cuda.set_device(cuda_device_id)
            logger.info(f"Using GPU {cuda_device_id} for model {generator_key}")
        except Exception as e:
            logger.warning(f"Error setting CUDA device: {e}")

        # Generate structures
        try:
            output_dir = Path(self._output_dir) / datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            output_dir.mkdir(parents=True, exist_ok=True)
            generator.generate(output_dir=output_dir)
        except Exception as e:
            logger.error(f"Error generating structures: {e}")
            return f"Error generating structures: {e}"

        # Read CIF file from ZIP
        cif_zip_path = output_dir / "generated_crystals_cif.zip"
        cif_content = ""

        if cif_zip_path.exists():
            try:
                with zipfile.ZipFile(cif_zip_path, 'r') as zip_file:
                    # Get all CIF files in the ZIP
                    cif_files = [f for f in zip_file.namelist() if f.endswith('.cif')]

                    # Read and combine all CIF files
                    cif_contents = []
                    for cif_file in cif_files:
                        with zip_file.open(cif_file) as f:
                            content = f.read().decode('utf-8', errors='replace')
                            cif_contents.append(content)

                    # Join all CIF contents
                    cif_content = '\n'.join(cif_contents)
            except Exception as e:
                logger.error(f"Error reading CIF ZIP file: {e}")
                # Fallback to reading as binary if ZIP extraction fails
                with open(cif_zip_path, 'rb') as f:
                    cif_content = f.read().decode('utf-8', errors='replace')

        # Create description
        if not properties:
            title = "Generated Material Structures"
            description = "These structures were generated unconditionally."
            property_description = "unconditionally"
        elif len(properties) == 1:
            property_name = list(properties.keys())[0]
            property_value = properties[property_name]
            title = f"Generated Material Structures Conditioned on {property_name} = {property_value}"
            description = f"These structures were generated targeting {property_name} = {property_value}."
            property_description = f"conditioned on {property_name} = {property_value}"
        else:
            title = "Generated Material Structures Conditioned on Multiple Properties"
            description = "These structures were generated with multi-property conditioning."
            property_description = f"conditioned on: {', '.join([f'{name} = {value}' for name, value in properties.items()])}"

        # Format result
        formatted_cif = format_cif_content(cif_content)

        result = f"""
# {title}

This data contains {batch_size * num_batches} crystal structures generated by MatterGen, {property_description}.

{f'Diffusion guidance factor: {diffusion_guidance_factor}' if properties else ''}

## CIF Files (Crystallographic Information Files)

```
{formatted_cif}
```

{description}
"""

        return result
