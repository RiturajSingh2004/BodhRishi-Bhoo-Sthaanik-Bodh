# core/llm_engine.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import re
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialLLMEngine:
    """
    Core LLM Engine for spatial analysis workflow generation using Chain-of-Thought reasoning
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the LLM engine with specified model
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Initialize model components
        self._load_model()
        
        # GIS operation templates
        self.gis_operations = {
            "buffer_analysis": {
                "description": "Creates buffer zones around geometric features",
                "parameters": ["distance", "units", "segments"],
                "tools": ["geopandas", "qgis", "gdal"],
                "input_types": ["vector"],
                "output_types": ["vector"]
            },
            "spatial_join": {
                "description": "Joins attributes based on spatial relationship",
                "parameters": ["join_type", "spatial_predicate"],
                "tools": ["geopandas", "qgis"],
                "input_types": ["vector", "vector"],
                "output_types": ["vector"]
            },
            "clip": {
                "description": "Clips input features to boundary extent",
                "parameters": ["clip_layer"],
                "tools": ["geopandas", "qgis", "gdal"],
                "input_types": ["vector", "vector"],
                "output_types": ["vector"]
            },
            "raster_calculator": {
                "description": "Performs mathematical operations on raster data",
                "parameters": ["expression", "output_type"],
                "tools": ["gdal", "rasterio", "qgis"],
                "input_types": ["raster"],
                "output_types": ["raster"]
            },
            "overlay_analysis": {
                "description": "Overlays multiple spatial layers",
                "parameters": ["overlay_type", "keep_geom_type"],
                "tools": ["geopandas", "qgis"],
                "input_types": ["vector", "vector"],
                "output_types": ["vector"]
            },
            "slope_analysis": {
                "description": "Calculates slope from elevation data",
                "parameters": ["slope_type", "units"],
                "tools": ["gdal", "grass", "qgis"],
                "input_types": ["raster"],
                "output_types": ["raster"]
            },
            "proximity_analysis": {
                "description": "Calculates distance to nearest features",
                "parameters": ["max_distance", "units"],
                "tools": ["gdal", "qgis", "grass"],
                "input_types": ["vector", "raster"],
                "output_types": ["raster"]
            },
            "classification": {
                "description": "Classifies data based on criteria",
                "parameters": ["classification_field", "classes"],
                "tools": ["qgis", "grass", "rasterio"],
                "input_types": ["vector", "raster"],
                "output_types": ["vector", "raster"]
            }
        }
        
    def _load_model(self):
        """Load the LLM model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                load_in_8bit=True if self.device == "cuda" else False,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_spatial_workflow(self, query: str, context: str = "", 
                                 max_retries: int = 3) -> Optional[Dict]:
        """
        Generate spatial analysis workflow using Chain-of-Thought reasoning
        
        Args:
            query: Natural language spatial analysis query
            context: Additional context from RAG system
            max_retries: Maximum retry attempts for valid JSON generation
            
        Returns:
            Dictionary containing workflow specification
        """
        
        # Construct Chain-of-Thought prompt
        cot_prompt = self._construct_cot_prompt(query, context)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating workflow (attempt {attempt + 1})")
                
                # Generate response using LLM
                response = self.pipeline(
                    cot_prompt,
                    max_new_tokens=2048,
                    temperature=0.7 + (0.1 * attempt),  # Increase temperature on retries
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract generated text
                generated_text = response[0]['generated_text']
                
                # Parse workflow from response
                workflow = self._parse_workflow_response(generated_text)
                
                if workflow and self._validate_workflow(workflow):
                    logger.info("Successfully generated valid workflow")
                    return workflow
                else:
                    logger.warning(f"Invalid workflow generated on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.error(f"Error generating workflow on attempt {attempt + 1}: {e}")
                
        logger.error("Failed to generate valid workflow after all attempts")
        return None
    
    def _construct_cot_prompt(self, query: str, context: str) -> str:
        """
        Construct Chain-of-Thought prompt for workflow generation
        
        Args:
            query: User query
            context: RAG context
            
        Returns:
            Formatted prompt string
        """
        
        available_operations = "\n".join([
            f"- {op}: {details['description']}"
            for op, details in self.gis_operations.items()
        ])
        
        prompt = f"""You are an expert GIS analyst with deep knowledge of spatial analysis workflows. 
Your task is to analyze a spatial problem and generate a step-by-step workflow using available GIS operations.

Available GIS Operations:
{available_operations}

Context Information:
{context}

User Query: {query}

CHAIN-OF-THOUGHT REASONING:
Think through this step by step:

1. PROBLEM ANALYSIS:
   - What is the core spatial problem?
   - What type of analysis is needed?
   - What are the key spatial relationships to examine?

2. DATA REQUIREMENTS:
   - What spatial datasets are needed?
   - What data sources should be used (OSM, Bhoonidhi, etc.)?
   - What coordinate systems are appropriate?

3. WORKFLOW DESIGN:
   - What sequence of operations will solve this problem?
   - How do outputs of one step feed into the next?
   - What parameters are needed for each operation?

4. VALIDATION STRATEGY:
   - How can we verify the results are accurate?
   - What quality checks should be performed?

Now, provide your reasoning and generate the workflow:

REASONING:
[Provide your step-by-step thinking process here]

WORKFLOW JSON:
```json
{{
    "reasoning": "Your detailed reasoning process",
    "problem_type": "flood_risk|site_suitability|land_cover|proximity_analysis|terrain_analysis|other",
    "data_sources": [
        {{
            "name": "data_source_name",
            "type": "vector|raster|api",
            "source": "osm|bhoonidhi|local|other",
            "description": "description of the data"
        }}
    ],
    "workflow_steps": [
        {{
            "step": 1,
            "operation": "operation_name",
            "tool": "geopandas|qgis|gdal|rasterio|grass",
            "description": "what this step accomplishes",
            "parameters": {{
                "param1": "value1",
                "param2": "value2"
            }},
            "inputs": ["input_file1", "input_file2"],
            "outputs": ["output_file1"],
            "reasoning": "why this step is necessary"
        }}
    ],
    "expected_outputs": [
        {{
            "name": "output_name",
            "type": "vector|raster|report",
            "description": "description of the output",
            "format": "shapefile|geojson|geotiff|pdf"
        }}
    ],
    "validation_criteria": [
        "criterion1: description",
        "criterion2: description"
    ],
    "estimated_runtime": "time_estimate",
    "complexity": "low|medium|high"
}}
```

Remember to:
- Use only the available GIS operations
- Ensure logical flow between steps
- Consider coordinate reference systems
- Include proper error handling
- Make the workflow executable
"""
        
        return prompt
    
    def _parse_workflow_response(self, response: str) -> Optional[Dict]:
        """
        Parse workflow JSON from LLM response
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed workflow dictionary or None
        """
        try:
            # Find JSON content between ```json markers
            json_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON by looking for { and }
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                else:
                    logger.error("No JSON found in response")
                    return None
            
            # Parse JSON
            workflow = json.loads(json_str)
            
            # Clean up any potential formatting issues
            workflow = self._clean_workflow_json(workflow)
            
            return workflow
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing workflow response: {e}")
            return None
    
    def _clean_workflow_json(self, workflow: Dict) -> Dict:
        """
        Clean and validate workflow JSON structure
        
        Args:
            workflow: Raw workflow dictionary
            
        Returns:
            Cleaned workflow dictionary
        """
        # Ensure required fields exist
        required_fields = {
            "reasoning": "",
            "problem_type": "other",
            "data_sources": [],
            "workflow_steps": [],
            "expected_outputs": [],
            "validation_criteria": [],
            "estimated_runtime": "unknown",
            "complexity": "medium"
        }
        
        for field, default_value in required_fields.items():
            if field not in workflow:
                workflow[field] = default_value
        
        # Validate workflow steps
        for i, step in enumerate(workflow.get("workflow_steps", [])):
            if "step" not in step:
                step["step"] = i + 1
            if "operation" not in step:
                step["operation"] = "unknown"
            if "tool" not in step:
                step["tool"] = "qgis"
            if "parameters" not in step:
                step["parameters"] = {}
            if "inputs" not in step:
                step["inputs"] = []
            if "outputs" not in step:
                step["outputs"] = []
            if "reasoning" not in step:
                step["reasoning"] = "Processing step"
        
        return workflow
    
    def _validate_workflow(self, workflow: Dict) -> bool:
        """
        Validate workflow structure and content
        
        Args:
            workflow: Workflow dictionary to validate
            
        Returns:
            True if workflow is valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ["reasoning", "problem_type", "workflow_steps"]
            for field in required_fields:
                if field not in workflow:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate workflow steps
            steps = workflow.get("workflow_steps", [])
            if not steps:
                logger.error("No workflow steps found")
                return False
            
            for i, step in enumerate(steps):
                # Check required step fields
                step_fields = ["step", "operation", "tool"]
                for field in step_fields:
                    if field not in step:
                        logger.error(f"Missing field '{field}' in step {i + 1}")
                        return False
                
                # Check if operation is valid
                operation = step.get("operation")
                if operation not in self.gis_operations and operation != "load_data":
                    logger.warning(f"Unknown operation: {operation}")
                    # Don't fail validation for unknown operations, just warn
            
            # Check logical flow (outputs of one step should be inputs of next)
            for i in range(len(steps) - 1):
                current_outputs = steps[i].get("outputs", [])
                next_inputs = steps[i + 1].get("inputs", [])
                
                # At least one output should be used as input in subsequent steps
                if current_outputs and not any(
                    output in " ".join(next_inputs) for output in current_outputs
                ):
                    logger.warning(f"Potential workflow discontinuity between steps {i + 1} and {i + 2}")
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow validation error: {e}")
            return False
    
    def explain_workflow(self, workflow: Dict) -> str:
        """
        Generate human-readable explanation of the workflow
        
        Args:
            workflow: Workflow dictionary
            
        Returns:
            Formatted explanation string
        """
        try:
            explanation = f"""
# Spatial Analysis Workflow Explanation

## Problem Analysis
**Type:** {workflow.get('problem_type', 'Unknown')}
**Complexity:** {workflow.get('complexity', 'Medium')}
**Estimated Runtime:** {workflow.get('estimated_runtime', 'Unknown')}

## Chain-of-Thought Reasoning
{workflow.get('reasoning', 'No reasoning provided')}

## Data Sources Required
"""
            
            for i, source in enumerate(workflow.get('data_sources', []), 1):
                explanation += f"{i}. **{source.get('name', 'Unknown')}** ({source.get('type', 'Unknown')})\n"
                explanation += f"   - Source: {source.get('source', 'Unknown')}\n"
                explanation += f"   - Description: {source.get('description', 'No description')}\n\n"
            
            explanation += "## Workflow Steps\n"
            
            for step in workflow.get('workflow_steps', []):
                explanation += f"### Step {step.get('step', 'Unknown')}: {step.get('operation', 'Unknown')}\n"
                explanation += f"**Tool:** {step.get('tool', 'Unknown')}\n"
                explanation += f"**Description:** {step.get('description', 'No description')}\n"
                explanation += f"**Reasoning:** {step.get('reasoning', 'No reasoning provided')}\n"
                
                if step.get('parameters'):
                    explanation += "**Parameters:**\n"
                    for param, value in step['parameters'].items():
                        explanation += f"- {param}: {value}\n"
                
                explanation += f"**Inputs:** {', '.join(step.get('inputs', []))}\n"
                explanation += f"**Outputs:** {', '.join(step.get('outputs', []))}\n\n"
            
            explanation += "## Expected Outputs\n"
            for i, output in enumerate(workflow.get('expected_outputs', []), 1):
                explanation += f"{i}. **{output.get('name', 'Unknown')}** ({output.get('type', 'Unknown')})\n"
                explanation += f"   - Format: {output.get('format', 'Unknown')}\n"
                explanation += f"   - Description: {output.get('description', 'No description')}\n\n"
            
            explanation += "## Validation Criteria\n"
            for criterion in workflow.get('validation_criteria', []):
                explanation += f"- {criterion}\n"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating workflow explanation: {e}")
            return "Error generating explanation"
    
    def get_operation_details(self, operation_name: str) -> Dict:
        """
        Get detailed information about a specific GIS operation
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Operation details dictionary
        """
        return self.gis_operations.get(operation_name, {})
    
    def list_available_operations(self) -> List[str]:
        """
        Get list of available GIS operations
        
        Returns:
            List of operation names
        """
        return list(self.gis_operations.keys())
    
    def clear_memory(self):
        """Clear GPU memory to prevent OOM errors"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def __del__(self):
        """Cleanup resources"""
        self.clear_memory()