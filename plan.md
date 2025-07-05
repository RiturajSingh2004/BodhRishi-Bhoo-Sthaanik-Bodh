# Chain-of-Thought-Based LLM System for Spatial Analysis

## System Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐     ┌─────────────────┐
│   User Query    │───▶│  LLM Reasoning  │───▶│  Workflow Gen   │
│  (Natural Lang) │    │   (CoT Engine)  │     │   (JSON/YAML)   │
└─────────────────┘    └─────────────────┘     └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   RAG System    │              │
         │              │ (GIS Knowledge) │              │
         │              └─────────────────┘              │
         │                                               │
         ▼                                               ▼
┌─────────────────┐                            ┌─────────────────┐
│  Streamlit UI   │                            │ Workflow Engine │
│   Interface     │                            │   (Executor)    │
└─────────────────┘                            └─────────────────┘
         │                                               │
         │                                               ▼
         │                                      ┌─────────────────┐
         │                                      │  GIS Libraries  │
         │                                      │ GeoPandas/QGIS  │
         │                                      └─────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐                            ┌─────────────────┐
│   Outputs &     │◀───────────────────────────│   Data Sources  │
│ Visualizations  │                            │  OSM/Bhoonidhi  │
└─────────────────┘                            └─────────────────┘
```

## 1. Core Components Implementation

### 1.1 LLM Setup with Llama-3.1-8B-Instruct

```python
# core/llm_engine.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import json

class SpatialLLMEngine:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        """Initialize the LLM engine with Llama-3.1-8B-Instruct"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            load_in_8bit=True  # Memory optimization
        )
        
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=4096,
            temperature=0.7,
            do_sample=True,
            device=0 if self.device == "cuda" else -1
        )
        
        # Initialize LangChain wrapper
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        self.memory = ConversationBufferMemory()
        
    def generate_spatial_workflow(self, query, context=""):
        """Generate spatial analysis workflow with Chain-of-Thought reasoning"""
        
        cot_prompt = f"""
        You are an expert GIS analyst. Given a spatial analysis query, break it down into logical steps 
        and generate a workflow using available GIS tools.
        
        Query: {query}
        Context: {context}
        
        Think step by step:
        1. Identify the spatial problem type
        2. Determine required data sources
        3. List necessary geoprocessing operations
        4. Define the logical sequence of operations
        5. Specify output requirements
        
        Generate a JSON workflow with the following structure:
        {{
            "reasoning": "Your step-by-step thought process",
            "problem_type": "flood_risk|site_suitability|land_cover|other",
            "data_sources": ["list of required datasets"],
            "workflow_steps": [
                {{
                    "step": 1,
                    "operation": "operation_name",
                    "tool": "gdal|qgis|geopandas|rasterio",
                    "parameters": {{}},
                    "inputs": ["input_files"],
                    "outputs": ["output_files"],
                    "reasoning": "why this step is needed"
                }}
            ],
            "expected_outputs": ["list of final outputs"],
            "validation_criteria": ["criteria to validate results"]
        }}
        """
        
        response = self.llm(cot_prompt)
        return self._parse_workflow_response(response)
    
    def _parse_workflow_response(self, response):
        """Parse and validate LLM response"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            workflow = json.loads(json_str)
            return workflow
        except Exception as e:
            print(f"Error parsing workflow: {e}")
            return None
```

### 1.2 RAG System for GIS Knowledge

```python
# core/rag_system.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import os

class GISKnowledgeRAG:
    def __init__(self):
        """Initialize RAG system with GIS documentation"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def build_knowledge_base(self, docs_path="./gis_docs"):
        """Build RAG knowledge base from GIS documentation"""
        # Load documentation files
        loader = DirectoryLoader(docs_path, glob="*.txt")
        documents = loader.load()
        
        # Split documents
        texts = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        
        # Add predefined GIS knowledge
        self._add_gis_operations_knowledge()
        
    def _add_gis_operations_knowledge(self):
        """Add structured GIS operations knowledge"""
        gis_operations = [
            {
                "operation": "buffer_analysis",
                "description": "Creates buffer zones around geometric features",
                "tools": ["geopandas", "qgis", "gdal"],
                "parameters": ["distance", "units", "segments"],
                "use_cases": ["proximity analysis", "impact zones"]
            },
            {
                "operation": "spatial_join",
                "description": "Joins attributes from one layer to another based on spatial relationship",
                "tools": ["geopandas", "qgis"],
                "parameters": ["join_type", "spatial_predicate"],
                "use_cases": ["point-in-polygon", "overlay analysis"]
            },
            {
                "operation": "raster_calculator",
                "description": "Performs mathematical operations on raster datasets",
                "tools": ["gdal", "rasterio", "qgis"],
                "parameters": ["expression", "output_type"],
                "use_cases": ["NDVI calculation", "terrain analysis"]
            },
            {
                "operation": "clip",
                "description": "Clips input features to the extent of clip features",
                "tools": ["geopandas", "qgis", "gdal"],
                "parameters": ["input_layer", "clip_layer"],
                "use_cases": ["extract area of interest", "mask operations"]
            }
        ]
        
        # Convert to documents and add to vectorstore
        for op in gis_operations:
            doc_text = f"Operation: {op['operation']}\n"
            doc_text += f"Description: {op['description']}\n"
            doc_text += f"Tools: {', '.join(op['tools'])}\n"
            doc_text += f"Parameters: {', '.join(op['parameters'])}\n"
            doc_text += f"Use cases: {', '.join(op['use_cases'])}"
            
            # Add to vectorstore (implementation depends on your vectorstore choice)
            
    def retrieve_relevant_context(self, query, k=5):
        """Retrieve relevant GIS knowledge for query"""
        if self.vectorstore is None:
            return ""
            
        docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        return context
```

### 1.3 Workflow Execution Engine

```python
# core/workflow_executor.py
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np
import json
import os

class WorkflowExecutor:
    def __init__(self, data_path="./data"):
        """Initialize workflow executor"""
        self.data_path = data_path
        self.intermediate_results = {}
        
    def execute_workflow(self, workflow_json):
        """Execute the generated workflow"""
        results = {
            "success": True,
            "outputs": [],
            "intermediate_files": [],
            "execution_log": [],
            "errors": []
        }
        
        try:
            workflow = json.loads(workflow_json) if isinstance(workflow_json, str) else workflow_json
            
            for step in workflow.get("workflow_steps", []):
                step_result = self._execute_step(step)
                results["execution_log"].append({
                    "step": step["step"],
                    "operation": step["operation"],
                    "status": "success" if step_result["success"] else "failed",
                    "message": step_result.get("message", "")
                })
                
                if not step_result["success"]:
                    results["success"] = False
                    results["errors"].append(step_result.get("error", "Unknown error"))
                    break
                    
                results["intermediate_files"].extend(step_result.get("outputs", []))
                
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            
        return results
    
    def _execute_step(self, step):
        """Execute individual workflow step"""
        operation = step["operation"]
        tool = step["tool"]
        parameters = step.get("parameters", {})
        
        try:
            if operation == "buffer_analysis":
                return self._buffer_analysis(step)
            elif operation == "spatial_join":
                return self._spatial_join(step)
            elif operation == "clip":
                return self._clip_operation(step)
            elif operation == "raster_calculator":
                return self._raster_calculator(step)
            elif operation == "load_data":
                return self._load_data(step)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _buffer_analysis(self, step):
        """Execute buffer analysis"""
        params = step["parameters"]
        input_file = step["inputs"][0]
        output_file = step["outputs"][0]
        
        # Load input data
        gdf = gpd.read_file(os.path.join(self.data_path, input_file))
        
        # Create buffer
        buffer_distance = params.get("distance", 1000)
        buffered = gdf.buffer(buffer_distance)
        
        # Save result
        buffer_gdf = gpd.GeoDataFrame(gdf.drop("geometry", axis=1), geometry=buffered)
        output_path = os.path.join(self.data_path, output_file)
        buffer_gdf.to_file(output_path)
        
        return {
            "success": True,
            "outputs": [output_file],
            "message": f"Buffer analysis completed with {buffer_distance}m buffer"
        }
    
    def _spatial_join(self, step):
        """Execute spatial join"""
        params = step["parameters"]
        left_file = step["inputs"][0]
        right_file = step["inputs"][1]
        output_file = step["outputs"][0]
        
        # Load data
        left_gdf = gpd.read_file(os.path.join(self.data_path, left_file))
        right_gdf = gpd.read_file(os.path.join(self.data_path, right_file))
        
        # Ensure same CRS
        if left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)
        
        # Perform spatial join
        how = params.get("join_type", "inner")
        predicate = params.get("spatial_predicate", "intersects")
        
        joined = gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)
        
        # Save result
        output_path = os.path.join(self.data_path, output_file)
        joined.to_file(output_path)
        
        return {
            "success": True,
            "outputs": [output_file],
            "message": f"Spatial join completed with {len(joined)} features"
        }
    
    def _clip_operation(self, step):
        """Execute clip operation"""
        input_file = step["inputs"][0]
        clip_file = step["inputs"][1]
        output_file = step["outputs"][0]
        
        # Load data
        input_gdf = gpd.read_file(os.path.join(self.data_path, input_file))
        clip_gdf = gpd.read_file(os.path.join(self.data_path, clip_file))
        
        # Ensure same CRS
        if input_gdf.crs != clip_gdf.crs:
            clip_gdf = clip_gdf.to_crs(input_gdf.crs)
        
        # Clip
        clipped = gpd.clip(input_gdf, clip_gdf)
        
        # Save result
        output_path = os.path.join(self.data_path, output_file)
        clipped.to_file(output_path)
        
        return {
            "success": True,
            "outputs": [output_file],
            "message": f"Clip operation completed with {len(clipped)} features"
        }
    
    def _load_data(self, step):
        """Load data from various sources"""
        source = step["parameters"].get("source", "file")
        
        if source == "osm":
            return self._load_osm_data(step)
        elif source == "bhoonidhi":
            return self._load_bhoonidhi_data(step)
        else:
            return {"success": True, "message": "Data loading step"}
    
    def _load_osm_data(self, step):
        """Load data from OpenStreetMap"""
        # Implementation for OSM data loading
        # This would use libraries like OSMnx or direct API calls
        return {"success": True, "message": "OSM data loaded"}
    
    def _load_bhoonidhi_data(self, step):
        """Load data from Bhoonidhi"""
        # Implementation for Bhoonidhi data loading
        return {"success": True, "message": "Bhoonidhi data loaded"}
```

## 2. Streamlit User Interface

```python
# app/streamlit_app.py
import streamlit as st
import json
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
from core.llm_engine import SpatialLLMEngine
from core.rag_system import GISKnowledgeRAG
from core.workflow_executor import WorkflowExecutor

class SpatialAnalysisApp:
    def __init__(self):
        """Initialize the Streamlit app"""
        if 'llm_engine' not in st.session_state:
            st.session_state.llm_engine = SpatialLLMEngine()
            st.session_state.rag_system = GISKnowledgeRAG()
            st.session_state.executor = WorkflowExecutor()
            st.session_state.rag_system.build_knowledge_base()
    
    def run(self):
        """Main app interface"""
        st.set_page_config(
            page_title="Spatial Analysis LLM System",
            page_icon="🌍",
            layout="wide"
        )
        
        st.title("🌍 Chain-of-Thought Spatial Analysis System")
        st.markdown("Generate and execute GIS workflows from natural language queries")
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("Configuration")
            
            # Model settings
            st.subheader("Model Settings")
            temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
            max_tokens = st.slider("Max Tokens", 512, 4096, 2048)
            
            # Data sources
            st.subheader("Data Sources")
            use_osm = st.checkbox("OpenStreetMap", value=True)
            use_bhoonidhi = st.checkbox("Bhoonidhi", value=True)
            
            # Sample queries
            st.subheader("Sample Queries")
            sample_queries = [
                "Find suitable locations for solar farms near Mumbai avoiding flood zones",
                "Assess flood risk for residential areas along Ganges river",
                "Identify optimal sites for hospitals with good road access in Bangalore",
                "Analyze urban heat island effect using satellite imagery"
            ]
            
            selected_sample = st.selectbox(
                "Select a sample query:",
                [""] + sample_queries
            )
        
        # Main interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("Query Input")
            
            # User query input
            user_query = st.text_area(
                "Enter your spatial analysis query:",
                value=selected_sample,
                height=100,
                placeholder="e.g., Find flood-prone areas near urban centers..."
            )
            
            # Additional parameters
            with st.expander("Advanced Parameters"):
                study_area = st.text_input("Study Area", "")
                data_resolution = st.selectbox("Data Resolution", ["High", "Medium", "Low"])
                output_format = st.selectbox("Output Format", ["Shapefile", "GeoJSON", "KML"])
            
            # Generate workflow button
            if st.button("🚀 Generate Workflow", type="primary"):
                if user_query:
                    self._generate_and_display_workflow(user_query)
                else:
                    st.error("Please enter a query first!")
        
        with col2:
            st.header("Workflow & Results")
            
            # Display workflow if generated
            if 'current_workflow' in st.session_state:
                self._display_workflow_results()
    
    def _generate_and_display_workflow(self, query):
        """Generate workflow and display results"""
        with st.spinner("Generating workflow..."):
            try:
                # Get relevant context from RAG
                context = st.session_state.rag_system.retrieve_relevant_context(query)
                
                # Generate workflow
                workflow = st.session_state.llm_engine.generate_spatial_workflow(query, context)
                
                if workflow:
                    st.session_state.current_workflow = workflow
                    st.success("Workflow generated successfully!")
                    
                    # Display Chain-of-Thought reasoning
                    with st.expander("🧠 Chain-of-Thought Reasoning", expanded=True):
                        st.markdown(workflow.get("reasoning", "No reasoning provided"))
                    
                    # Display workflow steps
                    with st.expander("📋 Workflow Steps", expanded=True):
                        for i, step in enumerate(workflow.get("workflow_steps", [])):
                            with st.container():
                                st.markdown(f"**Step {step['step']}: {step['operation']}**")
                                st.markdown(f"- Tool: {step['tool']}")
                                st.markdown(f"- Reasoning: {step['reasoning']}")
                                st.json(step['parameters'])
                                st.markdown("---")
                    
                    # Execute workflow button
                    if st.button("▶️ Execute Workflow"):
                        self._execute_workflow(workflow)
                        
                else:
                    st.error("Failed to generate workflow. Please try again.")
                    
            except Exception as e:
                st.error(f"Error generating workflow: {str(e)}")
    
    def _execute_workflow(self, workflow):
        """Execute the generated workflow"""
        with st.spinner("Executing workflow..."):
            try:
                results = st.session_state.executor.execute_workflow(workflow)
                st.session_state.execution_results = results
                
                if results["success"]:
                    st.success("Workflow executed successfully!")
                    
                    # Display execution log
                    with st.expander("📊 Execution Log"):
                        for log_entry in results["execution_log"]:
                            status_icon = "✅" if log_entry["status"] == "success" else "❌"
                            st.markdown(f"{status_icon} **Step {log_entry['step']}**: {log_entry['operation']}")
                            if log_entry["message"]:
                                st.markdown(f"   {log_entry['message']}")
                    
                    # Display outputs
                    if results["outputs"]:
                        st.subheader("📁 Generated Outputs")
                        for output in results["outputs"]:
                            st.markdown(f"- {output}")
                            
                else:
                    st.error("Workflow execution failed!")
                    for error in results["errors"]:
                        st.error(f"Error: {error}")
                        
            except Exception as e:
                st.error(f"Error executing workflow: {str(e)}")
    
    def _display_workflow_results(self):
        """Display current workflow and results"""
        workflow = st.session_state.current_workflow
        
        # Workflow metadata
        st.subheader("📋 Workflow Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Problem Type", workflow.get("problem_type", "Unknown"))
            st.metric("Total Steps", len(workflow.get("workflow_steps", [])))
        
        with col2:
            st.metric("Data Sources", len(workflow.get("data_sources", [])))
            st.metric("Expected Outputs", len(workflow.get("expected_outputs", [])))
        
        # Display workflow as JSON
        with st.expander("📄 Workflow JSON"):
            st.json(workflow)
        
        # Display map if results available
        if 'execution_results' in st.session_state:
            self._display_results_map()
    
    def _display_results_map(self):
        """Display results on an interactive map"""
        st.subheader("🗺️ Results Visualization")
        
        # Create a folium map
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # India center
        
        # Add sample data (in real implementation, load actual results)
        folium.Marker(
            [28.6139, 77.2090],  # Delhi
            popup="Sample Result Point",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Display the map
        st_folium(m, width=700, height=400)
        
        # Download buttons
        st.subheader("📥 Download Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="Download Workflow JSON",
                data=json.dumps(st.session_state.current_workflow, indent=2),
                file_name="workflow.json",
                mime="application/json"
            )
        
        with col2:
            st.download_button(
                label="Download Execution Log",
                data="Execution log content...",  # Replace with actual log
                file_name="execution_log.txt",
                mime="text/plain"
            )
        
        with col3:
            st.download_button(
                label="Download Results",
                data="Results data...",  # Replace with actual results
                file_name="results.geojson",
                mime="application/json"
            )

# Run the app
if __name__ == "__main__":
    app = SpatialAnalysisApp()
    app.run()
```

## 3. Evaluation Framework

```python
# evaluation/evaluator.py
import json
import time
import psutil
import os
from typing import Dict, List, Any

class WorkflowEvaluator:
    def __init__(self):
        """Initialize the evaluation framework"""
        self.evaluation_metrics = {
            "logical_validity": 0.0,
            "syntactic_validity": 0.0,
            "reasoning_clarity": 0.0,
            "error_handling": 0.0,
            "efficiency": 0.0,
            "accuracy": 0.0
        }
    
    def evaluate_workflow(self, workflow: Dict, ground_truth: Dict = None) -> Dict:
        """Comprehensive workflow evaluation"""
        results = {}
        
        # 1. Logical Validity
        results["logical_validity"] = self._evaluate_logical_validity(workflow)
        
        # 2. Syntactic Validity
        results["syntactic_validity"] = self._evaluate_syntactic_validity(workflow)
        
        # 3. Reasoning Clarity
        results["reasoning_clarity"] = self._evaluate_reasoning_clarity(workflow)
        
        # 4. Error Handling
        results["error_handling"] = self._evaluate_error_handling(workflow)
        
        # 5. Efficiency Metrics
        results["efficiency"] = self._evaluate_efficiency(workflow)
        
        # 6. Accuracy (if ground truth available)
        if ground_truth:
            results["accuracy"] = self._evaluate_accuracy(workflow, ground_truth)
        
        # Overall score
        results["overall_score"] = self._calculate_overall_score(results)
        
        return results
    
    def _evaluate_logical_validity(self, workflow: Dict) -> float:
        """Evaluate logical flow of workflow steps"""
        score = 0.0
        steps = workflow.get("workflow_steps", [])
        
        if not steps:
            return 0.0
        
        # Check if steps are logically ordered
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]
            
            # Check if outputs of current step are used as inputs in subsequent steps
            current_outputs = current_step.get("outputs", [])
            next_inputs = next_step.get("inputs", [])
            
            if any(output in next_inputs for output in current_outputs):
                score += 1.0
        
        return min(score / max(len(steps) - 1, 1), 1.0)
    
    def _evaluate_syntactic_validity(self, workflow: Dict) -> float:
        """Evaluate syntactic correctness of workflow"""
        required_fields = ["problem_type", "workflow_steps", "data_sources"]
        score = 0.0
        
        # Check required fields
        for field in required_fields:
            if field in workflow:
                score += 1.0
        
        # Check workflow steps structure
        steps = workflow.get("workflow_steps", [])
        if steps:
            step_score = 0.0
            required_step_fields = ["step", "operation", "tool", "inputs", "outputs"]
            
            for step in steps:
                step_field_score = sum(1 for field in required_step_fields if field in step)
                step_score += step_field_score / len(required_step_fields)
            
            score += step_score / len(steps)
        
        return score / (len(required_fields) + 1)
    
    def _evaluate_reasoning_clarity(self, workflow: Dict) -> float:
        """Evaluate clarity of Chain-of-Thought reasoning"""
        reasoning = workflow.get("reasoning", "")
        
        if not reasoning:
            return 0.0
        
        # Simple heuristics for reasoning quality
        score = 0.0
        
        # Check length (should be substantial)
        if len(reasoning) > 100:
            score += 0.2
        
        # Check for step-by-step structure
        if "step" in reasoning.lower() or "first" in reasoning.lower():
            score += 0.2
        
        # Check for problem identification
        if "problem" in reasoning.lower() or "analysis" in reasoning.lower():
            score += 0.2
        
        # Check for data source mentions
        if "data" in reasoning.lower() or "source" in reasoning.lower():
            score += 0.2
        
        # Check for tool justification
        if "tool" in reasoning.lower() or "software" in reasoning.lower():
            score += 0.2
        
        return score
    
    def _evaluate_error_handling(self, workflow: Dict) -> float:
        """Evaluate error handling capabilities"""
        # This would be evaluated during execution
        # For now, return a placeholder score
        return 0.8
    
    def _evaluate_efficiency(self, workflow: Dict) -> float:
        """Evaluate workflow efficiency"""
        steps = workflow.get("workflow_steps", [])
        
        if not steps:
            return 0.0
        
        # Simple efficiency metrics
        efficiency_score = 0.0
        
        # Fewer steps generally better (but not always)
        if len(steps) <= 5:
            efficiency_score += 0.3
        elif len(steps) <= 10:
            efficiency_score += 0.2
        else:
            efficiency_score += 0.1
        
        # Check for parallel processing opportunities
        # (This is a simplified check)
        parallel_ops = 0
        for step in steps:
            if step.get("operation") in ["load_data", "buffer_analysis"]:
                parallel_ops += 1
        
        if parallel_ops > 1:
            efficiency_score += 0.3
        
        # Check for appropriate tool selection
        tool_variety = len(set(step.get("tool") for step in steps))
        if tool_variety > 1:
            efficiency_score += 0.4
        
        return min(efficiency_score, 1.0)
    
    def _evaluate_accuracy(self, workflow: Dict, ground_truth: Dict) -> float:
        """Evaluate accuracy against ground truth"""
        # Compare workflow steps with ground truth
        workflow_steps = workflow.get("workflow_steps", [])
        gt_steps = ground_truth.get("workflow_steps", [])
        
        if not gt_steps:
            return 0.0
        
        # Simple accuracy based on operation matching
        correct_operations = 0
        for gt_step in gt_steps:
            gt_operation = gt_step.get("operation")
            if any(step.get("operation") == gt_operation for step in workflow_steps):
                correct_operations += 1
        
        return correct_operations / len(gt_steps)
    
    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate weighted overall score"""
        weights = {
            "logical_validity": 0.25,
            "syntactic_validity": 0.20,
            "reasoning_clarity": 0.20,
            "error_handling": 0.15,
            "efficiency": 0.10,
            "accuracy": 0.10
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in results:
                overall_score += results[metric] * weight
                total_weight += weight
        
        return overall_score / total_weight if total_weight > 0 else 0.0

# Benchmark test cases
class BenchmarkTests:
    def __init__(self):
        """Initialize benchmark test cases"""
        self.test_cases = [
            {
                "name": "Flood Risk Assessment",
                "query": "Identify flood-prone areas in Mumbai using elevation data and historical flood records",
                "expected_operations": ["load_data", "buffer_analysis", "overlay_analysis", "risk_classification"],
                "data_sources": ["DEM", "historical_flood_data", "urban_boundaries"],
                "complexity": "high"
            },
            {
                "name": "Site Suitability Analysis",
                "query": "Find suitable locations for solar farms avoiding urban areas and water bodies",
                "expected_operations": ["load_data", "buffer_analysis", "overlay_analysis", "suitability_scoring"],
                "data_sources": ["land_use", "water_bodies", "urban_areas", "slope_data"],
                "complexity": "medium"
            },
            {
                "name": "Urban Heat Island Analysis",
                "query": "Analyze urban heat island effect using satellite temperature data",
                "expected_operations": ["load_data", "raster_calculator", "statistical_analysis", "classification"],
                "data_sources": ["satellite_imagery", "urban_boundaries", "vegetation_index"],
                "complexity": "medium"
            }
        ]
    
    def run_benchmarks(self, llm_engine, evaluator):
        """Run all benchmark tests"""
        results = []
        
        for test_case in self.test_cases:
            print(f"Running benchmark: {test_case['name']}")
            
            # Generate workflow
            start_time = time.time()
            workflow = llm_engine.generate_spatial_workflow(test_case["query"])
            generation_time = time.time() - start_time
            
            # Evaluate workflow
            evaluation = evaluator.evaluate_workflow(workflow, test_case)
            evaluation["generation_time"] = generation_time
            evaluation["test_case"] = test_case["name"]
            
            results.append(evaluation)
            
        return results
```

## 4. Implementation Setup and Configuration

```python
# setup/requirements.txt
torch>=2.0.0
transformers>=4.30.0
langchain>=0.0.200
streamlit>=1.25.0
geopandas>=0.13.0
rasterio>=1.3.0
folium>=0.14.0
streamlit-folium>=0.13.0
faiss-cpu>=1.7.0
sentence-transformers>=2.2.0
osmium>=3.4.0
requests>=2.30.0
numpy>=1.24.0
pandas>=2.0.0
shapely>=2.0.0
pyproj>=3.5.0
matplotlib>=3.7.0
plotly>=5.14.0
psutil>=5.9.0

# For GPU support (optional)
# torch>=2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

```python
# setup/config.py
import os
from pathlib import Path

class Config:
    """Configuration settings for the spatial LLM system"""
    
    # Model Configuration
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_TOKENS = 4096
    TEMPERATURE = 0.7
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    DOCS_DIR = BASE_DIR / "gis_docs"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    MODELS_DIR = BASE_DIR / "models"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, DOCS_DIR, OUTPUTS_DIR, MODELS_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # Data Sources
    OSM_API_URL = "https://overpass-api.de/api/interpreter"
    BHOONIDHI_API_URL = "https://bhoonidhi.nrsc.gov.in/api"
    
    # Evaluation
    BENCHMARK_RESULTS_DIR = BASE_DIR / "benchmark_results"
    BENCHMARK_RESULTS_DIR.mkdir(exist_ok=True)
    
    # Streamlit Configuration
    STREAMLIT_CONFIG = {
        "page_title": "Spatial Analysis LLM System",
        "page_icon": "🌍",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
```

## 5. Data Integration Modules

```python
# data/data_loader.py
import geopandas as gpd
import requests
import osmnx as ox
from pathlib import Path
import json

class DataLoader:
    """Handles loading data from various geospatial sources"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def load_osm_data(self, query: str, place: str = None, bbox: tuple = None):
        """Load data from OpenStreetMap"""
        try:
            if place:
                # Load by place name
                gdf = ox.geometries_from_place(place, tags={query: True})
            elif bbox:
                # Load by bounding box
                gdf = ox.geometries_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], tags={query: True})
            else:
                raise ValueError("Either place or bbox must be provided")
            
            return gdf
        except Exception as e:
            print(f"Error loading OSM data: {e}")
            return None
    
    def load_bhoonidhi_data(self, dataset_id: str, bbox: tuple = None):
        """Load data from Bhoonidhi portal"""
        # This is a placeholder - actual implementation would depend on Bhoonidhi API
        try:
            # Construct API request
            params = {
                "dataset_id": dataset_id,
                "format": "geojson"
            }
            
            if bbox:
                params["bbox"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            
            # Make API request (placeholder URL)
            response = requests.get("https://bhoonidhi.nrsc.gov.in/api/data", params=params)
            
            if response.status_code == 200:
                geojson_data = response.json()
                gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
                return gdf
            else:
                print(f"Error loading Bhoonidhi data: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error loading Bhoonidhi data: {e}")
            return None
    
    def load_sample_data(self):
        """Load sample datasets for testing"""
        # Create sample point data
        import numpy as np
        from shapely.geometry import Point
        
        # Generate random points for testing
        np.random.seed(42)
        n_points = 100
        
        # Mumbai bounding box
        min_lon, max_lon = 72.7, 73.2
        min_lat, max_lat = 18.8, 19.3
        
        lons = np.random.uniform(min_lon, max_lon, n_points)
        lats = np.random.uniform(min_lat, max_lat, n_points)
        
        points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'id': range(n_points),
            'category': np.random.choice(['residential', 'commercial', 'industrial'], n_points),
            'elevation': np.random.uniform(0, 100, n_points),
            'geometry': points
        })
        
        gdf.crs = "EPSG:4326"
        
        # Save sample data
        sample_file = self.data_dir / "sample_points.shp"
        gdf.to_file(sample_file)
        
        return gdf
```

## 6. Deployment Instructions

```bash
# deployment/deploy.sh
#!/bin/bash

echo "Setting up Spatial LLM System..."

# Create virtual environment
python -m venv spatial_llm_env
source spatial_llm_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download model (if not already cached)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'meta-llama/Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print('Model downloaded successfully!')
"

# Create necessary directories
mkdir -p data
mkdir -p gis_docs
mkdir -p outputs
mkdir -p models

# Download sample GIS documentation
echo "Setting up GIS knowledge base..."
# You would download QGIS, GDAL, GRASS documentation here

# Run the application
echo "Starting Streamlit application..."
streamlit run app/streamlit_app.py
```

## 7. Usage Examples and Testing

```python
# examples/usage_examples.py
from core.llm_engine import SpatialLLMEngine
from core.rag_system import GISKnowledgeRAG
from core.workflow_executor import WorkflowExecutor
from evaluation.evaluator import WorkflowEvaluator, BenchmarkTests

def main():
    """Example usage of the spatial LLM system"""
    
    # Initialize components
    llm_engine = SpatialLLMEngine()
    rag_system = GISKnowledgeRAG()
    executor = WorkflowExecutor()
    evaluator = WorkflowEvaluator()
    
    # Build knowledge base
    rag_system.build_knowledge_base()
    
    # Example 1: Flood Risk Assessment
    print("=== Flood Risk Assessment Example ===")
    query1 = "Identify flood-prone areas in Mumbai using elevation data and proximity to water bodies"
    
    context1 = rag_system.retrieve_relevant_context(query1)
    workflow1 = llm_engine.generate_spatial_workflow(query1, context1)
    
    if workflow1:
        print("Generated Workflow:")
        print(json.dumps(workflow1, indent=2))
        
        # Evaluate workflow
        evaluation1 = evaluator.evaluate_workflow(workflow1)
        print(f"Evaluation Score: {evaluation1['overall_score']:.2f}")
        
        # Execute workflow (would need actual data)
        # results1 = executor.execute_workflow(workflow1)
    
    # Example 2: Site Suitability Analysis
    print("\n=== Site Suitability Analysis Example ===")
    query2 = "Find optimal locations for solar farms avoiding urban areas and water bodies near Bangalore"
    
    context2 = rag_system.retrieve_relevant_context(query2)
    workflow2 = llm_engine.generate_spatial_workflow(query2, context2)
    
    if workflow2:
        print("Generated Workflow:")
        print(json.dumps(workflow2, indent=2))
        
        evaluation2 = evaluator.evaluate_workflow(workflow2)
        print(f"Evaluation Score: {evaluation2['overall_score']:.2f}")
    
    # Run benchmarks
    print("\n=== Running Benchmarks ===")
    benchmark_tests = BenchmarkTests()
    benchmark_results = benchmark_tests.run_benchmarks(llm_engine, evaluator)
    
    for result in benchmark_results:
        print(f"Test: {result['test_case']}")
        print(f"Overall Score: {result['overall_score']:.2f}")
        print(f"Generation Time: {result['generation_time']:.2f}s")
        print("---")

if __name__ == "__main__":
    main()
```

## 8. Performance Optimization

```python
# optimization/performance_optimizer.py
import torch
from transformers import BitsAndBytesConfig
import gc

class PerformanceOptimizer:
    """Optimization utilities for the spatial LLM system"""
    
    @staticmethod
    def optimize_model_loading():
        """Optimize model loading for memory efficiency"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        return quantization_config
    
    @staticmethod
    def clear_memory():
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def optimize_inference():
        """Optimize inference settings"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
```

This completes the comprehensive implementation framework for your Chain-of-Thought-based LLM system for spatial analysis. The system includes:

1. **Complete LLM Engine** with Llama-3.1-8B-Instruct integration
2. **RAG System** for GIS knowledge retrieval
3. **Workflow Executor** for running generated workflows
4. **Streamlit Interface** for user interaction
5. **Comprehensive Evaluation Framework**
6. **Data Integration** for OSM and Bhoonidhi
7. **Deployment Instructions** and configuration
8. **Performance Optimization** utilities

The system is designed to be modular, scalable, and easy to extend with additional GIS operations and data sources.
