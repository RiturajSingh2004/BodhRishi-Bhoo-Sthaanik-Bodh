# core/rag_system.py
import os
import json
import pickle
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

# Vector store and embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Text processing
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GISKnowledgeRAG:
    """
    Retrieval-Augmented Generation system for GIS knowledge
    Stores and retrieves relevant GIS documentation, operation details, and best practices
    """
    
    def __init__(self, 
                 docs_path: str = "./gis_docs",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_store_path: str = "./vector_store.pkl"):
        """
        Initialize RAG system
        
        Args:
            docs_path: Path to GIS documentation directory
            embedding_model: Sentence transformer model name
            vector_store_path: Path to save/load vector store
        """
        self.docs_path = Path(docs_path)
        self.vector_store_path = Path(vector_store_path)
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Knowledge base storage
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        # GIS-specific knowledge
        self.gis_operations_db = {}
        self.workflow_templates = {}
        self.best_practices = {}
        
        # Create docs directory if it doesn't exist
        self.docs_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize knowledge base
        self._initialize_gis_knowledge()
    
    def _initialize_gis_knowledge(self):
        """Initialize built-in GIS knowledge base"""
        
        # GIS Operations Database
        self.gis_operations_db = {
            "buffer_analysis": {
                "description": "Creates buffer zones around geometric features at specified distances",
                "use_cases": ["proximity analysis", "impact zones", "service areas", "safety buffers"],
                "parameters": {
                    "distance": "Buffer distance in map units",
                    "units": "Distance units (meters, feet, degrees)",
                    "segments": "Number of segments to approximate curves",
                    "cap_style": "End cap style (round, flat, square)",
                    "join_style": "Join style (round, mitre, bevel)"
                },
                "tools": ["geopandas", "qgis", "gdal", "shapely"],
                "input_types": ["point", "line", "polygon"],
                "output_types": ["polygon"],
                "complexity": "low",
                "best_practices": [
                    "Choose appropriate buffer distance based on analysis scale",
                    "Consider using projected coordinate systems for accurate distance calculations",
                    "Use dissolve option to merge overlapping buffers if needed"
                ]
            },
            
            "spatial_join": {
                "description": "Joins attributes from one layer to another based on spatial relationships",
                "use_cases": ["point-in-polygon analysis", "attribute transfer", "spatial aggregation"],
                "parameters": {
                    "join_type": "Type of join (inner, left, right)",
                    "spatial_predicate": "Spatial relationship (intersects, contains, within, touches)"
                },
                "tools": ["geopandas", "qgis", "arcgis"],
                "input_types": ["vector", "vector"],
                "output_types": ["vector"],
                "complexity": "medium",
                "best_practices": [
                    "Ensure both layers have same coordinate reference system",
                    "Use spatial index for better performance with large datasets",
                    "Choose appropriate spatial predicate for your analysis"
                ]
            },
            
            "overlay_analysis": {
                "description": "Combines multiple vector layers to create new features",
                "use_cases": ["land use analysis", "suitability modeling", "impact assessment"],
                "parameters": {
                    "overlay_type": "Type of overlay (union, intersection, difference, symmetric_difference)",
                    "keep_geom_type": "Whether to keep original geometry type"
                },
                "tools": ["geopandas", "qgis", "grass"],
                "input_types": ["polygon", "polygon"],
                "output_types": ["polygon"],
                "complexity": "high",
                "best_practices": [
                    "Validate geometries before overlay operations",
                    "Use appropriate topology tolerance",
                    "Consider memory usage with large datasets"
                ]
            },
            
            "raster_calculator": {
                "description": "Performs mathematical operations on raster datasets",
                "use_cases": ["NDVI calculation", "slope analysis", "terrain modeling", "change detection"],
                "parameters": {
                    "expression": "Mathematical expression to apply",
                    "output_type": "Output data type (float32, int16, etc.)",
                    "no_data_value": "Value to use for no-data pixels"
                },
                "tools": ["gdal", "rasterio", "qgis", "grass"],
                "input_types": ["raster"],
                "output_types": ["raster"],
                "complexity": "medium",
                "best_practices": [
                    "Ensure all input rasters have same extent and resolution",
                    "Handle no-data values appropriately",
                    "Choose appropriate output data type to avoid precision loss"
                ]
            },
            
            "clip": {
                "description": "Clips input features to the extent of clip features",
                "use_cases": ["area of interest extraction", "data masking", "boundary analysis"],
                "parameters": {
                    "clip_layer": "Layer to use as clipping boundary"
                },
                "tools": ["geopandas", "qgis", "gdal"],
                "input_types": ["vector", "raster"],
                "output_types": ["vector", "raster"],
                "complexity": "low",
                "best_practices": [
                    "Ensure clip layer completely covers area of interest",
                    "Use appropriate coordinate reference system",
                    "Consider using buffer on clip layer for edge effects"
                ]
            },
            
            "proximity_analysis": {
                "description": "Calculates distance to nearest features",
                "use_cases": ["accessibility analysis", "service area modeling", "cost-distance analysis"],
                "parameters": {
                    "max_distance": "Maximum distance to calculate",
                    "units": "Distance units",
                    "distunits": "Units for distance values"
                },
                "tools": ["gdal", "qgis", "grass"],
                "input_types": ["vector", "raster"],
                "output_types": ["raster"],
                "complexity": "medium",
                "best_practices": [
                    "Use appropriate cell size for analysis scale",
                    "Consider using cost-distance for realistic modeling",
                    "Validate results with known distances"
                ]
            },
            
            "slope_analysis": {
                "description": "Calculates slope from elevation data",
                "use_cases": ["terrain analysis", "landslide risk assessment", "watershed modeling"],
                "parameters": {
                    "slope_type": "Output units (degrees, percent, radians)",
                    "algorithm": "Calculation algorithm (Horn, Zevenbergen-Thorne)"
                },
                "tools": ["gdal", "grass", "qgis"],
                "input_types": ["raster"],
                "output_types": ["raster"],
                "complexity": "medium",
                "best_practices": [
                    "Use high-resolution DEM for accurate results",
                    "Consider smoothing noisy elevation data",
                    "Validate results in known terrain areas"
                ]
            }
        }
        
        # Workflow templates for common spatial analysis tasks
        self.workflow_templates = {
            "flood_risk_assessment": {
                "description": "Assess flood risk for given area",
                "steps": [
                    "Load elevation data (DEM)",
                    "Load water bodies and rivers",
                    "Create buffers around water bodies",
                    "Calculate slope from DEM",
                    "Identify low-lying areas",
                    "Overlay with urban areas",
                    "Classify flood risk levels"
                ],
                "data_requirements": ["DEM", "water_bodies", "urban_boundaries"],
                "complexity": "high"
            },
            
            "site_suitability": {
                "description": "Find suitable locations based on criteria",
                "steps": [
                    "Load constraint layers",
                    "Create buffer zones around exclusion areas",
                    "Load factor layers (slope, accessibility, etc.)",
                    "Reclassify factor layers",
                    "Apply weighted overlay",
                    "Exclude constrained areas",
                    "Rank suitable sites"
                ],
                "data_requirements": ["constraint_layers", "factor_layers"],
                "complexity": "high"
            },
            
            "urban_heat_island": {
                "description": "Analyze urban heat island effect",
                "steps": [
                    "Load satellite thermal imagery",
                    "Load land cover data",
                    "Calculate land surface temperature",
                    "Calculate vegetation indices",
                    "Identify urban and rural areas",
                    "Compare temperature differences",
                    "Create heat island map"
                ],
                "data_requirements": ["satellite_imagery", "land_cover", "urban_boundaries"],
                "complexity": "medium"
            }
        }
        
        # Best practices for spatial analysis
        self.best_practices = {
            "coordinate_systems": [
                "Always check and align coordinate reference systems",
                "Use projected CRS for distance and area calculations",
                "Document CRS transformations performed",
                "Validate results after CRS transformations"
            ],
            "data_quality": [
                "Validate geometry before analysis",
                "Check for missing or invalid data",
                "Document data sources and vintage",
                "Perform quality checks on outputs"
            ],
            "performance": [
                "Use spatial indexes for large datasets",
                "Process data in appropriate chunks",
                "Consider using simplified geometries for analysis",
                "Monitor memory usage with large datasets"
            ],
            "error_handling": [
                "Implement robust error handling",
                "Validate inputs before processing",
                "Provide meaningful error messages",
                "Log processing steps for debugging"
            ]
        }
    
    def build_knowledge_base(self, force_rebuild: bool = False):
        """
        Build or load knowledge base from documentation
        
        Args:
            force_rebuild: Whether to rebuild from scratch
        """
        if self.vector_store_path.exists() and not force_rebuild:
            logger.info("Loading existing vector store")
            self._load_vector_store()
        else:
            logger.info("Building new knowledge base")
            self._build_new_knowledge_base()
            self._save_vector_store()
    
    def _build_new_knowledge_base(self):
        """Build knowledge base from documentation and built-in knowledge"""
        
        # Add built-in GIS operations knowledge
        self._add_gis_operations_to_kb()
        
        # Add workflow templates
        self._add_workflow_templates_to_kb()
        
        # Add best practices
        self._add_best_practices_to_kb()
        
        # Load documentation files if they exist
        if self.docs_path.exists():
            self._load_documentation_files()
        
        # Create embeddings for all documents
        if self.documents:
            logger.info(f"Creating embeddings for {len(self.documents)} documents")
            self.embeddings = self.embedding_model.encode(
                self.documents, 
                convert_to_tensor=False,
                show_progress_bar=True
            )
        else:
            logger.warning("No documents found to create embeddings")
    
    def _add_gis_operations_to_kb(self):
        """Add GIS operations to knowledge base"""
        
        for operation, details in self.gis_operations_db.items():
            # Main operation description
            doc_text = f"GIS Operation: {operation}\n"
            doc_text += f"Description: {details['description']}\n"
            doc_text += f"Complexity: {details['complexity']}\n"
            doc_text += f"Input types: {', '.join(details['