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
            },

            "watershed_analysis": {
                "description": "Delineates watersheds and drainage basins from elevation data",
                "use_cases": ["hydrological modeling", "drainage analysis", "flood risk assessment"],
                "parameters": {
                    "pour_points": "Outlet points for watershed delineation",
                    "flow_direction": "Flow direction raster",
                    "threshold": "Minimum drainage area threshold"
                },
                "tools": ["grass", "qgis", "arcgis"],
                "input_types": ["raster"],
                "output_types": ["vector", "raster"],
                "complexity": "high",
                "best_practices": [
                    "Use high-quality DEM data",
                    "Fill sinks before analysis",
                    "Validate results with known watersheds"
                ]
            },

            "visibility_analysis": {
                "description": "Calculates visible areas from observer points",
                "use_cases": ["viewshed analysis", "tower placement", "landscape planning"],
                "parameters": {
                    "observer_points": "Points from which to calculate visibility",
                    "observer_height": "Height of observer above ground",
                    "target_height": "Height of target objects"
                },
                "tools": ["grass", "qgis", "gdal"],
                "input_types": ["raster", "vector"],
                "output_types": ["raster"],
                "complexity": "medium",
                "best_practices": [
                    "Use appropriate DEM resolution for analysis scale",
                    "Consider earth curvature for large areas",
                    "Account for vegetation and buildings"
                ]
            },

            "network_analysis": {
                "description": "Analyzes spatial networks for routing and accessibility",
                "use_cases": ["shortest path", "service areas", "accessibility analysis"],
                "parameters": {
                    "network_dataset": "Road or path network",
                    "impedance": "Cost attribute (distance, time, etc.)",
                    "barriers": "Network barriers or restrictions"
                },
                "tools": ["osmnx", "qgis", "arcgis"],
                "input_types": ["vector"],
                "output_types": ["vector"],
                "complexity": "high",
                "best_practices": [
                    "Ensure network topology is correct",
                    "Use appropriate impedance attributes",
                    "Consider one-way restrictions and barriers"
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
                "complexity": "high",
                "operations": ["buffer_analysis", "slope_analysis", "overlay_analysis", "raster_calculator"]
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
                "complexity": "high",
                "operations": ["buffer_analysis", "overlay_analysis", "raster_calculator"]
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
                "complexity": "medium",
                "operations": ["raster_calculator", "overlay_analysis", "clip"]
            },

            "accessibility_analysis": {
                "description": "Analyze accessibility to services or facilities",
                "steps": [
                    "Load road network data",
                    "Load facility locations",
                    "Calculate travel times/distances",
                    "Create service areas",
                    "Analyze population access",
                    "Identify underserved areas"
                ],
                "data_requirements": ["road_network", "facilities", "population_data"],
                "complexity": "high",
                "operations": ["network_analysis", "buffer_analysis", "overlay_analysis"]
            },

            "land_change_detection": {
                "description": "Detect changes in land use/cover over time",
                "steps": [
                    "Load multi-temporal imagery",
                    "Perform geometric correction",
                    "Calculate change indices",
                    "Classify change types",
                    "Quantify change areas",
                    "Create change maps"
                ],
                "data_requirements": ["multi_temporal_imagery", "reference_data"],
                "complexity": "medium",
                "operations": ["raster_calculator", "overlay_analysis"]
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
            ],
            "workflow_design": [
                "Break complex analyses into logical steps",
                "Use intermediate outputs for validation",
                "Document assumptions and limitations",
                "Consider alternative approaches"
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
            doc_text += f"Input types: {', '.join(details['input_types'])}\n"
            doc_text += f"Output types: {', '.join(details['output_types'])}\n"
            doc_text += f"Tools: {', '.join(details['tools'])}\n"
            doc_text += f"Use cases: {', '.join(details['use_cases'])}\n"
            
            # Parameters
            doc_text += "Parameters:\n"
            for param, desc in details['parameters'].items():
                doc_text += f"- {param}: {desc}\n"
            
            # Best practices
            doc_text += "Best practices:\n"
            for practice in details['best_practices']:
                doc_text += f"- {practice}\n"
            
            self.documents.append(doc_text)
            self.metadata.append({
                "type": "gis_operation",
                "name": operation,
                "complexity": details['complexity'],
                "tools": details['tools']
            })
    
    def _add_workflow_templates_to_kb(self):
        """Add workflow templates to knowledge base"""
        
        for template_name, template in self.workflow_templates.items():
            doc_text = f"Workflow Template: {template_name}\n"
            doc_text += f"Description: {template['description']}\n"
            doc_text += f"Complexity: {template['complexity']}\n"
            
            # Steps
            doc_text += "Workflow Steps:\n"
            for i, step in enumerate(template['steps'], 1):
                doc_text += f"{i}. {step}\n"
            
            # Data requirements
            doc_text += f"Data Requirements: {', '.join(template['data_requirements'])}\n"
            
            # Operations
            if 'operations' in template:
                doc_text += f"GIS Operations Used: {', '.join(template['operations'])}\n"
            
            self.documents.append(doc_text)
            self.metadata.append({
                "type": "workflow_template",
                "name": template_name,
                "complexity": template['complexity'],
                "operations": template.get('operations', [])
            })
    
    def _add_best_practices_to_kb(self):
        """Add best practices to knowledge base"""
        
        for category, practices in self.best_practices.items():
            doc_text = f"Best Practices: {category.replace('_', ' ').title()}\n"
            doc_text += f"Category: {category}\n"
            
            for practice in practices:
                doc_text += f"- {practice}\n"
            
            self.documents.append(doc_text)
            self.metadata.append({
                "type": "best_practice",
                "category": category
            })
    
    def _load_documentation_files(self):
        """Load documentation files from docs directory"""
        
        supported_extensions = ['.txt', '.md', '.rst']
        
        for ext in supported_extensions:
            doc_files = list(self.docs_path.glob(f'*{ext}'))
            
            for doc_file in doc_files:
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split large documents into chunks
                    chunks = self._split_document(content)
                    
                    for i, chunk in enumerate(chunks):
                        self.documents.append(chunk)
                        self.metadata.append({
                            "type": "documentation",
                            "file": doc_file.name,
                            "chunk": i
                        })
                        
                    logger.info(f"Loaded {len(chunks)} chunks from {doc_file.name}")
                    
                except Exception as e:
                    logger.error(f"Error loading {doc_file}: {e}")
    
    def _split_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split document into overlapping chunks"""
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def _save_vector_store(self):
        """Save vector store to disk"""
        
        store_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'embedding_model': self.embedding_model_name
        }
        
        try:
            with open(self.vector_store_path, 'wb') as f:
                pickle.dump(store_data, f)
            logger.info(f"Vector store saved to {self.vector_store_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def _load_vector_store(self):
        """Load vector store from disk"""
        
        try:
            with open(self.vector_store_path, 'rb') as f:
                store_data = pickle.load(f)
            
            self.documents = store_data['documents']
            self.embeddings = store_data['embeddings']
            self.metadata = store_data['metadata']
            
            # Verify embedding model compatibility
            if store_data.get('embedding_model') != self.embedding_model_name:
                logger.warning("Embedding model mismatch - rebuilding embeddings")
                self.embeddings = self.embedding_model.encode(
                    self.documents, 
                    convert_to_tensor=False,
                    show_progress_bar=True
                )
            
            logger.info(f"Vector store loaded with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self._build_new_knowledge_base()
    
    def retrieve_relevant_context(self, query: str, k: int = 5, min_similarity: float = 0.1) -> str:
        """
        Retrieve relevant context for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            Concatenated relevant context
        """
        
        if not self.documents or self.embeddings is None:
            logger.warning("No knowledge base available")
            return ""
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k similar documents
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Filter by minimum similarity
        relevant_docs = []
        for idx in top_indices:
            if similarities[idx] >= min_similarity:
                relevant_docs.append({
                    'document': self.documents[idx],
                    'similarity': similarities[idx],
                    'metadata': self.metadata[idx]
                })
        
        # Format context
        context = self._format_context(relevant_docs)
        
        return context
    
    def _format_context(self, relevant_docs: List[Dict]) -> str:
        """Format retrieved documents into context"""
        
        if not relevant_docs:
            return ""
        
        context = "Relevant GIS Knowledge:\n\n"
        
        for i, doc in enumerate(relevant_docs, 1):
            context += f"[{i}] {doc['document']}\n"
            context += f"Similarity: {doc['similarity']:.3f}\n"
            context += f"Type: {doc['metadata'].get('type', 'unknown')}\n"
            context += "---\n"
        
        return context
    
    def search_operations(self, query: str, operation_type: str = None) -> List[Dict]:
        """
        Search for specific GIS operations
        
        Args:
            query: Search query
            operation_type: Filter by operation type
            
        Returns:
            List of matching operations
        """
        
        matching_ops = []
        
        for op_name, op_details in self.gis_operations_db.items():
            # Check if query matches operation name or description
            if (query.lower() in op_name.lower() or 
                query.lower() in op_details['description'].lower() or
                any(query.lower() in uc.lower() for uc in op_details['use_cases'])):
                
                if operation_type is None or op_details.get('complexity') == operation_type:
                    matching_ops.append({
                        'name': op_name,
                        'details': op_details
                    })
        
        return matching_ops
    
    def get_workflow_template(self, template_name: str) -> Dict:
        """Get specific workflow template"""
        
        return self.workflow_templates.get(template_name, {})
    
    def get_best_practices(self, category: str = None) -> Dict:
        """Get best practices by category"""
        
        if category:
            return {category: self.best_practices.get(category, [])}
        else:
            return self.best_practices
    
    def add_custom_operation(self, operation_name: str, operation_details: Dict):
        """Add custom GIS operation to knowledge base"""
        
        self.gis_operations_db[operation_name] = operation_details
        
        # Add to documents for retrieval
        doc_text = f"GIS Operation: {operation_name}\n"
        doc_text += f"Description: {operation_details['description']}\n"
        # Add other details...
        
        self.documents.append(doc_text)
        self.metadata.append({
            "type": "custom_operation",
            "name": operation_name
        })
        
        # Update embeddings
        if self.embeddings is not None:
            new_embedding = self.embedding_model.encode([doc_text], convert_to_tensor=False)
            self.embeddings = np.vstack([self.embeddings, new_embedding])
        
        logger.info(f"Added custom operation: {operation_name}")
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        
        stats = {
            "total_documents": len(self.documents),
            "gis_operations": len(self.gis_operations_db),
            "workflow_templates": len(self.workflow_templates),
            "best_practice_categories": len(self.best_practices),
            "has_embeddings": self.embeddings is not None
        }
        
        # Document type breakdown
        doc_types = defaultdict(int)
        for meta in self.metadata:
            doc_types[meta.get('type', 'unknown')] += 1
        
        stats['document_types'] = dict(doc_types)
        
        return stats
    
    def export_knowledge_base(self, export_path: str):
        """Export knowledge base to JSON file"""
        
        export_data = {
            "gis_operations": self.gis_operations_db,
            "workflow_templates": self.workflow_templates,
            "best_practices": self.best_practices,
            "documents": self.documents,
            "metadata": self.metadata
        }
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Knowledge base exported to {export_path}")
        except Exception as e:
            logger.error(f"Error exporting knowledge base: {e}")
    
    def import_knowledge_base(self, import_path: str):
        """Import knowledge base from JSON file"""
        
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Merge with existing knowledge
            if 'gis_operations' in import_data:
                self.gis_operations_db.update(import_data['gis_operations'])
            
            if 'workflow_templates' in import_data:
                self.workflow_templates.update(import_data['workflow_templates'])
            
            if 'best_practices' in import_data:
                self.best_practices.update(import_data['best_practices'])
            
            # Rebuild knowledge base
            self._build_new_knowledge_base()
            
            logger.info(f"Knowledge base imported from {import_path}")
            
        except Exception as e:
            logger.error(f"Error importing knowledge base: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG system
    rag = GISKnowledgeRAG()
    
    # Build knowledge base
    rag.build_knowledge_base()
    
    # Test retrieval
    query = "How to perform buffer analysis for flood risk assessment?"
    context = rag.retrieve_relevant_context(query)
    print("Query:", query)
    print("Context:", context)
    
    # Search operations
    flood_ops = rag.search_operations("flood")
    print("\nFlood-related operations:")
    for op in flood_ops:
        print(f"- {op['name']}: {op['details']['description']}")
    
    # Get statistics
    stats = rag.get_statistics()
    print("\nKnowledge base statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")
