"""
KGRAG project configuration file
Manages paths and settings shared across all scripts
"""

from pathlib import Path
import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class KGRAGConfig:
    """KGRAG project configuration management class"""
    
    def __init__(self, dataset_name: str = None):
        self.project_root = Path(__file__).parent
        self.dataset_name = dataset_name
        
        # Basic directory structure
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"
        self.temp_dir = self.project_root / "temp"
        
        # Results directory structure
        self.index_results_dir = self.results_dir / "index"
        self.generated_results_dir = self.results_dir / "generated"
        self.evaluated_results_dir = self.results_dir / "evaluated"
        self.chunks_dir = self.results_dir / "chunks"
        
        # Load configuration from environment variables
        self._load_config()
        
        # Create directories
        self._ensure_directories()
        
    def _load_config(self):
        """Load configuration from environment variables."""
        # API settings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Model settings
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.eval_model = os.getenv("EVAL_MODEL", "gpt-4o-mini")
        
        # Model parameters  
        self.temperature = float(os.getenv("TEMPERATURE", "0.5"))
        self.max_tokens_response = int(os.getenv("MAX_TOKENS_RESPONSE", "2000"))
        self.answer_temperature = float(os.getenv("ANSWER_TEMPERATURE", "0.3"))
        self.answer_max_tokens = int(os.getenv("ANSWER_MAX_TOKENS", "1000"))
        self.eval_temperature = float(os.getenv("EVAL_TEMPERATURE", "0.1"))
        
        # Text processing settings
        self.max_tokens = int(os.getenv("MAX_TOKENS", "3000"))
        self.overlap = int(os.getenv("OVERLAP", "300"))
        self.max_workers = int(os.getenv("MAX_WORKERS", "10"))
        self.alt_max_tokens = int(os.getenv("ALT_MAX_TOKENS", "1200"))
        self.alt_overlap = int(os.getenv("ALT_OVERLAP", "100"))
        
        # Topic/subtopic selection settings
        self.topic_choice_min = int(os.getenv("TOPIC_CHOICE_MIN", "5"))
        self.topic_choice_max = int(os.getenv("TOPIC_CHOICE_MAX", "10"))
        self.subtopic_choice_min = int(os.getenv("SUBTOPIC_CHOICE_MIN", "10"))
        self.subtopic_choice_max = int(os.getenv("SUBTOPIC_CHOICE_MAX", "25"))
        
        # Retry settings
        self.max_retries = int(os.getenv("MAX_RETRIES", "10"))
        self.retry_backoff = float(os.getenv("RETRY_BACKOFF", "0.2"))
        
        # RAG retrieval parameters
        self.top_k1 = int(os.getenv("TOP_K1", "50"))
        self.top_k2 = int(os.getenv("TOP_K2", "10"))
        self.top_k1_long = int(os.getenv("TOP_K1_LONG", "25"))
        self.top_k2_long = int(os.getenv("TOP_K2_LONG", "5"))
        self.embedding_top_k = int(os.getenv("EMBEDDING_TOP_K", "5"))
        self.overretrieve_factor = int(os.getenv("OVERRETRIEVE_FACTOR", "5"))
        
        # Context settings
        self.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
        
        # System settings
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "kgrag.log")
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "30"))
        self.enable_cache = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
    
    def _ensure_directories(self):
        """Create necessary directories."""
        dirs = [
            self.data_dir, self.results_dir, self.temp_dir,
            self.index_results_dir, self.generated_results_dir,
            self.evaluated_results_dir, self.chunks_dir
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_dir(self, dataset_name: str = None) -> Path:
        """Return dataset directory path."""
        name = dataset_name or self.dataset_name
        if not name:
            raise ValueError("Dataset name must be provided")
        return self.data_dir / name
    
    def get_input_file(self, dataset_name: str = None) -> Path:
        """Return input contexts.txt file path."""
        dataset_dir = self.get_dataset_dir(dataset_name)
        return dataset_dir / "contexts.txt"
    
    def get_qa_file(self, dataset_name: str = None) -> Path:
        """Return QA JSON file path."""
        name = dataset_name or self.dataset_name
        return self.index_results_dir / f"{name}_qa.json"
    
    def get_graph_json_file(self, dataset_name: str = None) -> Path:
        """Return graph JSON file path."""
        name = dataset_name or self.dataset_name
        return self.index_results_dir / f"{name}_graph.json"
    
    def get_graph_gexf_file(self, dataset_name: str = None) -> Path:
        """Return graph GEXF file path."""
        name = dataset_name or self.dataset_name
        return self.index_results_dir / f"{name}_graph.gexf"
    
    def get_kv_store_file(self, dataset_name: str = None) -> Path:
        """Return KV store file path."""
        name = dataset_name or self.dataset_name
        return self.index_results_dir / f"{name}_kv_store.json"
    
    def get_edge_index_file(self, dataset_name: str = None) -> Path:
        """Return edge index FAISS file path."""
        name = dataset_name or self.dataset_name
        return self.index_results_dir / f"{name}_edge_index.faiss"
    
    def get_edge_payload_file(self, dataset_name: str = None) -> Path:
        """Return edge payload file path."""
        name = dataset_name or self.dataset_name
        return self.index_results_dir / f"{name}_edge_payloads.npy"
    
    def get_answer_file(self, dataset_name: str = None, answer_type: str = "short") -> Path:
        """Return answer generation result file path."""
        name = dataset_name or self.dataset_name
        return self.generated_results_dir / f"{name}_answers_{answer_type}.json"
    
    def get_chunk_log_file(self, dataset_name: str = None, answer_type: str = "short") -> Path:
        """Return chunk log file path."""
        name = dataset_name or self.dataset_name
        return self.chunks_dir / f"{name}_chunks_{answer_type}.jsonl"
    
    def get_evaluation_file(self, dataset_name: str = None, eval_method: str = "f1") -> Path:
        """Return evaluation result file path."""
        name = dataset_name or self.dataset_name
        return self.evaluated_results_dir / f"{name}_eval_{eval_method}.json"
    
    def get_pipeline_state_file(self) -> Path:
        """파이프라인 상태 파일 경로를 반환합니다."""
        return self.temp_dir / "pipeline_state.json"
    
    def save_pipeline_state(self, state: Dict):
        """파이프라인 상태를 저장합니다."""
        with open(self.get_pipeline_state_file(), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def load_pipeline_state(self) -> Optional[Dict]:
        """Load pipeline state."""
        state_file = self.get_pipeline_state_file()
        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def clear_pipeline_state(self):
        """Clear pipeline state."""
        state_file = self.get_pipeline_state_file()
        if state_file.exists():
            state_file.unlink()
    
    def list_available_datasets(self) -> list:
        """Return list of available datasets."""
        if not self.data_dir.exists():
            return []
        
        datasets = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and (item / "contexts.txt").exists():
                datasets.append(item.name)
        return sorted(datasets)
    
    def list_indexed_datasets(self) -> list:
        """Return list of indexed datasets."""
        if not self.index_results_dir.exists():
            return []
        
        datasets = []
        for item in self.index_results_dir.iterdir():
            if item.name.endswith("_qa.json"):
                dataset_name = item.name[:-7]  # Remove _qa.json
                datasets.append(dataset_name)
        return sorted(datasets)
    
    def list_generated_datasets(self) -> list:
        """Return list of datasets with generated answers."""
        if not self.generated_results_dir.exists():
            return []
        
        datasets = []
        for item in self.generated_results_dir.iterdir():
            if "_answers_" in item.name and item.name.endswith(".json"):
                # Extract dataset from dataset_answers_type.json format
                dataset_name = item.name.split("_answers_")[0]
                if dataset_name not in datasets:
                    datasets.append(dataset_name)
        return sorted(datasets)

# Global configuration instance
config = KGRAGConfig()

def get_config(dataset_name: str = None) -> KGRAGConfig:
    """Get configuration instance."""
    if dataset_name:
        return KGRAGConfig(dataset_name)
    return config
