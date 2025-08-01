"""
KGRAG 프로젝트 공통 설정 파일
모든 스크립트에서 공유하는 경로 및 설정 관리
"""

from pathlib import Path
import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class KGRAGConfig:
    """KGRAG 프로젝트 설정 관리 클래스"""
    
    def __init__(self, dataset_name: str = None):
        self.project_root = Path(__file__).parent
        self.dataset_name = dataset_name
        
        # 기본 디렉터리 구조
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"
        self.temp_dir = self.project_root / "temp"
        
        # 결과 디렉터리 구조
        self.index_results_dir = self.results_dir / "index"
        self.generated_results_dir = self.results_dir / "generated"
        self.evaluated_results_dir = self.results_dir / "evaluated"
        self.chunks_dir = self.results_dir / "chunks"
        
        # 환경 변수에서 설정 로드
        self._load_config()
        
        # 디렉터리 생성
        self._ensure_directories()
        
    def _load_config(self):
        """환경 변수에서 설정을 로드합니다."""
        # API 설정
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # 모델 설정
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.eval_model = os.getenv("EVAL_MODEL", "gpt-4o-mini")
        
        # 모델 파라미터
        self.temperature = float(os.getenv("TEMPERATURE", "0.5"))
        self.max_tokens_response = int(os.getenv("MAX_TOKENS_RESPONSE", "2000"))
        self.answer_temperature = float(os.getenv("ANSWER_TEMPERATURE", "0.3"))
        self.answer_max_tokens = int(os.getenv("ANSWER_MAX_TOKENS", "1000"))
        self.eval_temperature = float(os.getenv("EVAL_TEMPERATURE", "0.1"))
        
        # 텍스트 처리 설정
        self.max_tokens = int(os.getenv("MAX_TOKENS", "3000"))
        self.overlap = int(os.getenv("OVERLAP", "300"))
        self.max_workers = int(os.getenv("MAX_WORKERS", "10"))
        self.alt_max_tokens = int(os.getenv("ALT_MAX_TOKENS", "1200"))
        self.alt_overlap = int(os.getenv("ALT_OVERLAP", "100"))
        
        # 토픽/서브토픽 선택 설정
        self.topic_choice_min = int(os.getenv("TOPIC_CHOICE_MIN", "5"))
        self.topic_choice_max = int(os.getenv("TOPIC_CHOICE_MAX", "10"))
        self.subtopic_choice_min = int(os.getenv("SUBTOPIC_CHOICE_MIN", "10"))
        self.subtopic_choice_max = int(os.getenv("SUBTOPIC_CHOICE_MAX", "25"))
        
        # 재시도 설정
        self.max_retries = int(os.getenv("MAX_RETRIES", "10"))
        self.retry_backoff = float(os.getenv("RETRY_BACKOFF", "0.2"))
        
        # RAG 검색 파라미터
        self.top_k1 = int(os.getenv("TOP_K1", "50"))
        self.top_k2 = int(os.getenv("TOP_K2", "10"))
        self.top_k1_long = int(os.getenv("TOP_K1_LONG", "25"))
        self.top_k2_long = int(os.getenv("TOP_K2_LONG", "5"))
        self.embedding_top_k = int(os.getenv("EMBEDDING_TOP_K", "5"))
        self.overretrieve_factor = int(os.getenv("OVERRETRIEVE_FACTOR", "5"))
        
        # 컨텍스트 설정
        self.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
        
        # 시스템 설정
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "kgrag.log")
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "30"))
        self.enable_cache = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
    
    def _ensure_directories(self):
        """필요한 디렉터리들을 생성합니다."""
        dirs = [
            self.data_dir, self.results_dir, self.temp_dir,
            self.index_results_dir, self.generated_results_dir,
            self.evaluated_results_dir, self.chunks_dir
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_dir(self, dataset_name: str = None) -> Path:
        """데이터셋 디렉터리 경로를 반환합니다."""
        name = dataset_name or self.dataset_name
        if not name:
            raise ValueError("Dataset name must be provided")
        return self.data_dir / name
    
    def get_input_file(self, dataset_name: str = None) -> Path:
        """입력 contexts.txt 파일 경로를 반환합니다."""
        dataset_dir = self.get_dataset_dir(dataset_name)
        return dataset_dir / "contexts.txt"
    
    def get_qa_file(self, dataset_name: str = None) -> Path:
        """QA JSON 파일 경로를 반환합니다."""
        name = dataset_name or self.dataset_name
        return self.index_results_dir / f"{name}_qa.json"
    
    def get_graph_json_file(self, dataset_name: str = None) -> Path:
        """그래프 JSON 파일 경로를 반환합니다."""
        name = dataset_name or self.dataset_name
        return self.index_results_dir / f"{name}_graph.json"
    
    def get_graph_gexf_file(self, dataset_name: str = None) -> Path:
        """그래프 GEXF 파일 경로를 반환합니다."""
        name = dataset_name or self.dataset_name
        return self.index_results_dir / f"{name}_graph.gexf"
    
    def get_kv_store_file(self, dataset_name: str = None) -> Path:
        """KV store 파일 경로를 반환합니다."""
        name = dataset_name or self.dataset_name
        return self.index_results_dir / f"{name}_kv_store.json"
    
    def get_edge_index_file(self, dataset_name: str = None) -> Path:
        """Edge 인덱스 FAISS 파일 경로를 반환합니다."""
        name = dataset_name or self.dataset_name
        return self.index_results_dir / f"{name}_edge_index.faiss"
    
    def get_edge_payload_file(self, dataset_name: str = None) -> Path:
        """Edge payload 파일 경로를 반환합니다."""
        name = dataset_name or self.dataset_name
        return self.index_results_dir / f"{name}_edge_payloads.npy"
    
    def get_answer_file(self, dataset_name: str = None, answer_type: str = "short") -> Path:
        """답변 생성 결과 파일 경로를 반환합니다."""
        name = dataset_name or self.dataset_name
        return self.generated_results_dir / f"{name}_answers_{answer_type}.json"
    
    def get_chunk_log_file(self, dataset_name: str = None, answer_type: str = "short") -> Path:
        """청크 로그 파일 경로를 반환합니다."""
        name = dataset_name or self.dataset_name
        return self.chunks_dir / f"{name}_chunks_{answer_type}.jsonl"
    
    def get_evaluation_file(self, dataset_name: str = None, eval_method: str = "f1") -> Path:
        """평가 결과 파일 경로를 반환합니다."""
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
        """파이프라인 상태를 로드합니다."""
        state_file = self.get_pipeline_state_file()
        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def clear_pipeline_state(self):
        """파이프라인 상태를 초기화합니다."""
        state_file = self.get_pipeline_state_file()
        if state_file.exists():
            state_file.unlink()
    
    def list_available_datasets(self) -> list:
        """사용 가능한 데이터셋 목록을 반환합니다."""
        if not self.data_dir.exists():
            return []
        
        datasets = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and (item / "contexts.txt").exists():
                datasets.append(item.name)
        return sorted(datasets)
    
    def list_indexed_datasets(self) -> list:
        """인덱싱이 완료된 데이터셋 목록을 반환합니다."""
        if not self.index_results_dir.exists():
            return []
        
        datasets = []
        for item in self.index_results_dir.iterdir():
            if item.name.endswith("_qa.json"):
                dataset_name = item.name[:-7]  # _qa.json 제거
                datasets.append(dataset_name)
        return sorted(datasets)
    
    def list_generated_datasets(self) -> list:
        """답변 생성이 완료된 데이터셋 목록을 반환합니다."""
        if not self.generated_results_dir.exists():
            return []
        
        datasets = []
        for item in self.generated_results_dir.iterdir():
            if "_answers_" in item.name and item.name.endswith(".json"):
                # dataset_answers_type.json 형식에서 dataset 추출
                dataset_name = item.name.split("_answers_")[0]
                if dataset_name not in datasets:
                    datasets.append(dataset_name)
        return sorted(datasets)

# 전역 설정 인스턴스
config = KGRAGConfig()

def get_config(dataset_name: str = None) -> KGRAGConfig:
    """설정 인스턴스를 가져옵니다."""
    if dataset_name:
        return KGRAGConfig(dataset_name)
    return config
