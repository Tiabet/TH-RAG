#!/usr/bin/env python3
"""
KGRAG integrated pipeline script
Automates data flow by executing all steps sequentially.
"""

import sys
import os
import argparse
from pathlib import Path
import json

# Set project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "index"))
sys.path.insert(0, str(PROJECT_ROOT / "generate"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluate"))

from config import get_config

def run_pipeline(dataset_name: str, steps: list = None, force_rebuild: bool = False):
    """
    KGRAG 파이프라인 실행
    
    Args:
        dataset_name: 처리할 데이터셋 이름
        steps: 실행할 단계 리스트 (None이면 모든 단계)
        force_rebuild: 기존 결과가 있어도 강제로 재실행
    """
    config = get_config(dataset_name)
    
    # 사용 가능한 단계들
    available_steps = [
        'graph_construction',
        'json_to_gexf', 
        'edge_embedding',
        'answer_generation_short',
        'answer_generation_long',
        'evaluation_f1'
    ]
    
    if steps is None:
        steps = available_steps
    
    # 파이프라인 상태 로드
    state = config.load_pipeline_state() or {}
    dataset_state = state.get(dataset_name, {})
    
    print(f"🚀 Starting KGRAG pipeline for dataset: {dataset_name}")
    print(f"📋 Steps to run: {', '.join(steps)}")
    print("=" * 70)
    
    results = {}
    
    # 1. Graph Construction (QA 생성)
    if 'graph_construction' in steps:
        print("\n📊 Step 1: Graph Construction")
        print("-" * 30)
        
        if not force_rebuild and dataset_state.get('graph_construction', {}).get('completed'):
            print("✅ Already completed. Use --force to rebuild.")
            qa_file = config.get_qa_file()
        else:
            try:
                from index.graph_construction import main as gc_main
                qa_file = gc_main(dataset_name)
                print(f"✅ Graph construction completed: {qa_file}")
                results['graph_construction'] = qa_file
            except Exception as e:
                print(f"❌ Graph construction failed: {e}")
                return None
    
    # 2. JSON to GEXF 변환
    if 'json_to_gexf' in steps:
        print("\n🔄 Step 2: JSON to GEXF Conversion")
        print("-" * 30)
        
        if not force_rebuild and dataset_state.get('json_to_gexf', {}).get('completed'):
            print("✅ Already completed. Use --force to rebuild.")
        else:
            try:
                from index.json_to_gexf import convert_json_to_gexf
                json_file = str(config.get_graph_json_file()) 
                gexf_file = str(config.get_graph_gexf_file())
                
                # QA 파일에서 그래프 JSON 생성이 필요한 경우 여기서 처리
                convert_json_to_gexf(json_file)
                print(f"✅ GEXF conversion completed: {gexf_file}")
                
                # 상태 업데이트
                dataset_state['json_to_gexf'] = {'completed': True, 'gexf_file': gexf_file}
                state[dataset_name] = dataset_state
                config.save_pipeline_state(state)
                
                results['json_to_gexf'] = gexf_file
            except Exception as e:
                print(f"❌ GEXF conversion failed: {e}")
                if 'graph_construction' not in results:
                    print("💡 Hint: Run graph_construction step first")
                return None
    
    # 3. Edge Embedding
    if 'edge_embedding' in steps:
        print("\n🔍 Step 3: Edge Embedding")
        print("-" * 30)
        
        if not force_rebuild and dataset_state.get('edge_embedding', {}).get('completed'):
            print("✅ Already completed. Use --force to rebuild.")
        else:
            try:
                import subprocess
                cmd = [
                    sys.executable, 
                    str(PROJECT_ROOT / "index" / "edge_embedding.py"),
                    "--dataset", dataset_name
                ]
                if force_rebuild:
                    cmd.append("--rebuild")
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
                if result.returncode == 0:
                    print("✅ Edge embedding completed")
                    results['edge_embedding'] = str(config.get_edge_index_file())
                else:
                    print(f"❌ Edge embedding failed: {result.stderr}")
                    return None
            except Exception as e:
                print(f"❌ Edge embedding failed: {e}")
                return None
    
    # 4. Answer Generation (Short)
    if 'answer_generation_short' in steps:
        print("\n💬 Step 4: Answer Generation (Short)")
        print("-" * 30)
        
        if not force_rebuild and dataset_state.get('answer_generation_short', {}).get('completed'):
            print("✅ Already completed. Use --force to rebuild.")
            answer_file = config.get_answer_file(answer_type="short")
        else:
            try:
                from generate.answer_generation_short import main as ags_main
                answer_file = ags_main(dataset_name)
                print(f"✅ Answer generation (short) completed: {answer_file}")
                results['answer_generation_short'] = answer_file
            except Exception as e:
                print(f"❌ Answer generation (short) failed: {e}")
                return None
    
    # 5. Answer Generation (Long)
    if 'answer_generation_long' in steps:
        print("\n💬 Step 5: Answer Generation (Long)")
        print("-" * 30)
        
        if not force_rebuild and dataset_state.get('answer_generation_long', {}).get('completed'):
            print("✅ Already completed. Use --force to rebuild.")
            answer_file = config.get_answer_file(answer_type="long")
        else:
            try:
                from generate.answer_generation_long import main as agl_main
                answer_file = agl_main(dataset_name)
                print(f"✅ Answer generation (long) completed: {answer_file}")
                results['answer_generation_long'] = answer_file
            except Exception as e:
                print(f"❌ Answer generation (long) failed: {e}")
                return None
    
    # 6. F1 Evaluation
    if 'evaluation_f1' in steps:
        print("\n📊 Step 6: F1 Evaluation")
        print("-" * 30)
        
        try:
            from evaluate.judge_F1 import main as f1_main
            eval_results = f1_main(dataset_name)
            if eval_results:
                print(f"✅ F1 evaluation completed")
                print(f"   F1 Score: {eval_results['f1_score']:.3f}")
                print(f"   Accuracy: {eval_results['accuracy']:.3f}")
                results['evaluation_f1'] = eval_results
            else:
                print("❌ F1 evaluation failed")
                return None
        except Exception as e:
            print(f"❌ F1 evaluation failed: {e}")
            return None
    
    print("\n" + "=" * 70)
    print("🎉 Pipeline completed successfully!")
    
    # 최종 결과 요약
    final_state = config.load_pipeline_state() or {}
    dataset_final_state = final_state.get(dataset_name, {})
    
    print(f"\n📋 Final Results for {dataset_name}:")
    print("-" * 40)
    
    for step_name, step_info in dataset_final_state.items():
        if step_info.get('completed'):
            print(f"✅ {step_name}")
            if step_name == 'evaluation_f1' and 'f1_score' in step_info:
                print(f"   F1: {step_info['f1_score']:.3f}, Acc: {step_info.get('accuracy', 0):.3f}")
        else:
            print(f"❌ {step_name}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="KGRAG Integrated Pipeline")
    parser.add_argument("--dataset", help="Dataset name to process")
    parser.add_argument("--steps", nargs="+", help="Specific steps to run", 
                       choices=['graph_construction', 'json_to_gexf', 'edge_embedding', 
                               'answer_generation_short', 'answer_generation_long', 'evaluation_f1'])
    parser.add_argument("--force", action="store_true", help="Force rebuild even if completed")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    if args.list_datasets:
        config = get_config()
        available = config.list_available_datasets()
        indexed = config.list_indexed_datasets()
        generated = config.list_generated_datasets()
        
        print("📂 Available datasets (with contexts.txt):")
        for dataset in available:
            status = []
            if dataset in indexed:
                status.append("indexed")
            if dataset in generated:
                status.append("generated")
            status_str = f" ({', '.join(status)})" if status else ""
            print(f"  - {dataset}{status_str}")
        
        if not available:
            print("  No datasets found. Please add contexts.txt files to data/ subdirectories.")
        
        return
    
    # dataset 인수가 필요한 경우 확인
    if not args.dataset:
        parser.error("--dataset is required unless using --list-datasets")
    
    # 데이터셋 존재 확인
    config = get_config(args.dataset)
    input_file = config.get_input_file()
    
    if not input_file.exists():
        print(f"❌ Dataset '{args.dataset}' not found.")
        print(f"   Expected file: {input_file}")
        print("   Use --list-datasets to see available datasets.")
        return
    
    # 파이프라인 실행
    results = run_pipeline(args.dataset, args.steps, args.force)
    
    if results is None:
        print("❌ Pipeline failed!")
        sys.exit(1)
    else:
        print(f"✅ Pipeline completed for {args.dataset}")

if __name__ == "__main__":
    main()
