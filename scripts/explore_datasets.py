#!/usr/bin/env python3
# Explore the downloaded RAGbench datasets


from datasets import load_from_disk
import json
import os

def explore_dataset(dataset_path, dataset_name):
    """Load and explore a dataset"""
    print(f"\n{'='*60}")
    print(f"🔍 Exploring: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        dataset = load_from_disk(dataset_path)
        
        # Get first split (usually 'train' or 'test')
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        
        print(f"\n📊 Dataset Structure:")
        print(f"  Split: {split_name}")
        print(f"  Total examples: {len(data)}")
        print(f"  Columns: {data.column_names}")
        
        # Show first example
        print(f"\n📝 First Example:")
        first_example = data[0]
        print(json.dumps(first_example, indent=2, default=str))
        
        # Show one more example for context
        if len(data) > 1:
            print(f"\n📝 Second Example:")
            second_example = data[1]
            print(json.dumps(second_example, indent=2, default=str))
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None

def main():
    """Explore all downloaded datasets"""
    
    base_dir = "data/benchmarks"
    
    # List all downloaded datasets
    if not os.path.exists(base_dir):
        print(f"❌ No datasets found in {base_dir}")
        print("Run download_datasets.py first!")
        return
    
    datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]
    
    if not datasets:
        print(f"❌ No datasets found in {base_dir}")
        print("Run download_datasets.py first!")
        return
    
    print("="*60)
    print("🚀 Exploring RAGbench Datasets")
    print("="*60)
    print(f"\nFound {len(datasets)} dataset(s):")
    for ds in datasets:
        print(f"  - {ds}")
    
    # Explore each dataset
    for dataset_name in datasets:
        dataset_path = os.path.join(base_dir, dataset_name)
        explore_dataset(dataset_path, dataset_name)
    
    print("\n" + "="*60)
    print("✨ Exploration Complete!")
    print("="*60)

if __name__ == "__main__":
    main()