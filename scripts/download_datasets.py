from datasets import load_dataset
import os
from pathlib import Path

def download_dataset(dataset_name, config_name, save_dir):
    """
    Download a dataset configuration from Hugging Face and save it locally
    
    Args:
        dataset_name: Full dataset name (e.g., 'G4KMU/t2-ragbench')
        config_name: Specific config/subset (e.g., 'FinQA')
        save_dir: Directory to save the dataset
    """
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_name} - {config_name}")
    print(f"{'='*60}")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset(dataset_name, config_name)
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save to disk
        dataset.save_to_disk(save_dir)
        
        print(f"Successfully downloaded to: {save_dir}")
        
        # Show dataset info
        print(f"\nDataset Info:")
        print(f"  Splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} examples")
            if len(split_data) > 0:
                print(f"  Columns: {split_data.column_names}")
        
        return dataset
        
    except Exception as e:
        print(f"Error downloading {dataset_name}/{config_name}: {e}")
        return None

def main():
    """Download selected RAGbench datasets"""
    
    # Base directory for datasets
    base_dir = "data/benchmarks"
    
    print("="*60)
    print("RAGbench Dataset Download")
    print("="*60)
    
    t2_configs = ['FinQA', 'ConvFinQA', 'TAT-DQA']
    
    for config in t2_configs:
        download_dataset(
            dataset_name="G4KMU/t2-ragbench",
            config_name=config,
            save_dir=os.path.join(base_dir, f"t2-ragbench-{config}")
        )
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"\nDatasets saved to: {base_dir}/")
    print("\nDownloaded configs:")
    print("  t2-ragbench: FinQA, ConvFinQA, TAT-DQA")

if __name__ == "__main__":
    main()