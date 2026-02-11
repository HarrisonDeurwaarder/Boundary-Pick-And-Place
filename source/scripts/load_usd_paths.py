import os
from huggingface_hub import snapshot_download
from datasets import load_dataset
from source.utils.file_io import write_list
from source.utils.logger import logging


def main() -> None:
    '''
    Main function
    '''
    ALLOWED_CLASS: str = 'Prop general hand manipulation'
    LOCAL_DATA_DIR = os.path.join(os.getcwd(), 'source', 'data', 'assets')
    # Save data locally
    snapshot_download(
        repo_id='nvidia/PhysicalAI-SimReady-Warehouse-01',
        repo_type='dataset',
        local_dir=LOCAL_DATA_DIR,
        allow_patterns=['*.usd', '*.usdc', '*.usda', '*.csv', '*.parquet'],
    )
    
    data = load_dataset(
        "nvidia/PhysicalAI-SimReady-Warehouse-01", 
        split='train'
    )
    paths: list[str] = data['relative_path']
    classifications: list[str] = data['classification']
    
    # Extract paths for small, handheld items
    filtered_paths = []
    logging.info('Starting filtering...')
    for path, classification in zip(paths, classifications):
        if classification == ALLOWED_CLASS and 'simready' in path:
            # Add file extension
            path = f'{LOCAL_DATA_DIR}/{path if path.endswith((".usd")) else path + ".usd"}'
            # Access physics usd
            path = path.replace('.usd', '.usd')
            path = path.replace('\\', '/')
            filtered_paths.append(path)
        
    logging.info('Completed filtering.')
    write_list(filtered_paths, 'source/data/usd_paths.txt')
        
    
if __name__ == '__main__':
    main()