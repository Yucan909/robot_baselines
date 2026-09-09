import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, SiglipTextModel
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

def collect_children(root: dict, result: list, old_id: dict):
    if root.get('name') not in result:
        result.append(root.get('name'))
        old_id[root.get('name')] = root.get('id')
    if "children" in root:
        for child in root["children"]:
            collect_children(child, result, old_id)

def collect_category_and_desc(base_path: str):
    result = dict()
    old_id = dict()
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == "result.json":
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, base_path)
                first_level_folder = relative_path.split(os.sep)[0]
                if int(first_level_folder) > 100000:
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        assert len(data) == 1, f"In {file_path}, Expected one object in JSON file, but got {len(data)}"
                        for obj in data:
                            category= obj.get('name')
                            if category not in result:
                                result[category] = dict()
                                old_id[category] = dict()
                                result[category]['desc'] = []
                                result[category]['count'] = 1
                            else:
                                result[category]['count'] += 1
                            collect_children(obj, result[category]['desc'], old_id[category])
    
    for category, value in result.items():
        print(f"Category: {category}, Count: {value['count']}")
        print(f"Description: {value['desc']}")
        pass
    
    return result, old_id

def generate_category_level_id(categorys: dict):
    category_level_id = {}
    for category, value in categorys.items():
        sorted_desc = sorted(value['desc'])
        category_level_id[category] = {}
        
        for idx, desc in enumerate(sorted_desc):
            category_level_id[category][desc] = idx
        
    return category_level_id

def override_id(mapping: dict, base_path: str):
    folders = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    total_folders = len(folders)
    
    def process_folder(folder):
        folder_path = os.path.join(base_path, folder)
        if int(folder) > 100000:
            return
        
        with open(os.path.join(folder_path, 'result.json'), 'r', encoding='utf-8') as f:
            category = json.load(f)[0].get('name')
        
        label_file_path = os.path.join(folder_path, "point_sample/sample-points-all-label-10000.txt")
        new_label_file_path = os.path.join(folder_path, "point_sample/label-10000-new.txt")
        if os.path.exists(new_label_file_path):
            return
        
        with open(label_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            old_id = line.strip()
            new_id = None
            parts_render_path = os.path.join(folder_path, "parts_render", f"{old_id}.txt")
            if os.path.exists(parts_render_path):
                with open(parts_render_path, 'r', encoding='utf-8') as f:
                    parts_render_lines = f.readline().strip()
                    words = parts_render_lines.split()
                    if len(words) > 1:
                        desc = words[1]
                        new_id = mapping[category][desc]
            else:
                print(f"File not found: {parts_render_path}")
            
            assert new_id is not None, f"Old ID {old_id} not found in mapping for category {category}"
            new_lines.append(f"{new_id}\n")
        
        with open(new_label_file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(process_folder, folder) for folder in folders]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {total_folders} folders", unit="folder"):
            pass

def save_category_embeddings(mapping: dict[str, dict[str, int]], output_dir: str = "."):
    tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
    model = SiglipTextModel.from_pretrained("google/siglip-base-patch16-224")
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for category, desc_to_index in tqdm(mapping.items(), desc="Processing categories"):
        P = len(desc_to_index)
        embeddings = torch.zeros(P, 768)

        descs = [None] * P
        for desc, idx in desc_to_index.items():
            if idx < 0 or idx >= P:
                raise ValueError(f"Invalid index {idx} for category {category}")
            descs[idx] = category + "'s " + desc
        print(f"Category: {category}, descs: {len(descs)}")

        inputs = tokenizer(
            descs,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.pooler_output  # shape: (P, 768)

        out_path = os.path.join(output_dir, f"{category}.npy")
        np.save(out_path, embeddings.cpu().numpy())

    print("All category embeddings saved.")
    

def create_train_test_split(base_path: str, train_ratio: float = 0.75):
    
    train_dir = os.path.join(base_path, '..', 'train')
    test_dir = os.path.join(base_path, '..', 'val')
    
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    # Group folders by category
    category_folders = {}
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path) or int(folder) > 100000:
            continue
        
        result_file = os.path.join(folder_path, 'result.json')
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                category = json.load(f)[0].get('name')
            
            if category not in category_folders:
                category_folders[category] = []
            category_folders[category].append(folder)
            
    print(category_folders['bottle'])
    
    # Split and create symlinks
    for category, folders in tqdm(category_folders.items(), desc="Creating train/test split"):
        folders = sorted(folders)
        split_idx = int(len(folders) * train_ratio)
        
        train_folders = folders[:split_idx]
        test_folders = folders[split_idx:]
        
        for folder in train_folders:
            src = os.path.abspath(os.path.join(base_path, folder))
            dst = os.path.join(train_dir, folder)
            if not os.path.exists(dst):
                os.symlink(src, dst)
        
        for folder in test_folders:
            src = os.path.abspath(os.path.join(base_path, folder))
            dst = os.path.join(test_dir, folder)
            if not os.path.exists(dst):
                os.symlink(src, dst)

if __name__ == "__main__":
    result, old_id = collect_category_and_desc('./partnet/dataset')
    category_level_id = generate_category_level_id(result)
    override_id(category_level_id, './partnet/dataset')
    create_train_test_split('./partnet/dataset')
    # This step is already done, and the generated .npy files are already included in github repo. You can re-run this step if you want to.
    # save_category_embeddings(category_level_id)