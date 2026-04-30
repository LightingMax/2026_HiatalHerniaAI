import os
import random
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# ============================================================
#              用户参数区域 (请根据您的实际路径修改)
# ============================================================
EXCEL_PATH = "/z_data/jjt/classfication/humanvscomputer.xlsx"  # ← Excel文件路径
IMG_ROOT_DIR = "/z_data/jjt/classfication/完成分级/"         # ← 包含所有图片的顶层目录(包含子文件夹)

BATCH_SIZE = 64
IMG_SIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2025

# 固定随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

# ============================================================
#                  数据增强 & Transform
# ============================================================
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
#             自定义 Dataset (读取Excel + 忽略嵌套文件夹)
# ============================================================
class ExcelImageDataset(Dataset):
    def __init__(self, excel_path, img_root_dir, transform=None, rule=1):
        self.transform = transform
        self.rule = rule
        self.samples = []
        
        # 1. 遍历 img_root_dir 下所有文件，建立 "文件名 -> 绝对路径" 的映射字典
        print(f"Scanning directory {img_root_dir} for images...")
        self.img_paths_exact = {} # 全名匹配 (如 img.jpg)
        self.img_paths_stem = {}  # 去掉后缀匹配 (如 img)
        
        for p in Path(img_root_dir).rglob('*'):
            if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.img_paths_exact[p.name] = str(p)
                self.img_paths_stem[p.stem] = str(p)

        # 2. 读取 Excel 文件
        df = pd.read_excel(excel_path)
        
        # 3. 提取 图片名1~3 和 正确分类1~3
        for i in range(1, 4):
            img_col = f'图片名{i}'
            label_col = f'正确分类{i}'
            
            if img_col in df.columns and label_col in df.columns:
                # 过滤掉空的行
                valid_df = df.dropna(subset=[img_col, label_col])
                for _, row in valid_df.iterrows():
                    img_name = str(row[img_col]).strip()
                    label_val = row[label_col]
                    
                    # 查找图片真实路径（兼容带后缀和不带后缀的Excel写法）
                    img_path = self.img_paths_exact.get(img_name) or self.img_paths_stem.get(img_name)
                    
                    if img_path:
                        mapped_label = self._map_label(label_val, self.rule)
                        if mapped_label is not None:
                            self.samples.append((img_path, mapped_label))
                    else:
                        pass # 如果找不到该图片则跳过

        print(f"Rule {self.rule}: Found {len(self.samples)} matching valid samples.")

    def _map_label(self, label_val, rule):
        """
        根据规则映射 Label 索引
        (神经网络要求标签从0开始。假设原标签是 1, 2, 3, 4)
        """
        try:
            val = int(float(label_val))
        except ValueError:
            return None

        if rule == 1:
            # 规则一：4个类别 (1->0, 2->1, 3->2, 4->3)
            if val in [1, 2, 3, 4]:
                return val - 1
        elif rule == 2:
            # 规则二：2个类别 (1合并为类0，2/3/4合并为类1)
            if val == 1:
                return 0
            elif val in [2, 3, 4]:
                return 1
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ============================================================
#                      统一测试评估函数
# ============================================================
def evaluate_model(ckpt_path, rule, num_classes):
    print(f"\n{'='*50}")
    print(f" 开始评估 - 规则 {rule} | 类别数: {num_classes} | 权重: {os.path.basename(ckpt_path)}")
    print(f"{'='*50}")
    
    # 加载数据集
    dataset = ExcelImageDataset(EXCEL_PATH, IMG_ROOT_DIR, val_transform, rule=rule)
    if len(dataset) == 0:
        print("未找到任何有效数据，请检查 Excel 格式或图片路径！")
        return
        
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 创建模型
    model = timm.create_model("timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k", 
                              pretrained=False, 
                              num_classes=num_classes)
    model = model.to(DEVICE)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 安全加载权重：先尝试 weights_only=True，失败后对可信文件回退
    try:
        state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
    except Exception as safe_err:
        warnings.warn(
            "weights_only=True 加载失败，正在回退到 weights_only=False。"
            f"仅在权重来源可信时可使用。原始错误: {safe_err}"
        )
        state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if isinstance(state_dict, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    if torch.cuda.device_count() == 1 and list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.eval()
    all_preds, all_labels = [], []

    # 测试循环
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Testing Rule {rule}"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    p = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    r = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)

    print(f"\n[ 规则 {rule} 测试结果 ]")
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test F1-score:  {f1:.4f}")
    print(f"Test Precision: {p:.4f}")
    print(f"Test Recall:    {r:.4f}")
    print(f"Test Kappa:     {kappa:.4f}\n")


if __name__ == "__main__":
    # 执行规则一：严格分类测试（假设有 4 个类别：1,2,3,4）
    # 如果您的分类不是4个，请修改这里的 num_classes 参数
    evaluate_model(ckpt_path='/z_data/jjt/classfication/best_model_liekongshan.pth', rule=1, num_classes=4)
    
    # 执行规则二：合并分类测试（2 个类别：1 为一类，2/3/4 为一类）
    evaluate_model(ckpt_path='/z_data/jjt/classfication/best_model_liekongshan_2classes.pth', rule=2, num_classes=2)
