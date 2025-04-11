import os
import h5py
import numpy as np
import random
from PIL import Image
import logging
from glob import glob
from tqdm import tqdm
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_h5_structure(h5_path):
    """
    测试h5文件的结构
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            keys = list(f.keys())
            logger.info(f"H5文件结构: {keys}")
            
            if 'img' not in keys:
                logger.warning(f"警告: 文件 {h5_path} 不包含'img'数据集")
                return False
            
            img_dataset = f['img']
            logger.info(f"'img'数据集形状: {img_dataset.shape}, 类型: {img_dataset.dtype}")
            return True
    except Exception as e:
        logger.error(f"读取H5文件出错: {str(e)}")
        return False

def load_patch_from_h5_test(base_dir, organ, parent_dir=None, train_test_split=0.8, seed=42, max_patches=10):
    """
    精简版的load_patch_from_h5函数，用于测试
    
    Parameters:
    -----------
    base_dir : str
        基础目录路径 (如 "./hest_data")
    organ : str
        要处理的器官名称 (如 "breast")
    parent_dir : str, optional
        输出数据的父目录。如果为None，将使用base_dir的父目录。
    train_test_split : float, default=0.8
        训练集的比例
    seed : int, default=42
        随机种子
    max_patches : int, default=10
        每个h5文件最多处理的patches数量（用于快速测试）
    """
    if parent_dir is None:
        parent_dir = os.path.dirname(os.path.abspath(base_dir))
    
    data_dir = os.path.join(parent_dir, 'data')
    
    logger.info(f"基础目录: {base_dir}")
    logger.info(f"父目录: {parent_dir}")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"处理器官: {organ}")
    
    # 创建输出目录
    organ_dir = os.path.join(data_dir, organ)
    train_dir = os.path.join(organ_dir, 'train')
    test_dir = os.path.join(organ_dir, 'test')
    
    os.makedirs(organ_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 查找该器官的patches目录
    patches_dir = os.path.join(base_dir, organ, 'patches')
    if not os.path.exists(patches_dir):
        logger.error(f"器官 {organ} 的patches目录不存在: {patches_dir}")
        return False
    
    h5_files = [f for f in os.listdir(patches_dir) if f.endswith('.h5')]
    if not h5_files:
        logger.error(f"在 {patches_dir} 中没有找到h5文件")
        return False
    
    logger.info(f"找到 {len(h5_files)} 个h5文件")
    
    # 为了快速测试，只处理第一个h5文件
    h5_file = h5_files[0]
    h5_path = os.path.join(patches_dir, h5_file)
    
    random.seed(seed)
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'img' not in f:
                logger.error(f"文件 {h5_path} 不包含'img'数据集")
                return False
            
            patches = f['img'][...]
            logger.info(f"文件 {h5_file} 包含 {len(patches)} 个patches")
            
            # 限制处理的patches数量以加快测试速度
            num_patches = min(len(patches), max_patches)
            patches = patches[:num_patches]
            
            indices = list(range(num_patches))
            random.shuffle(indices)
            split = int(num_patches * train_test_split)
            train_idx = indices[:split]
            test_idx = indices[split:]
            
            logger.info(f"处理 {num_patches} 个patches (训练: {len(train_idx)}, 测试: {len(test_idx)})")
            
            for i, idx in enumerate(train_idx):
                patch = patches[idx]
                img = Image.fromarray(patch.astype(np.uint8))
                save_path = os.path.join(train_dir, f"{h5_file[:-3]}_{i}.png")
                img.save(save_path)
            
            for i, idx in enumerate(test_idx):
                patch = patches[idx]
                img = Image.fromarray(patch.astype(np.uint8))
                save_path = os.path.join(test_dir, f"{h5_file[:-3]}_{i}.png")
                img.save(save_path)
            
            logger.info(f"成功保存 {len(train_idx)} 个训练图像和 {len(test_idx)} 个测试图像")
            
            # 验证生成的文件
            train_files = glob(os.path.join(train_dir, f"{h5_file[:-3]}*.png"))
            test_files = glob(os.path.join(test_dir, f"{h5_file[:-3]}*.png"))
            
            logger.info(f"验证: 训练目录中有 {len(train_files)} 个文件，测试目录中有 {len(test_files)} 个文件")
            return True
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='测试从h5文件中提取图像并创建训练/测试数据集')
    parser.add_argument('--base_dir', type=str, default='./hest_data', 
                        help='包含所有器官数据的基础目录')
    parser.add_argument('--organ', type=str, default='breast', 
                        help='要处理的器官名称')
    parser.add_argument('--max_patches', type=int, default=20, 
                        help='每个h5文件最多处理的patches数量')
    parser.add_argument('--parent_dir', type=str, default=None, 
                        help='输出数据的父目录')
    
    args = parser.parse_args()
    
    logger.info("=== 开始测试h5文件到PNG的转换 ===")
    logger.info(f"基础目录: {args.base_dir}")
    logger.info(f"处理器官: {args.organ}")
    
    # 检查器官目录结构
    organ_dir = os.path.join(args.base_dir, args.organ)
    patches_dir = os.path.join(organ_dir, "patches")
    
    if not os.path.exists(organ_dir):
        logger.error(f"错误: 器官目录 {organ_dir} 不存在")
    elif not os.path.exists(patches_dir):
        logger.error(f"错误: patches目录 {patches_dir} 不存在")
    else:
        logger.info(f"目录结构正确: {patches_dir} 存在")
        
        # 检查h5文件
        h5_files = [f for f in os.listdir(patches_dir) if f.endswith('.h5')]
        if h5_files:
            logger.info(f"找到 {len(h5_files)} 个h5文件")
            
            # 测试第一个h5文件结构
            test_h5_structure(os.path.join(patches_dir, h5_files[0]))
            
            # 测试转换功能
            logger.info("\n=== 开始测试将h5转换为PNG并分割为训练/测试集 ===")
            if load_patch_from_h5_test(
                base_dir=args.base_dir,
                organ=args.organ,
                parent_dir=args.parent_dir,
                max_patches=args.max_patches
            ):
                logger.info("✅ 测试成功: 成功将h5文件转换为PNG并分割为训练/测试集")
            else:
                logger.error("❌ 测试失败: 无法完成转换")
        else:
            logger.error(f"错误: 在 {patches_dir} 中没有找到h5文件")
    
    logger.info("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()