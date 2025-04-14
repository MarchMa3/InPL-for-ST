import os
import h5py
import numpy as np
import random
from PIL import Image
import logging
from glob import glob
from tqdm import tqdm
import argparse
from datetime import datetime

def setup_logger(log_dir='../results/log', log_level=logging.INFO):
    """设置日志记录器，将日志同时输出到控制台和文件"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'breast_h5_to_png_{timestamp}.log')
    
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除现有的处理器，避免重复日志
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志配置完成，日志文件保存在: {log_file}")
    
    return logger

def test_breast_h5_to_png(base_dir='./hest_data', train_test_split=0.8, seed=42, max_patches=None):
    """
    测试将breast数据集中的h5文件转换为PNG图像，并分割为训练/测试集
    
    Parameters:
    -----------
    base_dir : str, default='./hest_data'
        包含breast目录的基础目录
    train_test_split : float, default=0.8
        训练集的比例
    seed : int, default=42
        随机种子
    max_patches : int, optional
        每个h5文件最多处理的patches数量，用于测试。如果为None，处理所有patches
    """
    # 固定处理的器官为breast
    organ = 'breast'
    
    # 设置输出目录
    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    data_dir = os.path.join(parent_dir, 'data')
    breast_data_dir = os.path.join(data_dir, organ)
    train_dir = os.path.join(breast_data_dir, 'train')
    test_dir = os.path.join(breast_data_dir, 'test')
    
    # 创建必要的目录
    os.makedirs(breast_data_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    logging.info(f"基础目录: {base_dir}")
    logging.info(f"数据输出目录: {data_dir}")
    logging.info(f"训练目录: {train_dir}")
    logging.info(f"测试目录: {test_dir}")
    
    # 定位breast数据集的patches目录
    breast_dir = os.path.join(base_dir, organ)
    patches_dir = os.path.join(breast_dir, 'patches')
    
    if not os.path.exists(patches_dir):
        logging.error(f"错误: breast/patches目录不存在: {patches_dir}")
        return False
    
    # 查找所有h5文件
    h5_files = [f for f in os.listdir(patches_dir) if f.endswith('.h5')]
    if not h5_files:
        logging.error(f"错误: 在 {patches_dir} 中没有找到h5文件")
        return False
    
    logging.info(f"找到 {len(h5_files)} 个h5文件")
    total_patches = 0
    total_train = 0
    total_test = 0
    
    # 设置随机种子
    random.seed(seed)
    
    # 处理每个h5文件
    for h5_file in tqdm(h5_files, desc="处理h5文件"):
        h5_path = os.path.join(patches_dir, h5_file)
        try:
            with h5py.File(h5_path, 'r') as f:
                # 检查h5文件是否包含img数据集
                if 'img' not in f:
                    logging.warning(f"警告: 文件 {h5_path} 不包含'img'数据集，跳过")
                    continue
                
                # 获取图像数据
                patches = f['img'][...]
                file_patches = len(patches)
                logging.info(f"文件 {h5_file} 包含 {file_patches} 个patches")
                
                # 如果指定了max_patches，则限制处理的patches数量
                if max_patches is not None:
                    patches = patches[:min(file_patches, max_patches)]
                    logging.info(f"限制处理 {len(patches)} 个patches进行测试")
                
                # 随机打乱索引并分割为训练/测试集
                indices = list(range(len(patches)))
                random.shuffle(indices)
                split = int(len(indices) * train_test_split)
                train_idx = indices[:split]
                test_idx = indices[split:]
                
                # 保存训练集图像
                for i, idx in enumerate(train_idx):
                    patch = patches[idx]
                    img = Image.fromarray(patch.astype(np.uint8))
                    save_path = os.path.join(train_dir, f"{h5_file[:-3]}_{i}.png")
                    img.save(save_path)
                    total_train += 1
                
                # 保存测试集图像
                for i, idx in enumerate(test_idx):
                    patch = patches[idx]
                    img = Image.fromarray(patch.astype(np.uint8))
                    save_path = os.path.join(test_dir, f"{h5_file[:-3]}_{i}.png")
                    img.save(save_path)
                    total_test += 1
                
                total_patches += len(patches)
                logging.info(f"已处理 {len(patches)} 个patches (训练: {len(train_idx)}, 测试: {len(test_idx)})")
        
        except Exception as e:
            logging.error(f"处理文件 {h5_path} 时出错: {str(e)}")
    
    # 验证结果
    train_files = glob(os.path.join(train_dir, "*.png"))
    test_files = glob(os.path.join(test_dir, "*.png"))
    
    logging.info(f"\n处理完成:")
    logging.info(f"总共处理 {total_patches} 个patches")
    logging.info(f"训练集: {total_train} 个图像 (验证: {len(train_files)} 个文件)")
    logging.info(f"测试集: {total_test} 个图像 (验证: {len(test_files)} 个文件)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='将breast数据集中的h5文件转换为PNG并分割为训练/测试集')
    parser.add_argument('--base_dir', type=str, default='./hest_data', 
                        help='包含breast目录的基础目录')
    parser.add_argument('--max_patches', type=int, default=None, 
                        help='每个h5文件最多处理的patches数量 (用于测试，默认处理所有patches)')
    parser.add_argument('--log_dir', type=str, default='../results/log', 
                        help='日志文件保存目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                        help='训练集的比例 (0.0-1.0)')
    
    args = parser.parse_args()
    
    # 设置日志记录器
    setup_logger(log_dir=args.log_dir)
    
    logging.info("=== 开始处理breast数据集 ===")
    
    if test_breast_h5_to_png(
        base_dir=args.base_dir,
        train_test_split=args.train_ratio,
        max_patches=args.max_patches
    ):
        logging.info("✅ 处理成功: 已将breast数据集的h5文件转换为PNG并分割为训练/测试集")
    else:
        logging.error("❌ 处理失败")
    
    logging.info("=== 处理完成 ===")

if __name__ == "__main__":
    main()