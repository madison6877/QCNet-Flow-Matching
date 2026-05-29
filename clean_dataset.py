import os
import shutil
from pathlib import Path
from tqdm import tqdm

# === 配置路径 ===
# 请确认这是你的数据路径
DATASET_ROOT = Path("/home/yuxinahng/Study/project/QCNet/datasets/motion-forecasting/train/raw")

def scan_and_clean():
    if not DATASET_ROOT.exists():
        print(f"错误: 路径不存在 {DATASET_ROOT}")
        return

    print(f"正在深度扫描目录: {DATASET_ROOT}")
    print("规则: 必须同时包含 .json 地图文件 和 .parquet 轨迹文件")

    scenario_folders = [f for f in DATASET_ROOT.iterdir() if f.is_dir()]
    bad_folders = []
    
    # 用于统计具体缺什么
    missing_map_count = 0
    missing_parquet_count = 0

    for folder in tqdm(scenario_folders, desc="全面体检中"):
        # 1. 检查地图
        map_files = list(folder.glob('log_map_archive_*.json'))
        # 2. 检查轨迹
        parquet_files = list(folder.glob('scenario_*.parquet'))
        
        is_bad = False
        
        if len(map_files) == 0:
            is_bad = True
            missing_map_count += 1
        
        if len(parquet_files) == 0:
            is_bad = True
            missing_parquet_count += 1
            
        if is_bad:
            bad_folders.append(folder)

    print("\n" + "="*40)
    print(f"扫描完成！")
    print(f"总文件夹数: {len(scenario_folders)}")
    print(f"损坏文件夹数: {len(bad_folders)}")
    print(f"  - 缺少地图文件: {missing_map_count} 个")
    print(f"  - 缺少轨迹文件: {missing_parquet_count} 个")
    print("="*40)

    if len(bad_folders) > 0:
        print("\n发现损坏文件夹（示例前5个）:")
        for f in bad_folders[:5]:
            print(f" - {f.name}")
            
        confirm = input(f"\n[操作] 是否删除这 {len(bad_folders)} 个损坏的文件夹？(y/n): ").strip().lower()
        if confirm == 'y':
            print("正在删除...")
            for folder in tqdm(bad_folders, desc="删除中"):
                try:
                    shutil.rmtree(folder)
                except Exception as e:
                    print(f"删除失败 {folder}: {e}")
            print("\n[成功] 所有不完整的数据已清理完毕。")
        else:
            print("\n[取消] 未执行删除。")
    else:
        print("\n[完美] 数据集非常健康，没有发现缺失文件！")

if __name__ == "__main__":
    scan_and_clean()
