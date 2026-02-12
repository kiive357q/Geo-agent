import os
import numpy as np
from scipy.signal import resample

# 导入解析器
from dingo_core.dataset.parsers import WaveformParser, AGSParser

class DingoDataset:
    """
    Dingo 数据集类
    负责加载 Dingo 数据并生成训练样本
    """
    
    def __init__(self, root_dir: str):
        """
        初始化 Dingo 数据集
        
        参数:
        - root_dir: 数据根目录
        """
        self.root_dir = root_dir
        self.samples = []  # 用于训练的波形
        self.metadata_registry = []  # 用于存档的参数
        
        # 初始化解析器
        self.waveform_parser = WaveformParser()
        self.ags_parser = AGSParser()
        
        # 扫描数据目录
        self._scan_data_dir()
    
    def _scan_data_dir(self):
        """
        扫描数据目录，构建数据索引
        """
        print(f"开始扫描数据目录: {self.root_dir}")
        waveform_count = 0
        metadata_count = 0
        
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    
                    # F 类文件 (Waveform Source: FILE____*.csv)
                    if file.startswith('FILE____'):
                        print(f"Processing F file: {file}")
                        # 调用 WaveformParser
                        result = self.waveform_parser.parse(file_path)
                        if result['valid']:
                            # 确保信号长度为1024
                            if len(result['signal']) != 1024:
                                result['signal'] = resample(result['signal'], 1024)
                            # 加入训练样本
                            self.samples.append({
                                'signal': result['signal'],
                                'length': result['length'],
                                'dt': result['dt']
                            })
                            waveform_count += 1
                            print(f"Added F file to training: {file}, length={result['length']}, dt={result['dt']}")
                        else:
                            print(f"Skipping invalid F file: {file}")
                    
                    # D 类文件 (Metadata Source: D01_*.csv)
                    elif file.startswith('D'):
                        # 调用 AGSParser
                        result = self.ags_parser.parse(file_path)
                        if result['valid'] and result['records']:
                            # 将结果存入 metadata_registry
                            self.metadata_registry.append({
                                'file_path': file_path,
                                'records': result['records']
                            })
                            metadata_count += len(result['records'])
                            print(f"Parsed Metadata: {len(result['records'])} records from {file}")
        
        # 统计信息
        f_files_count = waveform_count
        non_f_files_count = len(self.metadata_registry)
        total_files = f_files_count + non_f_files_count
        total_pile_foundations = f_files_count  # 每个F文件对应一个桩基
        
        print(f"Initialized Dataset: {waveform_count} waveforms ready for training.")
        print(f"Metadata Registry: {metadata_count} records archived.")
        print(f"Total files processed: {total_files}")
        print(f"F开头文件 (训练样本): {f_files_count}")
        print(f"非F开头文件 (元数据): {non_f_files_count}")
        print(f"Total pile foundations: {total_pile_foundations}")
        print(f"数据库分类统计:")
        print(f"  - F开头的文件: {f_files_count}")
        print(f"  - 非F开头的文件: {non_f_files_count}")
    
    def __len__(self) -> int:
        """
        返回数据集长度
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        获取数据项
        
        参数:
        - idx: 数据索引
        
        返回:
        - 包含训练数据的字典
        """
        item = self.samples[idx]
        return {
            'wave': item['signal'],
            'length': item['length'],
            'dt': item['dt'],
            'label': -1  # 所有样本都是无标签的
        }

if __name__ == "__main__":
    # 测试数据集
    data_dir = r"D:\本科论文\MEF-NSPC-RL\V2.0\MEF-NSPC-RL\3r14qbdhv648b2p83gjqby2fl8\ALL"
    dataset = DingoDataset(data_dir)
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample: wave shape={sample['wave'].shape}, length={sample['length']}, dt={sample['dt']}, label={sample['label']}")
