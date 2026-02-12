import numpy as np
import re

class WaveformParser:
    """
    专用于解析 F 开头文件 (FILE____*.csv) 的解析器
    """
    
    def __init__(self):
        pass
    
    def parse(self, file_path):
        """
        解析波形文件
        
        参数:
        - file_path: 文件路径
        
        返回:
        - dict: 包含解析结果的字典
        """
        try:
            # 使用 open().readlines() 读取
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Line 1: 查找 "Total Distance"，提取其后的浮点数作为 length
            length = 15.0  # 默认值
            if len(lines) >= 1:
                first_line = lines[0].strip()
                items = first_line.split(',')
                for i, item in enumerate(items):
                    if 'Total Distance' in item:
                        if i + 1 < len(items):
                            try:
                                length = float(items[i + 1].strip())
                                # 检查长度是否合理（1-50米）
                                if length <= 0 or length > 50:
                                    length = 15.0
                            except ValueError:
                                pass
                        break
            
            # Line 2: 查找 "Div."，提取其后的浮点数作为 dt
            dt = 1.0  # 默认值
            if len(lines) >= 2:
                second_line = lines[1].strip()
                items = second_line.split(',')
                for i, item in enumerate(items):
                    if item.strip() == 'Div.':
                        if i + 1 < len(items):
                            try:
                                dt = float(items[i + 1].strip())
                            except ValueError:
                                pass
                        break
            
            # Line 6+: 将剩余所有行拼接，替换换行符，按逗号分割，提取所有非空数值
            signal_list = []
            if len(lines) >= 6:
                # 将第6行及之后的所有行拼接
                data_str = ''
                for line in lines[5:]:
                    data_str += line.strip()
                # 替换换行符
                data_str = data_str.replace('\n', ',')
                # 按逗号分割，提取所有非空数值
                items = data_str.split(',')
                for item in items:
                    item = item.strip()
                    if item:
                        try:
                            signal_list.append(float(item))
                        except ValueError:
                            continue
            
            # 转换为numpy数组
            if signal_list:
                signal = np.array(signal_list, dtype=np.float32)
                # 归一化: signal = (signal - mean) / (max(abs(signal)) + 1e-6)
                mean = np.mean(signal)
                max_abs = np.max(np.abs(signal))
                signal = (signal - mean) / (max_abs + 1e-6)
            else:
                # 如果没有提取到信号，使用默认值
                signal = np.zeros(1024, dtype=np.float32)
            
            return {
                'signal': signal,
                'length': length,
                'dt': dt,
                'valid': True
            }
        except Exception as e:
            print(f"WaveformParser error: {e}")
            # 返回默认值
            return {
                'signal': np.zeros(1024, dtype=np.float32),
                'length': 15.0,
                'dt': 1.0,
                'valid': False
            }

class AGSParser:
    """
    专用于解析 D 开头文件 (D01_*.csv) 的解析器
    """
    
    def __init__(self):
        pass
    
    def parse(self, file_path):
        """
        解析AGS格式文件
        
        参数:
        - file_path: 文件路径
        
        返回:
        - dict: 包含解析结果的字典
        """
        try:
            # 逐行扫描
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            records = []
            in_loca_table = False
            header_indices = {}
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # 遇到 "TABLE","LOCA" 进入解析模式
                if 'TABLE' in line and 'LOCA' in line:
                    in_loca_table = True
                    continue
                
                if in_loca_table:
                    # 提取表头 indices
                    if 'Hole ID' in line:
                        headers = [h.strip('"') for h in line.split(',')]
                        for j, header in enumerate(headers):
                            header_indices[header] = j
                        continue
                    
                    # 提取数据行
                    if header_indices and line:
                        # 检查是否遇到新的TABLE块
                        if 'TABLE' in line:
                            break
                        
                        items = [item.strip('"') for item in line.split(',')]
                        if len(items) >= max(header_indices.values(), default=0) + 1:
                            record = {}
                            # 提取 Hole ID
                            if 'Hole ID' in header_indices:
                                record['Hole ID'] = items[header_indices['Hole ID']]
                            # 提取 Final depth
                            if 'Final depth' in header_indices:
                                record['Final depth'] = items[header_indices['Final depth']]
                            # 提取 Ground Level
                            if 'Ground Level' in header_indices:
                                record['Ground Level'] = items[header_indices['Ground Level']]
                            if record:
                                records.append(record)
            
            return {
                'records': records,
                'valid': True
            }
        except Exception as e:
            print(f"AGSParser error: {e}")
            return {
                'records': [],
                'valid': False
            }

if __name__ == "__main__":
    # 测试WaveformParser
    waveform_parser = WaveformParser()
    # 测试AGSParser
    ags_parser = AGSParser()