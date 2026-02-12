import numpy as np

# 生成测试波形数据
test_signal = np.random.randn(1024)

# 保存为 .npy 文件
np.save('test_waveform.npy', test_signal)
print('Created test waveform file: test_waveform.npy')
print(f'Waveform shape: {test_signal.shape}')
print(f'Waveform min: {test_signal.min():.4f}')
print(f'Waveform max: {test_signal.max():.4f}')
print(f'Waveform mean: {test_signal.mean():.4f}')
