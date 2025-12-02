import numpy as np
import scipy.io

# 1. 读取 npy 文件
npy_data = np.load('nep_fit_smooth.npy')

# 2. 转换为 mat 文件
# 注意：savemat 的第二个参数必须是一个字典
# 字典的 Key ('my_variable') 会变成 MATLAB 中的变量名
variable_name = 'NEPpWpersqrtHz'
file_name = 'nep_fit_smooth.mat'

scipy.io.savemat(file_name, {variable_name: npy_data})

print(f"转换完成！请在 MATLAB 中使用 load('{file_name}')")