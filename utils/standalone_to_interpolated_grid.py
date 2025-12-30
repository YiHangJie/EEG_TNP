from typing import Dict, Tuple, Union
import numpy as np
from scipy.interpolate import RBFInterpolator


class ToInterpolatedGrid:
    r"""
    Fast version of ToInterpolatedGrid using Matrix Multiplication (RBF).

    Projects per-channel EEG signals onto a 2D grid using a pre-computed
    interpolation matrix. This supports extrapolation (filling edges) and
    is significantly faster than iterative methods.

    Args:
        channel_location_dict: Mapping from electrode name to ``(row, col)``.
        apply_to_baseline: When True, applies same transform to baseline.
        interp_method: Kernel function for RBF. Options: 'linear', 'thin_plate_spline', 
                       'cubic', 'gaussian'. Default is 'linear' (robust & smooth).
    """

    def __init__(self,
                 channel_location_dict: Dict[str, Tuple[int, int]],
                 apply_to_baseline: bool = False,
                 interp_method: str = "linear"):
        self.apply_to_baseline = apply_to_baseline
        self.channel_location_dict = channel_location_dict
        
        # 1. 准备电极坐标 (Source Points)
        # shape: (n_channels, 2)
        self.location_array = np.array(list(channel_location_dict.values()))
        n_channels = len(self.location_array)

        # 2. 确定网格大小
        loc_x_list = self.location_array[:, 0]
        loc_y_list = self.location_array[:, 1]
        self.width = int(max(loc_x_list)) + 1
        self.height = int(max(loc_y_list)) + 1

        # 3. 生成网格坐标 (Target Points)
        # 使用 mgrid 生成密集网格，并拉平成 (N_pixels, 2) 的列表
        grid_x, grid_y = np.mgrid[
            min(loc_x_list):max(loc_x_list):self.width * 1j,
            min(loc_y_list):max(loc_y_list):self.height * 1j
        ]
        # 保存下来供可视化或反向映射用
        self.grid_x = grid_x
        self.grid_y = grid_y
        
        # 将网格坐标拉平: (H*W, 2)
        self.grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        # 4. 【核心魔法】预计算投影矩阵
        print(f"Pre-computing interpolation matrix ({interp_method})...")
        
        # 构建一个单位矩阵，模拟每个电极单独激活的情况
        # shape: (n_channels, n_channels)
        eye_signals = np.eye(n_channels)

        # 使用 RBFInterpolator 拟合单位信号
        # y: 观测点坐标 (Electrodes)
        # d: 观测点值 (Identity Matrix)
        rbf = RBFInterpolator(self.location_array, eye_signals, kernel=interp_method)
        
        # 预测网格点的值
        # weights shape: (n_pixels, n_channels) -> (H*W, C)
        self.weight_matrix = rbf(self.grid_points)
        
        # 重塑为 (Grid_H, Grid_W, Channel) 以便后续进行张量乘法
        self.weight_matrix = self.weight_matrix.reshape(self.width, self.height, n_channels)
        print("Pre-computation done.")

    def __call__(self,
                 *,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None) -> Dict[str, np.ndarray]:
        """
        Note: 'method' argument is removed because interpolation kernel is fixed at init.
        """
        if eeg is None:
            raise ValueError("eeg data is required")

        result = {'eeg': self.apply(eeg)}
        if self.apply_to_baseline and baseline is not None:
            result['baseline'] = self.apply(baseline)
        return result

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fast application using Einstein Summation.
        Input: [num_electrodes, num_timesteps]
        Output: [num_timesteps, grid_width, grid_height]
        """
        # eeg shape: (Channel, Time)
        # weight_matrix shape: (Height, Width, Channel)
        
        # 使用 einsum 进行张量乘法:
        # h, w: 网格尺寸
        # c: 电极通道
        # t: 时间步
        # 结果 -> (Time, Height, Width)
        return np.einsum('hwc, ct -> thw', self.weight_matrix, eeg)

    def reverse(self, eeg: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Inverts the grid back to channel order (Sampling).
        Input shape: [num_timesteps, grid_width, grid_height]
        """
        # transpose to (Width, Height, Timestep) to match coordinates
        eeg = np.asarray(eeg).transpose(1, 2, 0)
        num_electrodes = len(self.channel_location_dict)
        num_timesteps = eeg.shape[2]
        
        outputs = np.zeros([num_electrodes, num_timesteps])
        
        # 简单的最近邻采样 (从网格中取回数值)
        # 注意：这里假设 channel_location_dict 的坐标是整数索引
        for i, (x, y) in enumerate(self.channel_location_dict.values()):
            # 确保索引不越界
            x_idx = min(int(x), self.width - 1)
            y_idx = min(int(y), self.height - 1)
            outputs[i] = eeg[x_idx][y_idx]
            
        return {'eeg': outputs}

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(size={self.width}x{self.height}, "
                f"apply_to_baseline={self.apply_to_baseline})")


if __name__ == "__main__":
    # 模拟数据测试
    from torcheeg.datasets.constants.ssvep import TSUBENCHMARK_CHANNEL_LOCATION_DICT 
    import time

    # 1. 初始化 (计算矩阵)
    transform = ToInterpolatedGrid(TSUBENCHMARK_CHANNEL_LOCATION_DICT, interp_method='linear')

    # 2. 生成假数据 (64通道, 2048时间点)
    eeg_data = np.random.randn(64, 2048) 

    # 3. 性能测试
    start_time = time.time()
    out = transform(eeg=eeg_data)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print("Grid eeg shape:", out["eeg"].shape)
    
    # 4. 检查是否有全0 (死点)
    # RBF 插值不应该产生纯 0 (除非信号本身是0)，这里检查是否有大量 0
    zeros = np.sum(out["eeg"] == 0)
    print(f"Number of exact zeros in output: {zeros} (Should be very low or 0)")

    # 5. 反向重构
    reversed_eeg = transform.reverse(out["eeg"])["eeg"]
    print("Reconstructed eeg shape:", reversed_eeg.shape)