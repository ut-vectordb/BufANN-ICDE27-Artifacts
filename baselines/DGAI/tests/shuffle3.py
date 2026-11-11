import numpy as np
import sys
import os

def read_bin(filename, dtype):
    """读取二进制向量文件，匹配C++的save_bin格式"""
    with open(filename, 'rb') as f:
        # 读取点数和维度（C++中用int存储，对应Python的int32）
        num_points = np.fromfile(f, dtype=np.int32, count=1)[0]
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # 读取数据
        data = np.fromfile(f, dtype=dtype, count=num_points * dim)
        data = data.reshape(num_points, dim)
        
    return data, num_points, dim

def write_bin(filename, data, num_points, dim, offset=0):
    """写入二进制向量文件，严格匹配C++的save_bin实现"""
    with open(filename, 'wb') as f:
        # 移动到偏移位置
        if offset > 0:
            f.seek(offset, 0)
        
        # 写入点数和维度（使用int32，与C++的int对应）
        np.int32(num_points).tofile(f)
        np.int32(dim).tofile(f)
        
        # 写入数据
        data.flatten().tofile(f)
        
        # 计算写入的字节数（与C++保持一致）
        bytes_written = num_points * dim * data.dtype.itemsize + 2 * np.dtype(np.int32).itemsize
        print(f"Writing bin: {filename}")
        print(f"bin: #pts = {num_points}, #dims = {dim}, size = {bytes_written}B")
        print("Finished writing bin.")
        
        return bytes_written

def generate_random_ortho_matrix(dim, seed=42):
    """生成随机正交矩阵"""
    # 设置随机种子确保可重复性
    np.random.seed(seed)
    
    # 生成高斯随机矩阵
    gaussian_matrix = np.random.normal(size=(dim, dim))
    
    # 进行QR分解
    q, r = np.linalg.qr(gaussian_matrix)
    
    # 调整符号以确保唯一性
    sign = np.sign(np.diag(r))
    q = q * sign
    
    return q

def random_ortho_transform(input_path, dtype=np.float32):
    """对向量数据进行随机正交变换"""
    # 读取数据
    data, num_points, dim = read_bin(input_path, dtype)
    print(f"num_points: {num_points}")
    print(f"dim: {dim}")
    
    # 生成随机正交矩阵
    ortho_matrix = generate_random_ortho_matrix(dim)
    
    # 进行正交变换 (数据点 × 正交矩阵)
    # 注意：如果是整数类型，先转换为float进行计算，再转回原类型
    original_dtype = dtype
    if np.issubdtype(original_dtype, np.integer):
        transformed_data = data.astype(np.float32) @ ortho_matrix
        # 转换回原整数类型（根据需要调整舍入方式）
        transformed_data = np.rint(transformed_data).astype(original_dtype)
    else:
        transformed_data = data @ ortho_matrix
        transformed_data = transformed_data.astype(original_dtype)
    
    # 保存变换后的结果
    output_path = f"{input_path}.dim_shuffled2"
    write_bin(output_path, transformed_data, num_points, dim)
    print(f"变换后的数据已保存到: {output_path}")
    
    # 保存正交矩阵到单独的文件
    matrix_output_path = f"{input_path}.shuffled_matrix"
    with open(matrix_output_path, 'wb') as f:
        np.int32(dim).tofile(f)  # 先写入维度（int32）
        ortho_matrix.astype(np.float32).tofile(f)  # 保存矩阵数据
    print(f"正交矩阵已保存到: {matrix_output_path}")

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <uint8/int8/float> <file>")
        return -1
    
    dtype_str = sys.argv[1]
    input_path = sys.argv[2]
    
    # 确定数据类型，与C++版本对应
    if dtype_str == "uint8":
        dtype = np.uint8
    elif dtype_str == "int8":
        dtype = np.int8
    elif dtype_str == "float":
        dtype = np.float32
    else:
        print("Unknown type")
        return -1
    
    # 执行随机正交变换
    random_ortho_transform(input_path, dtype)
    return 0

if __name__ == "__main__":
    main()
