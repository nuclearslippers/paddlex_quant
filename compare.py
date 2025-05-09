import argparse
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 引用 paddle inference 推理库
import paddle.inference as paddle_infer


def compare_quantization_results(original_output, quantized_output, threshold=1e-6):
    """
    比较量化前后的模型输出结果
    
    参数:
        original_output: 原始模型输出的numpy数组
        quantized_output: 量化后模型输出的numpy数组
        threshold: 判断输出是否相同的阈值
        
    返回:
        包含各种比较指标的字典
    """
    # 确保输入是numpy数组
    original_output = np.asarray(original_output)
    quantized_output = np.asarray(quantized_output)
    
    # 检查形状是否一致
    if original_output.shape != quantized_output.shape:
        raise ValueError(f"形状不匹配! 原始输出形状: {original_output.shape}, 量化输出形状: {quantized_output.shape}")
    
    # 计算各种指标
    abs_diff = np.abs(original_output - quantized_output)
    rel_diff = abs_diff / (np.abs(original_output) + 1e-12)  # 避免除以0
    
    results = {
        'max_absolute_difference': np.max(abs_diff),
        'mean_absolute_difference': np.mean(abs_diff),
        'median_absolute_difference': np.median(abs_diff),
        'max_relative_difference': np.max(rel_diff),
        'mean_relative_difference': np.mean(rel_diff),
        'median_relative_difference': np.median(rel_diff),
        'mse': mean_squared_error(original_output, quantized_output),
        'mae': mean_absolute_error(original_output, quantized_output),
        'cosine_similarity': np.dot(original_output.flatten(), quantized_output.flatten()) / 
                            (np.linalg.norm(original_output.flatten()) * np.linalg.norm(quantized_output.flatten()) + 1e-12),
        'elementwise_match_percentage': np.mean(np.abs(original_output - quantized_output) < threshold) * 100,
        'original_output_stats': {
            'mean': np.mean(original_output),
            'std': np.std(original_output),
            'min': np.min(original_output),
            'max': np.max(original_output)
        },
        'quantized_output_stats': {
            'mean': np.mean(quantized_output),
            'std': np.std(quantized_output),
            'min': np.min(quantized_output),
            'max': np.max(quantized_output)
        }
    }
    
    return results

def print_comparison_results(results):
    """打印比较结果"""
    print("="*50)
    print("模型量化精度对比结果")
    print("="*50)
    
    print("\n绝对差异指标:")
    print(f"- 最大绝对差异: {results['max_absolute_difference']:.6f}")
    print(f"- 平均绝对差异: {results['mean_absolute_difference']:.6f}")
    print(f"- 中位数绝对差异: {results['median_absolute_difference']:.6f}")
    
    print("\n相对差异指标:")
    print(f"- 最大相对差异: {results['max_relative_difference']:.6%}")
    print(f"- 平均相对差异: {results['mean_relative_difference']:.6%}")
    print(f"- 中位数相对差异: {results['median_relative_difference']:.6%}")
    
    print("\n其他指标:")
    print(f"- 均方误差(MSE): {results['mse']:.6f}")
    print(f"- 平均绝对误差(MAE): {results['mae']:.6f}")
    print(f"- 余弦相似度: {results['cosine_similarity']:.6f}")
    print(f"- 元素匹配百分比(阈值={1e-6}): {results['elementwise_match_percentage']:.2f}%")
    
    print("\n原始输出统计:")
    stats = results['original_output_stats']
    print(f"- 均值: {stats['mean']:.6f}, 标准差: {stats['std']:.6f}")
    print(f"- 最小值: {stats['min']:.6f}, 最大值: {stats['max']:.6f}")
    
    print("\n量化输出统计:")
    stats = results['quantized_output_stats']
    print(f"- 均值: {stats['mean']:.6f}, 标准差: {stats['std']:.6f}")
    print(f"- 最小值: {stats['min']:.6f}, 最大值: {stats['max']:.6f}")



def main():
    args = parse_args()

    # 创建 config
    config1 = paddle_infer.Config(args.model_file, args.params_file_8)

    # 根据 config 创建 predictor
    predictor1 = paddle_infer.create_predictor(config1)

    # 获取输入的名称
    input_names1 = predictor1.get_input_names()
    input_handle1 = predictor1.get_input_handle(input_names1[0])
    
    # 设置输入
    fake_input = np.random.randn(args.batch_size, 3, 318, 318).astype("float32")
    input_handle1.reshape([args.batch_size, 3, 318, 318])
    input_handle1.copy_from_cpu(fake_input)
    
    # 运行predictor
    predictor1.run()

    # 获取输出
    output_names1 = predictor1.get_output_names()
    output_handle1 = predictor1.get_output_handle(output_names1[0])
    output_data1 = output_handle1.copy_to_cpu() # numpy.ndarray类型
    
    # float32模型
    if not os.path.exists(args.params_file_32):
        print(f"模型文件 {args.params_file_32} 不存在，请检查路径。")
    else:
        print(f"模型文件 {args.params_file_32} 存在。")
    
    config2 = paddle_infer.Config(args.model_file, args.params_file_32)
    predictor2 = paddle_infer.create_predictor(config2)
    input_names2 = predictor2.get_input_names()
    input_handle2 = predictor2.get_input_handle(input_names2[0])
    input_handle2.reshape([args.batch_size, 3, 318, 318])
    input_handle2.copy_from_cpu(fake_input)
    predictor2.run()
    output_names2 = predictor2.get_output_names()
    output_handle2 = predictor2.get_output_handle(output_names2[0])
    output_data2 = output_handle2.copy_to_cpu() # numpy.ndarray类型

    results = compare_quantization_results(output_data1, output_data2)
    print_comparison_results(results)

    # print("Output data size is {}".format(output_data.size))
    # print("Output data shape is {}".format(output_data.shape))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="model filename", default="./model_infer/int8/inference.pdmodel")
    parser.add_argument("--params_file_8", type=str, help="parameter filename", default="./model_infer/int8/inference.pdiparams")
    parser.add_argument("--params_file_32", type=str, help="parameter filename", default="./model_infer/float32/PP-OCRv4_mobile_det_infer/inference.pdiparams")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    return parser.parse_args()

if __name__ == "__main__":
    main()
