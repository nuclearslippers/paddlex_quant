本仓库用于对PaddleX的OCR模型进行量化。

## 量化列表：
- PP-OCRv4_mobile_det
- PP-OCRv4_mobile_rec
- PP-OCRv4_server_det
- PP-OCRv4_server_rec
- PP-LCNet_x0_25_textline_ori
- PP-FormulaNet-S
- PP-FormulaNet-L
- PP-LCNet_x1_0_doc_ori
- PP-DocLayout-L
- PP-DocLayout-S
- SLANet_plus

## 量化要求
量化的INT8权重和原始的FP32权重精度diff保持在1%之内。

## 量化方法
使用PaddleSlim的量化工具进行量化。

## 量化脚本
- compare.py：用于比较量化前后的精度