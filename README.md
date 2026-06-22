# 1. Fine-tuning parameters
| Param    | Value   | 
| -------- | ------- |
| Model arc  |  DeBERTa   |
| Rank | 16     |
| Alpha    | 32    |
| Target modules  |  in_proj, dense   |
| Dropout | 0.05     |
| Batch Size    | 16    |
| Num epochs  |  2   |
| LR | 2e-4     |
| Warmup steps    | 400    |
| Weight decay  |  0.01   |
| Gradient Accumulation steps | 2     |
| Effective Batch Size    | 32    |
