# Unified-Mood-Classification

统一音乐情绪识别框架，面向毕业论文场景设计，联合建模两类任务：

- DEAM 数据集上的 Valence/Arousal 二维连续情绪回归
- MTG-Jamendo 数据集上的多标签情绪标签分类

该项目的核心思想是：将低层、中层和高层音频特征统一到同一隐空间后，通过共享时序编码器学习音乐片段表示，再分别输出连续情绪值与多标签情绪类别，实现一个统一的多任务学习框架。

## 1. 项目目标

传统音乐情绪识别通常只关注单一任务，例如只做情绪回归，或者只做标签分类。本项目尝试将两类任务统一到一个模型中，使模型同时学习：

- 连续情绪维度：Valence、Arousal
- 离散情绪语义：如 calm、happy、sad 等多标签 mood tags

这种设计的动机在于：

- DEAM 提供较强的连续情绪监督信号
- MTG-Jamendo 提供更丰富的语义标签监督
- 共享表示学习有助于提升模型对音乐情绪的整体建模能力

## 2. 模型整体架构

当前代码实现的统一模型由四个部分组成：

1. 多源特征提取
2. 特征投影与融合
3. 共享时序编码器
4. 多任务输出头

整体流程如下：

```text
Audio
	-> 预处理与切片
	-> 多层级特征提取
			 |- Mel-spectrogram   (128维)
			 |- Chroma            (12维)
			 |- Tempogram         (384维)
			 |- MERT layer 11     (1024维)
	-> 特征投影到统一隐空间 D=512
	-> 特征融合
			 |- Early Fusion: Cross-Attention
			 |- Late Fusion: Separate Encoders + Average
	-> Conformer Encoder
	-> 输出头
			 |- DEAM: Valence/Arousal 回归
			 |- MTG: 多标签情绪分类
```

### 2.1 输入特征组成

模型使用四类互补的音频特征：

| 特征 | 维度 | 层级 | 作用 |
|------|------|------|------|
| Mel-spectrogram | 128 | 低层声学特征 | 表征谱能量分布、音色细节 |
| Chroma | 12 | 中层和声特征 | 表征音高类分布与和声结构 |
| Tempogram | 384 | 中层节奏特征 | 表征节拍与节奏变化 |
| MERT layer 11 | 1024 | 高层语义特征 | 提供预训练音乐表示与更强语义信息 |

其中，MERT 是当前系统中最重要的高层表示来源。项目在实现中固定使用 MERT 隐层输出的第 11 层特征作为高层语义表示。

### 2.2 特征投影模块

由于四类特征的维度不同，模型先将它们映射到统一隐空间，默认维度为 512。

- Mel 特征经过两层一维卷积网络进行局部模式提取与通道映射
- Chroma 使用线性层投影到 512 维
- Tempogram 使用线性层投影到 512 维
- MERT 特征使用线性层投影到 512 维

这样设计的好处是：

- 不同模态可以在同一语义空间中进行交互
- 保留 Mel 对局部时频结构敏感的优势
- 降低后续融合与编码模块的设计复杂度

### 2.3 特征融合策略

当前代码支持两种融合策略：早融合和晚融合。

#### Early Fusion

早融合使用 Cross-Attention 完成多模态交互，其实现思路为：

- 将 Mel、Chroma、Tempogram 在通道维拼接
- 将拼接后的低中层特征映射为 Key 和 Value
- 将 MERT 作为 Query
- 执行多头交叉注意力
- 使用残差连接和 LayerNorm 保留 MERT 的高层语义主干信息

该策略可以理解为：让高层语义特征主动“查询”低层和中层的声学、和声与节奏信息，从而生成一个融合后的统一表示。

它的优点是：

- 跨模态交互发生在编码前，信息耦合更强
- MERT 作为主导语义流，能够减少低层特征噪声带来的干扰
- 对统一多任务学习更友好

#### Late Fusion

晚融合策略中，四类特征在投影后保持分离：

- Mel、Chroma、Tempogram、MERT 各自进入独立的 Conformer 编码器
- 获得四路编码后的时序表示
- 对四路输出做逐元素平均
- 将平均结果送入共享任务头

该策略更适合分析不同特征流的独立建模能力，也可以作为与早融合的对比实验方案。

### 2.4 时序编码器

融合后的特征使用 Conformer 编码器进行时序建模。当前实现参数默认如下：

- hidden dimension: 512
- encoder layers: 4
- attention heads: 8
- FFN dimension: 2048
- depthwise convolution kernel size: 31

选择 Conformer 的原因是它同时具备：

- Transformer 的全局依赖建模能力
- 卷积模块对局部时序模式的建模能力

这与音乐情绪识别的特点是匹配的，因为音乐情绪既依赖整体结构，也依赖局部节奏、织体和音色变化。

### 2.5 多任务输出头

编码器输出经过时间维全局平均池化后，进入两个任务头：

#### DEAM 回归头

- 输出 2 维结果，对应 Valence 和 Arousal
- 使用 sigmoid 后再映射到真实标签区间
- 当前代码中输出范围设置为：
	- Valence: [1.6, 8.4]
	- Arousal: [1.6, 8.2]

这种范围约束有助于稳定训练，并减少回归结果超出数据分布的问题。

#### MTG 多标签分类头

- 输出维度为情绪标签数
- 使用线性层输出 logits
- 训练时配合 BCEWithLogitsLoss 进行多标签学习

## 3. 数据预处理流程

当前训练流程默认使用“预计算特征 + 快速加载”的方式，这是本项目在工程实现上的重要优化。

### 3.1 音频预处理

无论是 DEAM 还是 MTG-Jamendo，特征提取前都会先执行以下步骤：

1. 使用 librosa 读取单声道音频
2. 如果采样率不是 24 kHz，则重采样到 24 kHz
3. 从每个音频中取 3 个 5 秒片段
4. 将 3 个片段拼接成一个更长的输入序列

因此，单个样本的有效输入长度约为 15 秒。

在预计算脚本中，为了保证可复现性，片段起始位置不是简单随机，而是使用音频路径的稳定哈希作为随机种子，确保同一音频在不同运行中产生相同切片。

### 3.2 四类特征提取

每个样本都会被提取为如下四种特征并保存：

- MERT 第 11 层隐藏状态
- Log-Mel 频谱
- Chroma CQT 特征
- Tempogram 特征

所有特征都会以 `.pt` 文件形式缓存到磁盘中，默认目录如下：

- DEAM: `data/features/deam`
- MTG-Jamendo: `data/features/mtg/{train,val,test}`

### 3.3 预计算缓存机制

默认训练脚本使用 `dataloader_fast.py` 中的缓存加载逻辑，而不是训练时实时调用 MERT。

这样做的原因是：

- MERT 推理是整个训练流程中最耗时的部分
- 预计算后可以显著缩短每个 epoch 的训练时间
- 更适合在笔记本或个人设备上进行实验迭代

训练前建议先执行：

```bash
python precompute_features.py --dataset all
```

### 3.4 DEAM 数据处理

DEAM 是二维连续情绪回归任务，标签来自静态歌曲级标注：

- Valence
- Arousal

数据划分由 `data/DEAM/deam_split.json` 指定，加载器会：

- 读取 train/val/test 划分
- 过滤掉没有缓存特征的样本
- 按需对子集进行确定性采样
- 可选进行 z-score 标签归一化

当前代码中预设的归一化统计量为：

- valence mean = 5.0, std = 1.5
- arousal mean = 5.0, std = 1.5

### 3.5 MTG-Jamendo 数据处理

MTG-Jamendo 是多标签分类任务。项目中先利用 `generate_mtg_csv.py` 生成整理后的标签文件 `mtg_labels.csv`，处理规则包括：

- 优先使用用户整理好的 train/val/test 文件夹列表
- 删除缺少标签元数据的音频
- 若某个 split 数量不足，则从原始标签池中按相同 split 进行确定性补足
- 最终写入真实存在的音频相对路径

当前生成目标规模为：

- train: 2100
- val: 750
- test: 550

加载阶段会：

- 读取对应 split 的标签
- 检查对应缓存文件是否存在
- 构建多热标签向量
- 在训练集上对 Mel 特征做 SpecAugment

### 3.6 批处理与变长序列对齐

由于不同样本的时序长度可能存在细微差异，`collate_fn` 会在 batch 级别执行：

- 按当前 batch 最大长度进行零填充
- 保留每个样本的原始长度 `lengths`
- 将 `lengths` 传入 Conformer，避免填充部分影响编码

这是当前实现中保证变长序列正确建模的关键细节。

## 4. 训练策略

### 4.1 多任务联合训练

训练阶段采用交替式多任务训练：

- 一个 DEAM batch
- 一个 MTG-Jamendo batch
- 交替进行反向传播与参数更新

这样可以在同一轮训练中持续接收两类监督信号，减少模型偏向单一任务的风险。

### 4.2 损失函数设计

当前实现中，两个任务分别使用：

- DEAM: MSELoss
- MTG-Jamendo: BCEWithLogitsLoss

同时，项目在 `MultitaskLoss` 中引入了基于任务不确定性的可学习加权。对应形式可写为：

$$
\mathcal{L}_{task} = \frac{1}{2} e^{-2\log \sigma} \cdot \mathcal{L}_{raw} + \log \sigma
$$

其中：

- $\mathcal{L}_{raw}$ 为原始任务损失
- $\log \sigma$ 为可学习参数

此外，在训练脚本外层还保留了人工任务权重：

- `deam_weight`
- `mtg_weight`

因此，当前系统的任务平衡同时结合了：

- 基于不确定性的自适应加权
- 人工指定的任务权重

### 4.3 训练稳定性设计

当前代码中还包含以下训练稳定性措施：

- 支持梯度累积 `accumulation_steps`
- 梯度裁剪 `max_norm=1.0`
- 在 MPS 设备上支持自动混合精度
- 支持断点恢复训练
- 支持 `torch.compile` 加速尝试

## 5. 推理与评估指标

本项目对两个任务分别采用不同的评估口径。

### 5.1 DEAM 回归指标

DEAM 任务在验证集和测试集上主要统计以下指标：

- CCC: Concordance Correlation Coefficient
- RMSE: Root Mean Squared Error
- Pearson correlation
- 分维度 CCC
	- Valence CCC
	- Arousal CCC

其中，CCC 是当前实现中最重要的回归一致性指标，因为它同时考虑：

- 预测值与真实值的相关性
- 均值偏差
- 方差一致性

因此，相比单纯使用 MSE 或 Pearson，CCC 更适合连续情绪回归任务。

### 5.2 MTG 多标签分类指标

MTG-Jamendo 任务在验证集和测试集上统计：

- F1-micro
- F1-macro
- Precision-micro
- Precision-macro
- Recall-micro
- Recall-macro
- ROC-AUC-micro
- ROC-AUC-macro
- PR-AUC-micro
- PR-AUC-macro

这些指标覆盖了：

- 标签不平衡场景下的整体性能
- 每类标签的平均表现
- 排序质量与阈值后的分类质量

### 5.3 阈值搜索策略

由于 MTG 是多标签任务，模型输出 logits 会先经过 sigmoid 变成概率，再通过阈值得到二值标签。

当前实现不是固定使用 0.5，而是在验证集上搜索全局阈值：

- 搜索范围：0.01 到 0.99
- 搜索目标：最大化 F1-micro

此外还加入了 top-1 fallback 机制：

- 如果某个样本在当前阈值下没有任何标签被预测为正类
- 则强制将概率最高的那个标签设为正类

这个策略能够避免“全零预测”在稀疏多标签任务中带来的评价偏差，也更符合音乐情绪标签通常至少具有一种主情绪的实际场景。

### 5.4 训练日志与结果记录

项目会将训练、验证和测试阶段的结果记录到：

- TensorBoard 日志目录：`runs/unified_mood_model_{fusion_type}`
- 逐 epoch 指标文件：`metrics.csv`
- checkpoint：
	- `checkpoint_last.pt`
	- `checkpoint_best.pt`

其中 `metrics.csv` 当前记录的字段包括：

- train_loss
- train_deam_ccc
- val_loss
- val_deam_ccc
- val_deam_rmse
- val_deam_pearson
- val_deam_ccc_epoch
- val_deam_ccc_valence
- val_deam_ccc_arousal
- mtg_threshold
- mtg_roc_auc_micro / macro
- mtg_pr_auc_micro / macro
- mtg_f1_micro / macro
- mtg_precision_micro / macro
- mtg_recall_micro / macro

## 6. 当前默认实验流程

### 6.1 生成 MTG 标签文件

```bash
python generate_mtg_csv.py
```

### 6.2 预计算特征

```bash
python precompute_features.py --dataset all
```

### 6.3 启动训练

默认早融合：

```bash
python train.py --fusion_type early --batch_size 8 --epochs 50
```

晚融合对比实验：

```bash
python train.py --fusion_type late --batch_size 8 --epochs 50
```

使用部分数据快速验证：

```bash
python train.py --train_pct 0.1 --subset_seed 42 --epochs 2
```

从断点恢复训练：

```bash
python train.py --resume runs/unified_mood_model_early/checkpoints/checkpoint_last.pt
```

Early 特征贡献消融示例：

```bash
python train.py --fusion_type early --enabled_features mert
python train.py --fusion_type early --enabled_features mert,mel
python train.py --fusion_type early --enabled_features mert,mel,chroma
```

### 6.4 查看训练曲线

```bash
tensorboard --logdir=runs/
```

## 7. 代码结构说明

```text
src/models/
	feature_projection.py   # 多源特征投影到统一隐空间
	feature_fusion.py       # 早融合 / 晚融合策略
	conv_transformer.py     # Conformer 时序编码器
	output_heads.py         # 回归头与多标签分类头
	unified_model.py        # 统一多任务模型封装

dataloader_fast.py        # 预计算特征缓存加载器（默认训练使用）
dataloader.py             # 实时提取特征版本（兼容旧接口）
precompute_features.py    # 特征预计算脚本
generate_mtg_csv.py       # MTG 标签文件生成脚本
train.py                  # 训练、验证、测试主流程
```

