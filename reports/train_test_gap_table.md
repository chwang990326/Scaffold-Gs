# Train-Test PSNR Gap 分析表

## 1. 目的

在已有的主表（PSNR / SSIM / LPIPS）基础上，补充一个 **train-test PSNR gap** 表。核心假设是：

- **Sparse view（1/3 训练视图）** 下，模型更容易过拟合训练集 → train PSNR 虚高，test PSNR 偏低 → gap 大。
- **Ours 的语义专家** 通过聚类正则化和专家共享参数，可有效抑制过拟合 → gap 缩小。
- 如果 Ours 的 gap 系统性小于 Baseline，就提供了"过拟合被缓解"的直接证据，而不是仅靠"sparse view 下 Ours 的 test PSNR 更高"的间接推测。

## 2. 数据来源

| 来源 | 含义 |
|---|---|
| **Train PSNR** | 训练日志 `[ITER 30000] Evaluating train` 行，训练过程中在 5 张训练视图上的评估结果。由于只采 5 张视图，绝对数值不如 test PSNR 可靠，但相对比较（Baseline vs Ours）是有效的。 |
| **Test PSNR** | 训练结束后 `render_sets` 对所有测试视图的完整渲染评估，写入 `results.json`。 |
| **Gap** | `Train PSNR - Test PSNR`，正值越大表示过拟合越严重。 |
| **Δ Test (Ours - Base)** | Ours 相对 Baseline 的测试集提升，正值说明泛化更好。 |

**实验设定：**

- Dataset: Mip-NeRF 360（全部 9 场景）
- Setting: Sparse view（`--train_view_stride 3`，只用 1/3 训练视图）
- Baseline: Scaffold-GS（paper_like 参数，未启用 semantic）
- Ours: Scaffold-GS + SAM/CLIP semantic lifting + 8 semantic clustered color experts（`method_profile=semantic_expert_strong_8_b10`）

## 3. 主表：Sparse View 下 Baseline vs Ours 的 Train-Test PSNR Gap

| Scene | Baseline Train | Baseline Test | Baseline Gap | Ours Train | Ours Test | Ours Gap | ΔGap (↓) | ΔTest (↑) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| bicycle | 27.349 | 22.669 | **+4.680** | 27.269 | 23.400 | **+3.869** | **−0.811** | +0.731 |
| bonsai | 35.547 | 31.859 | **+3.688** | 35.197 | 32.080 | **+3.117** | **−0.571** | +0.221 |
| counter | 34.518 | 27.806 | **+6.712** | 34.081 | 28.259 | **+5.822** | **−0.890** | +0.453 |
| flowers | 24.080 | 18.819 | **+5.261** | 24.103 | 19.389 | **+4.714** | **−0.547** | +0.570 |
| garden | 31.580 | 25.681 | **+5.899** | 30.852 | 26.208 | **+4.644** | **−1.255** | +0.527 |
| kitchen | 34.501 | 30.272 | **+4.229** | 33.626 | 30.486 | **+3.140** | **−1.089** | +0.214 |
| room | 35.853 | 30.266 | **+5.587** | 35.387 | 30.615 | **+4.772** | **−0.815** | +0.348 |
| stump | 32.680 | 23.079 | **+9.601** | 31.933 | 23.589 | **+8.344** | **−1.257** | +0.510 |
| treehill | 26.449 | 20.465 | **+5.984** | 25.857 | 20.873 | **+4.984** | **−1.000** | +0.408 |
| **Mean** | **31.395** | **25.657** | **+5.738** | **30.923** | **26.100** | **+4.823** | **−0.915** | +0.442 |

### 3.1 关键发现

1. **Ours 在所有 9 个场景上都缩小了 train-test gap**，平均缩小 **0.915 dB**，说明语义专家的正则化作用在稀疏视图下高度一致，不是偶发现象。
2. **同时，Ours 在所有 9 个场景上提升了 test PSNR**，平均提升 **0.442 dB**。缩小 gap 的同时提升 test performance，说明不是简单"压低了 train PSNR"，而是真正改善了泛化。
3. Baseline 在 stump 上的 gap 最大（9.601 dB），这符合直觉：stump 是最困难的场景之一，纹理复杂、视图覆盖不完整，过拟合最严重。Ours 在这里 gap 缩小最多（−1.257 dB），说明语义专家在困难场景中收益最大。
4. Baseline 的 train PSNR 普遍高于 Ours，但 test PSNR 全面低于 Ours。这意味着 Baseline 的额外 train 性能实际上是 **过拟合的噪声记忆**，而非有效特征学习。

## 4. 延伸参考：Full View 下 Gap 对比

Full view（所有训练视图参与训练）下的 gap 整体较小，部分场景甚至出现 Train < Test（说明 5 视图采样的噪声较大）。此表仅作为对照参考，不作为主要证据。

| Scene | Base Train | Base Test | Base Gap | Ours Train | Ours Test | Ours Gap |
|---|--:|--:|--:|--:|--:|--:|
| bicycle | 21.697 | 25.215 | −3.518 | 21.283 | 25.305 | −4.022 |
| bonsai | 32.424 | 32.709 | −0.285 | 33.111 | 32.719 | +0.392 |
| counter | 31.546 | 29.301 | +2.245 | 32.028 | 29.848 | +2.180 |
| flowers | 22.497 | 21.303 | +1.194 | 22.321 | 21.463 | +0.858 |
| garden | 27.465 | 27.318 | +0.147 | 27.454 | 27.612 | −0.158 |
| kitchen | 32.902 | 31.788 | +1.114 | 30.975 | 31.497 | −0.522 |
| room | 34.378 | 32.208 | +2.170 | 34.357 | 32.302 | +2.055 |
| stump | 26.080 | 26.641 | −0.561 | 25.493 | 27.159 | −1.666 |
| treehill | 20.993 | 23.144 | −2.151 | 21.042 | 23.463 | −2.421 |
| **Mean** | **27.776** | **27.736** | **+0.040** | **27.563** | **27.930** | **−0.367** |

**Full view 下的解读：**

- 由于训练视图充足，模型过拟合压力很小，gap 本身很小（均值 ~0）。
- 部分场景 train < test，是由于 train PSNR 来自仅 5 张训练视图的子采样，存在采样偏差。
- Ours 在 full view 下的 test PSNR 仍然优于 Baseline（均值 27.930 vs 27.736），说明语义专家的收益不完全是"缓解过拟合"——在数据充足时仍有正收益。

## 5. 结论

Sparse view 下：

- Baseline 的 train-test gap 均值 **5.738 dB** → Ours 缩减至 **4.823 dB**，缩小了 **0.915 dB**（约 16%）。
- 所有 9 场景均观测到 gap 缩小 + test 提升，效果一致性强。
- 结合已有的主表结论（SSIM / LPIPS 同步改善），可以形成完整的论证链条：**语义专家通过聚类正则化抑制了稀疏视图下的过拟合，从而同时提升泛化指标（test PSNR +0.44, SSIM +0.012, LPIPS −0.006）**。

> **注意**：Train PSNR 来自训练日志中的 5-view 子采样评估，而非全训练集评估。因此 absolute gap 数值仅作参考，但 Baseline vs Ours 的 relative 比较（gap 缩小、符号一致）是可靠的。
