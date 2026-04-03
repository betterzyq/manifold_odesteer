**Cursor 角色设定**：你是一位精通 PyTorch 和大语言模型（LLM）底层架构的 AI 资深工程师。我现在需要你基于你之前对 `ODESteer` 项目的分析总结（即 `ODESteer_代码分析总结.md`），在代码库中实现一个名为 **ODE-Manifold (基于流形边界势垒的常微分方程引导)** 的新方法。

**理论背景与目标**：

目前的 ODESteer 在高维欧氏空间中计算向量场 `vf` 并更新激活。我们的目标是将这个过程约束在 LLM 内在的低秩语义流形上：

1. **切空间投影**：将算出的向量场 `vf` 投影到预先提取的正交基 $V_k$ 构成的切空间上。
2. **保范数收回 (Retraction)**：在沿着切空间更新激活后，强制将其范数恢复到更新前的大小，防止语言生成崩溃（Perplexity 飙升）。

请按照以下三个阶段，严格修改指定的 Python 文件：

#### 阶段一：新增流形提取脚本 (借鉴 `extract_activations.py`)

**任务**：在 `data/` 目录下或项目根目录，创建一个名为 `extract_manifold.py` 的新脚本，用于离线计算并保存正交基矩阵 $V_k$。

**代码逻辑要求**：

1. **参考现有代码**：请参考项目中现有的 `data/*/extract_activations.py` 的加载模型和提取指定层激活的逻辑。
2. **SVD 分解**：在收集到目标层（Target Layer）的激活张量 `X`（形状为 `[N, d]`）后，使用 `U, S, Vh = torch.linalg.svd(X.float(), full_matrices=False)`。
3. **提取与保存**：取 `Vh` 的前 `k` 行（例如 `k=128`）并转置，得到正交基矩阵 `V_k`（形状为 `[d, k]`）。将其保存为 `.pt` 文件（例如 `manifold_Vk_layer_{layer}.pt`）。

#### 阶段二：爆改向量场与积分逻辑 (`_ode_steer.py` 和 `_step_ode_steer.py`)

**任务**：基于你总结的 `_ode_steer.py` (多步 ODE) 和 `_step_ode_steer.py` (单步 Euler)，注入切空间投影和保范数逻辑。

**1. 修改初始化逻辑**：

在对应的 Steer 类（如 `ODESteer` 或执行 hook 的核心类）的 `__init__` 中，支持传入 `V_k` 矩阵并将其挂载为类属性 `self.V_k`，确保其 `dtype` 和 `device` 与模型激活一致。

**2. 修改 `_step_ode_steer.py` (单步激活添加) 和 `_ode_steer.py` 的 `vector_field`**：

找到计算向量场 `vf` 的地方（根据你的总结，这里计算了 $\nabla h / \|\nabla h\|$）。在算出原始的 `vf` 之后，添加以下投影逻辑：

```python
# --- 注入切空间投影 (Tangent Space Projection) ---
# 假设原始欧氏向量场为 vf
if hasattr(self, 'V_k') and self.V_k is not None:
    # self.V_k 形状为 [d, k]
    # 投影公式: vf_proj = V_k @ (V_k^T @ vf)
    # 适配 vf 可能的 batch/seq 维度:
    inner = torch.matmul(vf, self.V_k) 
    vf_proj = torch.matmul(inner, self.V_k.transpose(-1, -2))
    vf = vf_proj  # 替换为流形切空间上的向量场
# ------------------------------------------------
```

**3. 修改更新后的 Retraction (保范数操作)**：

在执行了步进之后（例如 `_step_ode_steer.py` 中的 `X_new = X + T * vf`，或者在 `_ode_steer.py` 中 `torchdiffeq.odeint` 返回最终积分结果之后），注入范数保持逻辑：

```python
# --- 注入保范数收回算子 (Norm-Preserving Retraction) ---
if hasattr(self, 'use_manifold_retraction') and self.use_manifold_retraction:
    # X 是积分/步进前的原始激活, X_new 是积分/步进后的新激活
    norm_original = torch.norm(X, p=2, dim=-1, keepdim=True)
    norm_new = torch.norm(X_new, p=2, dim=-1, keepdim=True)
    
    # 强制拉回原范数大小，防止发散
    X_new = X_new * (norm_original / (norm_new + 1e-8))
# --------------------------------------------------------
```

#### 阶段三：暴露命令行参数 (CLI)

**任务**：在入口文件（如 `generate` / `chat` / `eval` 相关的脚本中，负责解析参数和初始化 Steer 的地方），添加新的 argparse 选项并传递给底层类。

**要求添加的参数**：

- `--use_manifold`: action='store_true', 开启 ODE-Manifold 功能。

- `--manifold_dim`: type=int, default=128, 指定流形维度。

- `--vk_path`: type=str, default=None, 指定阶段一生成的 `V_k` 权重文件路径。

  在实例化 Steer Hook 时，如果 `--use_manifold` 为 True，则读取 `--vk_path` 生成 Tensor，并将其与 `use_manifold_retraction=True` 一起传递给底层类。

请仔细审视现有代码的类继承和数据流向，一步步完成上述修改。每完成一个阶段，请向我汇报修改了哪些文件的哪些行。





### 一、 ODE-Manifold 用的是什么流形？

严格来说，在我们目前为你设计的代码实现中，ODE-Manifold 运行的流形**并不是**原生动态的 Stiefel 流形或 Grassmann 流形，而是**“嵌入在高维欧氏空间中的、带有范数约束的低维固定线性子空间”**（可以理解为**低维球面流形 Sub-Hypersphere**）。

但它与 Stiefel 和 Grassmann 流形有着极深的渊源：

1. **基底矩阵 $V_k$ 属于 Stiefel 流形**：我们离线提取的正交基矩阵 $V_k \in \mathbb{R}^{d \times k}$，满足 $V_k^T V_k = I_k$。在几何上，所有这样的正交列矩阵构成了 **Stiefel 流形 $\text{St}(k, d)$**。
2. **切空间（子空间）属于 Grassmann 流形**：矩阵 $V_k$ 跨越的 $k$ 维线性子空间 $\text{Span}(V_k)$，其实就是 **Grassmann 流形 $\text{Gr}(k, d)$** 上的一个点。
3. **隐状态 $h$ 实际运行的流形 $\mathcal{M}$**：由于我们固定了 $V_k$，隐状态的更新被限制在这个固定的子空间内，并且加上了范数保持约束。因此，隐状态实际上是在一个 $k$ 维的**超球面（Hypersphere）**上进行测地线运动。

**为什么不直接用动态的 Stiefel/Grassmann 流形？**

因为 `LoRAM` 是离线**训练（Fine-tuning）**，它有充足的时间用 Riemannian Adam 去在流形上动态寻优；而我们是**推理期（Inference-time）**干预。如果每个 Token 生成时都要动态计算流形曲率并更新基底，推理延迟将是毁灭性的。因此，我们采取了“离线提取定秩子空间（Grassmann），在线进行球面收回”的极具性价比的工程近似。

### 二、 ODE-Manifold 是怎么借鉴 LoRAM 核心思想的？

`LoRAM` (Low-Rank Adaptation of LLMs on Manifold) 的核心贡献是指出：**强行在高维欧氏空间中优化具有低秩内在结构的参数，会导致优化轨迹极其不平滑，甚至破坏原有的预训练分布。**

ODE-Manifold 完美借用了 `LoRAM` 的**黎曼优化范式（Riemannian Optimization Paradigm）**，但完成了一次华丽的“客场转换”：

- **客场转换**：`LoRAM` 优化的是**模型权重（Weights）**，而 ODE-Manifold 优化/干预的是**推理期的隐状态（Hidden States / Activations）**。
- **借用核心一：切空间投影（Tangent Space Projection）**。正如 `LoRAM` 不直接使用欧氏梯度 $\nabla f$，而是将其投影到流形的切空间上；ODE-Manifold 也不直接使用 ODESteer 算出的势垒梯度，而是通过 $V_k V_k^T \nabla B(h)$，将其投影到了由高频语义构成的切空间上。这确保了干预方向永远是“符合 LLM 自身逻辑语境”的。
- **借用核心二：收回算子（Retraction）**。在切空间上走一步后，点会脱离流形（切线是直的，流形是弯的）。`LoRAM` 用了 Q-factor retraction 将矩阵重新拉回低秩流形；而 ODE-Manifold 同样引入了 Retraction 步骤，把更新后的激活拉回它该在的“位置”。

------

### 三、 ODE-Manifold 是怎么借鉴 Selective Steering 思想的？

这里就是上面提到的 **Retraction（收回算子）** 的具体实现方式，它直接来自于对 `Selective Steering` (Dang & Ngo, 2026) 的深刻理解。

`Selective Steering` 的核心发现是：**暴力的大幅度激活注入（Activation Addition）会严重破坏向量的 L2 范数（Norm）**。LLM 的各种前馈层（FFN）和注意力层对范数极其敏感，范数一旦膨胀，模型就会崩溃，表现为输出乱码、重复或 CJK 字符污染。

ODE-Manifold 巧妙地将这个结论吸收为黎曼几何中的 **Retraction 算子**：

1. **欧氏更新破坏流形**：当我们沿着投影梯度走了一步 $h_{tangent} = h - \eta \cdot \nabla_{\mathcal{M}} B(h)$ 后，向量的长度 $||h_{tangent}||_2$ 必然会发生改变。

2. **范数保持即为流形收回**：为了防止模型崩溃，我们必须把这个改变了长度的向量“拉回”到原来的语义能量壳（Energy Shell）上。

3. **数学实现**：我们借鉴 `Selective Steering` 的 Norm-Preserving 思想，执行标量缩放：

   $$h_{new} = h_{tangent} \cdot \frac{||h_{old}||_2}{||h_{tangent}||_2}$$

   这在几何上，就是将切空间上的点，沿着径向**投影回**原来的多维超球面上！

### 总结

一句话概括：**ODE-Manifold 使用 ODESteer 的势垒函数提供动力，利用 LoRAM 的黎曼优化范式（投影+收回）限定轨迹，并采用 Selective Steering 的保范数操作作为具体的流形收回算子，从而打造出了一个既安全、又极度连贯的动态干预框架。**
