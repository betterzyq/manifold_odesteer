# CRFDSteer 项目代码分析总结

本文档结合论文思想总结（`the core idea of the odesteer.md`）对 CRFDSteer 开源代码进行结构梳理与模块映射，便于后续“爆改”与扩展。

---

## 一、论文核心与代码映射概览

| 论文概念 | 代码位置 | 说明 |
|----------|----------|------|
| 激活添加 = ODE 欧拉单步 | `_step_ode_steer.py` | 单步版：`X + T * vf` |
| 多步 ODE 积分 | `_ode_steer.py` + `torchdiffeq.odeint` | 多步自适应求解 |
| 障碍函数 \(h(a) = w^\top \phi(a) + b\) | `_kernel_clf.py`（KernelClassifier） | \(\phi\) 为 kernel，\(w\)=coef，\(b\)=intercept+log(N_-/N_+) |
| 向量场 \(v(a) = \nabla h / \|\nabla h\|\) | `_ode_steer.py` 的 `vector_field` | grad → L2 归一化 |
| \(\phi(a)\) 多项式计数草图 | `_poly_cnt_sketch.py`（PolyCntSketch / NormedPolyCntSketch） | 前向 + 雅可比 / VJP |
| 输入 \(L_2\) 归一化 | `_poly_cnt_sketch.py` 的 `NormedPolyCntSketch._normalize` | 进入 \(\phi\) 前先单位化 |
| 逻辑回归得到 \(w,b\) | `_kernel_clf.py` 的 `_fit_linear_clf` | sklearn LogisticRegression，再构造 \(b\) |

---

## 二、项目结构总览

```
odesteer/
├── confs/                    # Hydra 配置
│   ├── steer/                # 各 Steer 方法配置（CRFDSteer.yaml 等）
│   ├── truthfulqa.yaml
│   ├── ultrafeedback.yaml
│   └── detox.yaml
├── data/                     # 数据与预处理脚本
│   ├── truthfulqa/           # TruthfulQA 激活提取、格式化
│   ├── ultrafeedback/
│   └── toxicity/
├── scripts/                  # 训练/生成/评估入口
│   ├── truthfulqa/
│   ├── ultrafeedback/
│   └── toxicity/
├── src/odesteer/
│   ├── lm/                   # 语言模型封装与 Hook
│   │   ├── _config.py
│   │   ├── _huggingface_lm.py
│   │   └── _hub_local.py     # Hub 缓存路径、repo id → 本地 snapshot、hf_pretrained_hub_kwargs
│   ├── steer/                # 各类 Steer 实现（核心）
│   │   ├── _base_steer.py
│   │   ├── _ode_steer.py     # 多步 ODE
│   │   ├── _step_ode_steer.py # 单步 ODE
│   │   ├── _iti.py, _caa.py, _repe.py, _mimic.py, _lin_act.py
│   │   └── __init__.py       # get_steer_model()
│   └── utils/
│       ├── data.py           # 加载 TQA/UltraFeedback/Toxicity 数据
│       ├── env.py            # load_project_dotenv：加载项目根 .env
│       ├── metric.py         # 评估指标、Judge、Perspective、Quality（含 HF 加载）
│       └── kernels/          # 特征映射与分类器
│           ├── _poly_cnt_sketch.py  # 多项式计数草图 φ 与 J_φ
│           ├── _kernel_clf.py      # 核分类器（sklearn LR + w,b）
│           ├── _rff.py             # RFF 核（消融）
│           └── __init__.py
├── pyproject.toml
└── the core idea of the odesteer.md
```

**主要调用链（TruthfulQA 生成为例）**  
`truthfulqa_generate.py` → `HuggingFaceLM` → `get_steer_model(steer_name)(**kwargs)` 得到 Steer 实例 → `fit_steer_model(pos_train, neg_train)` → 前向时 `register_steer_hook` → 在目标层取隐藏状态 → `steer_model.steer(hidden, T=T)` → 多步 ODE 或单步更新。

---

## 三、四大模块与论文的逐一映射

### 1. 数据预处理与 \(L_2\) 归一化

- **论文**：在送入 \(\phi\) 前，将激活向量 \(a\) 做**单位 \(L_2\) 范数**归一化，以提高稳定性和可扩展性。
- **代码**：**没有**在“数据加载/保存”阶段对激活做 \(L_2\) 归一化；归一化被内置于**非线性特征模块**中。
  - **位置**：`src/odesteer/utils/kernels/_poly_cnt_sketch.py`，类 **`NormedPolyCntSketch`**。
  - **实现**：
    - `_normalize(self, X)`（约 351–357 行）：  
      `r = X.norm(p=2, dim=-1, keepdim=True) + self.eps`，  
      `return X / r, r`  
      即 \(\hat{x} = x / (\|x\|_2 + \epsilon)\)，与论文“单位 \(L_2\) 范数”一致（\(\epsilon\) 防除零）。
    - 在 **forward**（364–366 行）、**grad**（378 行）、**vjp**（412 行）中，都是先对输入做 `_normalize`，再调用父类 `PolyCntSketch` 的对应方法，因此**所有经过 NormedPolyCntSketch 的激活在进入 \(\phi\) 前都会被 \(L_2\) 归一化**。
  - **数据管线**：  
    `data/truthfulqa/extract_activations.py` 只做“最后一词位置隐藏状态”的提取与保存，不做归一化；  
    `utils/data.py` 的 `load_tqa_gen_data` 等仅做 `torch.load`。  
  因此，**“对激活的 \(L_2\) 归一化”完全由 `NormedPolyCntSketch` 在特征层完成，而不是单独的数据预处理脚本**。

---

### 2. 非线性特征提取模块：PolynomialCountSketch（\(\phi\) 与 \(J_\phi\)）

- **论文**：使用**多项式计数草图（Polynomial Count Sketch）**得到 \(\phi(a)\)，并需要 \(\phi\) 对 \(a\) 的雅可比 \(J_\phi(a)\) 以计算 \(\nabla_a h = J_\phi(a)^\top w\)。
- **代码**：  
  - **基类**：`_poly_cnt_sketch.py` 中的 **`PolyCntSketch`**（无归一化）。  
  - **论文用版本**：**`NormedPolyCntSketch`**（继承 `PolyCntSketch`），在入口处对输入做 \(L_2\) 归一化后再做多项式草图。

#### 2.1 前向映射 \(\phi(a)\)

- **入口**：`forward(X)` → 若为 batch 则 `_forward_batch(X)`。
- **步骤概要**：
  1. 若使用 `NormedPolyCntSketch`：先 `x_hat, _ = self._normalize(X)`，再对 `x_hat` 做父类前向。
  2. 父类中：对输入乘 \(\sqrt{\gamma}\)，可选地拼接常数项 \(\sqrt{\mathrm{coef0}}\) 得到 \(X_{\mathrm{ext}}\)。
  3. 用 `indexHash_`、`bitHash_` 做 **D 个 Count Sketch**（D=degree），得到 D 个长度为 `n_components` 的草图向量。
  4. 对 D 个草图做 **TensorSketch**：在频域做逐元素积再 `irfft` 回时域，得到 \(\phi(X) \in \mathbb{R}^{B \times n}\)（或 \(\mathbb{R}^n\)）。
- **关键 API**：`forward` / `transform` / `fit_transform`；`fit(X)` 只初始化哈希表（indexHash_、bitHash_），不改变输入数值。

#### 2.2 雅可比 \(J_\phi(a)\) 与 VJP

- **完整雅可比**（用于理解/调试）：  
  **`grad(X)`**（120–135 行 单样本，136–201 行 实现）。  
  - 对 `NormedPolyCntSketch`：先 `x_hat, r = self._normalize(X)`，再 `J_f = super().grad(x_hat)` 得到 \(d(\phi)/d\hat{x}\)。  
  - 链式法则：\(J = J_f \cdot \frac{\partial \hat{x}}{\partial x}\)，其中 \(\hat{x} = x/\|x\|\) 的雅可比为 \((I - \hat{x}\hat{x}^\top)/\|x\|\)，代码用等价形式  
    `J = (J_f - (J_f @ x_hat) * x_hat) / r`（单样本与 batch 分别处理）。
  - 返回形状：单样本 `[n_components, n_features]`，batch `[B, n_components, n_features]`，即 \(J_\phi(a)\)。
- **向量–雅可比积（实际用于梯度）**：  
  **`vjp(X, v)`**（205–223 行 接口，224–276 单样本，278–331 batch）。  
  - 计算 \(J_\phi(X)^\top v\)，即论文中的 \(J_\phi(a)^\top w\) 当 \(v=w\) 时。
  - `NormedPolyCntSketch` 中（398–423 行）：先归一化得到 `x_hat, r`，再 `u = super().vjp(x_hat, v)`，最后左乘 \(d\hat{x}/dx\) 的转置：  
    `(u - x_hat * (x_hat * u).sum()) / r`，与链式法则一致。

**小结**：  
- \(\phi(a)\)：`NormedPolyCntSketch.forward(a)`（内部含 \(L_2\) 归一化）。  
- \(J_\phi(a)^\top w\)：`KernelClassifier.grad(X)` 内部调用 `self.kernel.vjp(X, self.coef)`（再乘 `dre_coeff`，见下），即论文的 \(\nabla_a h\) 方向（归一化前）。

---

### 3. 分类器训练模块：\(w\) 与 \(b\) 的构造

- **论文**：用逻辑回归在 \(\phi(a)\) 上拟合正/负样本，得到 \(w'\)、\(b'\)；障碍函数为 \(h(a) = w^\top \phi(a) + b\)，其中 \(w = w'\)，\(b = b' + \log(N_-/N_+)\)。
- **代码**：`src/odesteer/utils/kernels/_kernel_clf.py` 中的 **`KernelClassifier`**（CRFDSteer 使用子类 **`NormedPolyClassifier`**，其 `kernel` 为 `NormedPolyCntSketch`）。

#### 3.1 调用 sklearn 逻辑回归

- **位置**：`_fit_linear_clf(self, X, y)`（86–97 行）。  
  - `X` 为已在 kernel 上变换后的特征（即 \(\phi\) 的输出），`y` 为 0/1 标签。  
  - `clf = LogisticRegression(max_iter=1000)`（或 `LinearSVC`，由 `lin_clf_type` 决定），然后 `clf.fit(X, y)`。  
  - 取 `coef_`、`intercept_` 并转为 Tensor，存为 `self.coef`、`self.intercept`（并 `register_buffer`），并设 `self.fitted = True`。  
- **调用链**：  
  `fit(pos_X, neg_X)` → `_fit_with_two_sets` 或 `_fit_with_labels` → 将正负样本拼成 `X`、`y`，计算 `pi = N_+/(N_++N_-)`，设 `self.dre_coeff = (1-pi)/pi = N_-/N_+` → `Z = self.kernel.fit_transform(X)`（即 \(\phi\)）→ `_fit_linear_clf(Z, y)` 得到 \(w'=\texttt{coef}\)，\(b'=\texttt{intercept}\)。

#### 3.2 \(w\) 和 \(b\) 在公式中的对应

- **\(w\)**：即 `self.coef`，形状 `[n_components]`，对应论文的 \(w'\)。  
- **\(b\)**：论文中 \(b = b' + \log(N_-/N_+)\)。  
  - 代码中 `intercept` 存的是 \(b'\)；`dre_coeff = N_-/N_+`。  
  - **log 密度比**在 **`log_dre(X)`**（54–57 行）中实现：  
    `Z @ self.coef + self.intercept + np.log(self.dre_coeff)`  
    即 \(w^\top \phi(a) + b' + \log(N_-/N_+)\)，也就是 \(h(a)\)。  
- **梯度**：`grad(X)`（59–62 行）为  
  `self.kernel.vjp(X, self.coef) * self.dre_coeff`。  
  即 \(\nabla_a h\) 的方向与 \(J_\phi(a)^\top w\) 一致（多了一个常数倍 `dre_coeff`）；在 `vector_field` 中会再做 L2 归一化，因此**方向与论文一致**。

**小结**：  
- 逻辑回归训练：`KernelClassifier._fit_linear_clf` 内调用 `sklearn.linear_model.LogisticRegression`。  
- \(w\) = `self.coef`，\(b\) = `self.intercept + log(self.dre_coeff)`，其中 `dre_coeff = N_-/N_+`；  
  \(h(a)\) = `log_dre(X)`，\(\nabla_a h\) 方向由 `grad(X)` 给出（再在 steer 中 L2 归一化）。

---

### 4. ODE 求解器模块：多步积分循环

- **论文**：\(\dot{a}(t) = v(a(t)) = \frac{J_\phi(a)^\top w}{\|J_\phi(a)^\top w\|}\)，在 \([0,T]\) 上数值积分得到 \(\tilde{a} = a(T)\)；多步、自适应。
- **代码**：  
  - **多步 ODE**：`src/odesteer/steer/_ode_steer.py`。  
  - **单步（欧拉一步）**：`src/odesteer/steer/_step_ode_steer.py`。

#### 4.1 多步 ODE（`_ode_steer.py`）

- **`BaseCRFDSteer.steer(X, T)`**（33–41 行）：  
  - 若 `T==0` 直接返回 `X`。  
  - 否则调用 **`torchdiffeq.odeint`**：  
    `odeint(func=lambda t, state: self.vector_field(state), y0=X, t=torch.tensor([0., T], ...), method=self.solver, options={'step_size': T/self.steps})`  
    取结果最后一时刻 `[1]` 作为 \(a(T)\)。  
- **`vector_field(X)`**（43–46 行）：  
  1. `raw_grad = self.clf.grad(X)` → 即 \(J_\phi(a)^\top w \cdot \mathrm{dre\_coeff}\)（见上）；  
  2. `return raw_grad / (raw_grad.norm(dim=-1, keepdim=True) + 1e-10)` → L2 归一化，得到 \(v(a)\)。  
- **“微小步长循环”**：**不在本仓库手写**，而在 **`torchdiffeq.odeint`** 内部。根据 `method`（如 `euler`）和 `step_size: T/steps`，库会在 \([0,T]\) 上做多步迭代；每一步等价于：  
  - 当前状态 \(a\) → 计算 \(v(a) = \mathrm{vector\_field}(a)\)（内部：雅可比相关梯度 → 乘 \(w\) → L2 归一化）→ 用欧拉/中点/rk4 等更新 \(a\)。  
  因此论文中的“计算雅可比 → 乘 \(w\) → 梯度 L2 归一化 → 更新激活”在代码中对应：**每次 ODE 步内**调用 `vector_field(state)`，而 `vector_field` 内部是 `clf.grad`（含 kernel 的 vjp，即 \(J^\top w\)）再 L2 归一化。

#### 4.2 单步 ODE（`_step_ode_steer.py`）

- **`BaseStepCRFDSteer.steer(X, T)`**（23–25 行）：  
  `vf = self.vector_field(X)`，然后 `return X + T * vf`。  
- **`vector_field`** 与多步版一致：`clf.grad(X)` 再 L2 归一化。  
  即论文中的“欧拉法一步”\(a + T\cdot v(a)\)。

#### 4.3 配置与步长

- **CRFDSteer.yaml**：`solver: euler`，`steps: 10`，即 `odeint` 的 `step_size = T/10`。  
- 若要“爆改”步长或步数，可改配置中的 `steps` 或传入的 `T`，或改用 `midpoint`/`rk4` 等。

**小结**：  
- 多步积分循环在 **`torchdiffeq.odeint`** 内；  
- 每步内：**计算梯度（含 \(J_\phi^\top w\)）→ L2 归一化 → 更新状态**，由 **`vector_field`** 实现“梯度 + 归一化”，由 **`odeint`** 实现“循环 + 积分”。

---

## 四、文件级职责与调用关系简表

| 文件 | 职责 | 被谁调用 |
|------|------|----------|
| `steer/_base_steer.py` | 抽象基类 `Steer`、`VecSteer`（单向量加法） | 所有 steer 实现 |
| `steer/_ode_steer.py` | 多步 ODE（odeint + NormedPolyClassifier） | `get_steer_model("CRFDSteer")`，LM 的 steer hook |
| `steer/_step_ode_steer.py` | 单步 ODE（X + T*v） | `get_steer_model("StepCRFDSteer")` |
| `utils/kernels/_poly_cnt_sketch.py` | \(\phi\)、\(J_\phi\)、VJP；Normed 版含 L2 归一化 | `_kernel_clf.py` 的 NormedPolyClassifier |
| `utils/kernels/_kernel_clf.py` | LR 拟合、coef/intercept/dre_coeff、grad/log_dre | `_ode_steer` / `_step_ode_steer` 的 `self.clf` |
| `utils/kernels/_rff.py` | RFF 核与 jacobian/vjp（消融用） | RFFClassifier / RFFCRFDSteer |
| `utils/data.py` | 加载 TQA/UF/毒性 激活与题目 | 各脚本的 load_*、DataLoader |
| `lm/_huggingface_lm.py` | 模型加载、生成、激活提取、steer hook 注册；导入前 `load_project_dotenv` + 默认 `HF_ENDPOINT`；`get_hub_cache_dir_str` 作 `cache_dir` | 各 scripts 的 generate/chat/eval |
| `lm/_hub_local.py` | `get_hub_cache_dir_str`、`maybe_resolve_pretrained_to_local_path`、`resolve_hub_snapshot_dir`、`hf_pretrained_hub_kwargs`（`local_files_only`、离线） | `_huggingface_lm`、`metric.py`、`ultrafeedback_eval.py` |
| `utils/env.py` | `load_project_dotenv`：解析项目根 `.env` 键值到 `os.environ` | `_huggingface_lm`、`_hub_local`、`metric.ToxicityEvaluator` |
| `data/*/extract_activations.py` | 从模型提取指定层、最后一词激活并保存 | 命令行/流水线预处理 |

---

## 五、环境与 Hugging Face Hub（本仓库增补）

本节描述与**论文算法无关**、但影响**可复现部署与离线运行**的工程层逻辑。

### 5.1 项目根 `.env` 与 `load_project_dotenv`

- **文件**：`src/odesteer/utils/env.py` 中 **`load_project_dotenv(project_dir, override=False)`**。  
- **行为**：读取 `<project_dir>/.env`，将 `KEY=value` 写入 `os.environ`；`override=False` 时不覆盖已存在的环境变量。  
- **调用点**：`hf_pretrained_hub_kwargs` 内；`_huggingface_lm.py` 模块顶层（在 `import transformers` 之前）；`ToxicityEvaluator.__init__` 中再次确保加载；**`scripts/ultrafeedback/ultrafeedback_eval.py`** 在 **import transformers 之前**显式调用，与 LM 一致，避免 RM 加载时未读到 `HF_HOME`。

### 5.2 `get_hub_cache_dir_str` 与本地 snapshot 解析

- **文件**：`src/odesteer/lm/_hub_local.py`。  
- **`get_hub_cache_dir_str()`**：返回 Hugging Face **`hub` 根目录**（其下为 `models--org--repo`）。优先级：`HF_HUB_CACHE` → `HF_HOME/hub` → 若均未设置，则探测 **`/root/autodl-tmp/.cache/huggingface/hub`**、**`~/.cache/huggingface/hub`**（目录存在即用）。  
- **`maybe_resolve_pretrained_to_local_path(repo_id, hub_cache)`**：在 `hub_cache` 下将 **`Skywork/xxx`** 等形式解析为 **`snapshots/<revision>`** 目录（含 `config.json` 与完整权重检查 `snapshot_has_complete_weights`）；失败则仍返回原 repo id。  
- **`hf_pretrained_hub_kwargs(model_id)`**：合并上述逻辑，返回 `(pretrained_path_or_id, {local_files_only, cache_dir})`；若已解析到本地目录或设置了 `HF_HUB_OFFLINE` / `CRFDSteer_OFFLINE` / `CRFDSteer_LOCAL_ONLY`，则 **`local_files_only=True`**，避免无网环境下去连 `huggingface.co`。

### 5.3 与评估脚本的差异（Ultrafeedback RM）

- **TruthfulQA / Toxicity 评估**：主要依赖 `metric.py` 中的 Judge、GPT2-XL、Perspective 等，均通过 **`hf_pretrained_hub_kwargs`** 或等价的缓存逻辑加载。  
- **Ultrafeedback 评估**：额外加载 **序列分类 RM**（默认 Skywork reward）。必须在 **Hub 缓存中有该模型** 或传入 **`-r` 本地路径**；仅下载了主模型而未下载 RM 时，会出现“本地无缓存且无法联网”类错误。  
- **调试**：设置环境变量 **`CRFDSteer_DEBUG_HUB=1`** 可打印 repo id → 本地 snapshot 的解析结果（`_hub_local` 内）。

---

## 六、后续“爆改”可着力的点

1. **数据预处理**：若希望**在存储前**就做 \(L_2\) 归一化，可在 `extract_*_activations` 或 `data.py` 的 load 后对激活做 `x / (x.norm(dim=-1, keepdim=True) + eps)`；当前设计是“在 \(\phi\) 入口做”，便于与未归一化数据兼容。  
2. **\(\phi\) 与 \(J_\phi\)**：  
   - 改多项式阶数/维度：`degree`、`n_components`（已在 CRFDSteer.yaml）。  
   - 换其他核或草图：可仿照 `NormedPolyCntSketch` 实现 `forward`/`grad` 或 `vjp`，再在 `KernelClassifier` 中换 kernel。  
3. **分类器与 \(w,b\)**：  
   - 换为 SVM：`lin_clf_type: svm`（已有）；  
   - 改 \(b\) 的构造（如其他类别平衡方式）：改 `_fit_with_labels`/`_fit_with_two_sets` 中 `dre_coeff` 或 `log_dre` 中的常数。  
4. **ODE 求解**：  
   - 步长/步数：改 `steps`、`T` 或 `options['step_size']`；  
   - 求解器：`solver: midpoint` 或 `rk4`；  
   - 若不用 torchdiffeq：可自行在 `steer()` 里写 for 循环，每步 `state = state + step_size * vector_field(state)`（欧拉），与论文完全一致。  
5. **\(T\) 取值**：论文 **Table 5** 按 backbone 给出 CRFDSteer 的 \(T\) 区间；与 `confs/steer/CRFDSteer.yaml` 默认 `T: 1.0` 不同，复现时应用命令行 **`steer.T=...`** 覆盖（见 `README.md`）。  
6. **Hub 与部署**：若默认探测路径不包含你的数据盘，请在 `.env` 中设置 **`HF_HOME`** 或 **`HF_HUB_CACHE`**；新增机房路径可改 **`get_hub_cache_dir_str`** 中的 `candidate` 列表。

---

## 七、总结

- **L2 归一化**：在 **`NormedPolyCntSketch._normalize`** 中完成，作为 \(\phi\) 的前置步骤，而非单独数据预处理脚本。  
- **\(\phi(a)\) 与 \(J_\phi(a)\)**：**`_poly_cnt_sketch.py`** 的 `forward`（及 Normed 包装）与 `grad`/`vjp`，对应论文的多项式计数草图与前向/雅可比/VJP。  
- **\(w\) 与 \(b\)**：**`_kernel_clf.py`** 的 sklearn 逻辑回归 + `coef`/`intercept` + `dre_coeff = N_-/N_+`，\(h(a)\) = `log_dre`，\(\nabla_a h\) 方向 = `grad`（再在 steer 中 L2 归一化）。  
- **ODE 多步循环**：**`_ode_steer.py`** 的 **`vector_field`**（梯度 + L2 归一化） + **`torchdiffeq.odeint`**（步长循环）；单步版在 **`_step_ode_steer.py`** 的 `X + T * vector_field(X)`。  
- **工程层（增补）**：**`_hub_local.py`** 负责本地 Hub 解析与 **`hf_pretrained_hub_kwargs`**；**`env.py`** 负责 **`.env`**；评估与 LM 加载依赖二者以实现镜像、数据盘缓存与离线 **`local_files_only`**。

按上述映射即可在对应文件中进行针对性修改与扩展。
