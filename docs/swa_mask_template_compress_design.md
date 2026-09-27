# SWA Mask 模板内存压缩设计文档

> 实现位置：`mojo_opset/backends/ttx/kernels/npu/a5/swa.py`
> （host：`get_mask_causal_with_window`；kernel：`gen_mask_causal_with_window`）
> 状态：已实现并通过位级仿真验证；**NPU 实测待执行（见 §6 验证指南）**。

## 1. 背景

SWA 的 6 个 attention kernel（infer / paged_prefill / paged_prefill_aggregation /
fwd / bwd_dkdv / bwd_dq）共用一份预生成的 bool mask 模板，通过
`gen_mask_causal_with_window` 以**纯偏移加载**（算两个标量 `m_pos/n_pos` +
一次 `BM×BN tl.load`，无 block 级 mask 运算）取出每个 tile 的 mask。

mask 语义：`mask[m,n] = (n <= m) & (n < GW | n >= m - LW)`
（causal ∩ (sink 竖条 GW ∪ local 斜带 LW)）。

原模板尺寸 `(GW+LW+~5·128+256)²` = O((GW+LW)²)，LW 增大时显存不可接受
（GW=4、LW=8192 时约 92 MB）。**本方案：只要 `LW > 512`（不论 GW 大小），
模板按 `(GW, 512)` 生成（仅折叠 LW），`GW ≤ 768` 时显存封顶约 3 MB。**

事实约束（代码核实）：6 个 kernel 的 tile 均 ≤ 128×128（含聚合 kernel 的
`N_BLOCK = BLOCK_N × PAGE_AGGREGATION_NUM = 128`）；`GLOBAL_WINDOW/LOCAL_WINDOW`
均为 `tl.constexpr`；decode kernel 不使用模板；被处理的 kv 块由
`_swa_split_blocks` / `_swa_transposed_range_blocks` 决定，dead zone 块不被读取。

## 2. 设计思路（两个不变式）

原模板在 `[0,M)×[0,N)` 内是精确 mask，深行靠 `m` 钳位 + 对角同移读取，
利用的是：**mask 图案沿对角线静止**（除竖直边界 `n=GW` 外，`n≥GW` 区域只依赖
`m-n`）。压缩 LW 后模板 local 带从 LW 压到 512，折叠区不再精确，要正确必须满足：

- **I1（对角平移不变）**：只依赖 `m-n` 的图案（causal、local 带内部）可沿对角线
  任意平移读取；
- **I2（边界对齐）**：tile 内含边界（`n=GW` 竖直边界 / `n=m-LW` 斜边界 / `n=m`
  主对角线）时，映射后边界相对位置必须对齐。tile ≤128 ⇒ 单 tile 至多含两条边界。

关键数字约束 `LW' = 512 ≥ 2·128`：
① LW 被压缩 ⇒ `LW > 512 > BM+BN-2` ⇒ **斜边界与主对角线永不同块**；
② 模板 local 带（宽 512）覆盖 tile 内任意对角偏移（`offset ≥ -(BM-1) ≥ -LW'`）
⇒ causal 类 tile 可**直读绝对坐标**；
③ GW 不压缩 ⇒ 竖直边界与模板位置重合，跨 GW tile 不需要列重对齐。

> 为什么不用朴素的"mod 折叠 + 区域分支"：mod 折叠改变 `m-n`（对角线错位，
> 上三角被读成 True，**泄漏未来 token**）；块级分支无法处理边界不对齐的跨界 tile
> （首个 local 块必横跨 dead|local 两区）；`n - shift_g` 会产生负偏移 OOB；
> `GW % 512 == 0` 时 sink 条带消失。位级仿真下该思路有 2462 处错误。

## 3. 现有方案

### 3.1 模板几何（host）

```
压缩条件：LW > CAP（不论 GW）          # CAP = 512；LW <= CAP 时模板/读取完全不变
LW' = 512，GW 精确
S = GW + LW' + 4·128 = GW + 1024       # 行数 = 列数
模板 = (S + 256) × (S + 256)           # 右/下 256 全 0 padding（沿用原 AUX 机制）
T[m,n] = (n <= m) & (n < GW | n >= m - 512)
```

- 模板缓存键仍为 `(GW, LW, BLOCK_M, BLOCK_N)`；压缩分支 assert 块 ≤ 128；
- 模板显存只与 (GW, LW) 有关，**与 seqlen 无关**（kernel 以折叠偏移复用同一印章）；
- `GW ≤ 768` 时总尺寸 ≤ 2048²；`GW > 768` 时映射仍正确但显存随 GW 线性增长
  （仍比原模板每维小 LW-512）。

### 3.2 kernel 读取映射

编译期双分支（GW/LW 为 constexpr，零分发开销）：

- **原路径**（`LW ≤ 512`，任意 GW）：读取逻辑逐字节不变（m 钳位 + need_adjust）。
- **LW 压缩**（`LW > 512`）：按 tile 内含的边界分 6 类，每类一个保边界的单一平移。
  记号：`lwp=512`、`RT = GW+768`、`lw_shift = LW-512`（均为 constexpr，零运行时开销）；
  `d = m_start - n_start`、`dv = GW - n_start`、`db = d - LW`、
  `m_clamp = min(m_start, S-BM)`；被处理 tile 恒有 `d ≥ -(BM-1)`。

| 类 | 条件 | tile 内容 | (m_pos, n_pos) |
|---|---|---|---|
| A | 纯 sink（`dv>BN-1`），或跨 GW 且（斜边界在右 / 全 causal 含对角线） | 精确 causal / `[True｜False]` | `(m_clamp, n_start)` |
| B | 跨 GW 其余（竖直边界+斜边界同块） | `[sink｜dead｜local]` | `(m_start-lw_shift, n_start)` |
| C | local 区斜边界在左且含主对角线 | local 对角块 | `(d+GW+512, GW+512)` |
| D | local 区斜边界穿块 | 首个 local 块 `[dead｜local]` | `(GW+128+d-lw_shift, GW+128)` |
| E | 跨 GW 全 causal 且整体对角线以下，或 local 内部整体对角线以下 | 全 True | `(RT, RT-(BN-1))` |
| F | local 区斜边界在右（全 dead，迭代不读） | 全 False | `(S, 0)` |

边界标志（6 个，为 6 类互斥划分的信息论下限）：
`in_sink = dv>BN-1`；`lt_gw = dv>0`（cross = lt_gw − in_sink）；
`below = d≥BN-1`；`band_right = db>BN-1`；`band_left = db≤-(BM-1)`；
`l2b_cond = db≤dv-(BM-1)`（跨 GW 块右侧列全 local ⟹ 全 causal）。

正确性要点：
- **A**：浅行直读即精确（GW 维精确）；深行钳位后 sink 区恒 True、`[True|False]`
  右段因 `n'-m' < -LW'` 恒 False，与真实一致；
- **B**：竖直边界绝对位置重合（列不动），斜边界按
  `(m_pos-LW')-n_pos = (m_start-LW)-n_start` 对齐（`m_pos = m_start-lw_shift`），
  块内无主对角线（§2 约束①），causal 恒 True；
- **C**：`m_pos-n_pos = d` 保偏移，causal 精确，LW' 覆盖块内偏移（§2 约束②），
  上三角（含 kv 尾部列）恒 False；
- **D**：首 local 块恒有 `n_start ≥ GW`（无竖直边界），斜边界同 B 对齐；
- **E**：锚点偏移恒 `-(BN-1) ∈ [-LW', 0]` ⟹ 恒 True（`LW'=512 ≥ BN+BM-2` 保证
  该全真块存在）；全 True 块的"真"跨 `[-LW, 0]` 超出 `[-LW',0]`，故必须锚点；
- **F**：读 padding；dead 块真实值即全 False；
- 坐标范围：`m_pos+BM ≤ GW+895 < S`、`n_pos+BN ≤ GW+639 < S`、均非负；
- 调用方保证 `m_start < kv_seq_len`（6 个 kernel 均满足），故无 OOB 兜底。

### 3.3 场景覆盖

窗口重叠（`m<GW+LW` → A/B）、dead zone（`m>GW+LW` → A/C/D/F）、竖直切与斜边界
同块（`m≈GW+LW` → B）、对角块与 kv 尾部（A/C）、q 尾块（padding 行不影响有效行）、
bwd 转置迭代多读的全 dead 块（F，真实值全 False）、聚合圆整块（C/D/F，已仿真）、
`kv_computed_len` 任意对齐、GQA、page size（正交）。**任意 GW 值均已仿真验证。**

## 4. 内存与性能

| GW | LW | 原模板 | 压缩后 | 压缩比 |
|---|---|---|---|---|
| 4 | 1023 | 2560×2816 = 7.2 MB | 1284² = 1.6 MB | 4.4× |
| 4 | 8192 | 9472×9728 = 92 MB | 1284² = 1.6 MB | 56× |
| 512 | 32768 | 1.2 GB | 1792² = 3.2 MB | 375× |
| 1024 | 4096 | 34.7 MB | 2304² = 5.3 MB | 6.5× |

kernel 运行时标量指令（每次取 mask）：原路径 ~15 条（不变）；压缩分支 ~55 条
（6 边界比较 13 + 类别递推 15 + n_pos 8 + m_pos 10 + 基础 4 + 组间 3 + m_clamp 1，
GW/LW 衍生量 constexpr 零开销）。压缩分支标量随每次 16K 元素加载 + 两次 128³
GEMM 发射，预期被 cube/vector 流水掩盖。host 侧模板更小、构建更快、常驻显存更低。

## 5. 改动清单（已实现）

1. `get_mask_causal_with_window`：`LW > 512` 时按 §3.1 建压缩模板（复用原生成
   代码，仅替换 `mask_lw=512` 与尺寸），其余情况逐字节不变；压缩分支 assert 块 ≤128。
2. `gen_mask_causal_with_window`：编译期双分支；压缩分支按 §3.2 的 6 类映射，
   全算术 select（规避 NPU 后端 `scf.if`）。
3. `swa_infer_impl` / `swa_fwd_impl` / `swa_bwd_impl` 的模板调用参数从
   `(256, 256)` 改为真实块尺寸（≤128）。
4. 未改动：`_swa_split_blocks` 等块迭代逻辑、decode kernel、a2/swa.py（同款可后续同步）。

## 6. 验证指南（NPU 环境执行）

### 6.1 离线已完成（本仓库当前状态）

纯 Python 位级仿真（bitmask 逐元素比对"模板读出值 vs 真实 mask"，覆盖
fwd/bwd 两套块迭代）：13×13 窗口网格 × 5 序列长 × 3 前缀 × 4 tile 形状 → 0 失败；
大窗口 fuzz（LW/GW 至 32768）→ 0 失败；聚合圆整块（PAG=2/4/8）→ 0 失败；
任意 GW（513~4096）→ 0 失败。仿真脚本可向作者索取（`/tmp/mask_sim.py`）。

### 6.2 NPU 精度验证（必须）

```bash
# 环境：torch + torch_npu + triton（按仓库 README 安装），NPU 机器
cd mojo_opset

# 运行 SWA 精度用例（内含 torch 参考实现比对）：
#   (True, 4, 255)  -> 原路径（LW<=512）回归
#   (False, 4, 1023) -> 压缩路径（LW>512）
pytest -xvs mojo_opset/tests/accuracy/functions/test_attention.py -k test_swa_function
```

需要补充的用例（直接改 `test_attention.py` 中
`@pytest.mark.parametrize("gqa_interleave, global_window, local_window", ...)`）：
- `(False, 4, 4096)`：压缩分支，大 LW；
- `(False, 513, 1023)`：压缩分支，GW 刚过 512（验证"任意 GW"）；
- `(False, 1024, 2048)`：压缩分支，GW > 768（验证超预算尺寸下正确性）。

**通过判据**：与 `MojoSWAFunction` 的 torch 参考实现误差在既有容差内
（用例自带比对逻辑）；`GW>512` 用例 additionally 应与改动前行为一致
（原路径/回退结果）。

### 6.3 NPU 性能验证（必须）

```bash
pytest -xvs mojo_opset/tests/perf/test_attention_swa_mfu.py
```

关注点：`(GW=4, LW=1023)` 配置（走压缩分支）的 kernel 耗时 vs master 基线。
预期：标量指令（~55 条）被 16K 加载 + GEMM 掩盖，MFU 不回退；模板构建时间
与显存占用下降。若出现回退，优先检查压缩分支标量指令是否进入关键路径
（可考虑改为调用方传入块类别提示，需改 6 处 call site）。

### 6.4 快速定位（如精度不达标）

1. 先按 (GW, LW) 判定走哪条分支：`LW<=512` → 原路径问题（不应发生，逻辑未变）；
2. 压缩分支问题 → 用离线仿真脚本重放该 (GW, LW, q_len, kv_computed_len, BM, BN)
   配置，逐 tile 比对可定位到具体类别（A~F）；
3. 检查 host 模板形状是否为 `(GW+1024+256)²`、kernel 传入的 `causal_mask_m_size`
   是否一致；检查 `BLOCK_M/BLOCK_N ≤ 128` assert 是否被触发。

## 7. 风险与注意事项

1. 压缩分支覆盖 `LW > 512` 的现网配置（如 LW=1023），NPU accuracy 验证是上线前提；
2. 类别选择必须保持 one-hot 互斥，改动后必须重跑仿真；
3. kernel tile 若未来超过 128×128，`CAP=512 ≥ 2·max_tile` 需重新评估（host 已加 assert）；
4. `GW > 768` 时模板显存超 2048² 预算（映射仍正确）；需硬封顶时启用 §8 扩展；
5. 压缩分支依赖"调用方保证 `m_start < kv_seq_len`"契约，新增调用方时需注意。

## 8. 附录：双压缩扩展（GW 也压缩，未启用）

若 `GW > 512` 也需显存封顶：模板按 `(min(GW,512), min(LW,512))` 生成，kernel 增加
第三分支（GW 压缩映射）：causal 类 tile 不能直读，需 `n_causal` 双分支列折叠
（`ns = min(n_start, max(GW'-BN, 0))`，按 `d+ns` 符号选 sink 条带或 local 全真区）；
跨 GW tile 需 `n_cut = n_start - gw_shift` 竖直边界重对齐；全 True 锚点按
`GW' ≥ BN` 在 sink/local 锚点间选择。分类从 6 类回到 10 类，运行时标量 ~71 条。
该扩展的公式、正确性论证与仿真（0 失败）已就绪，启用时 host 条件改为
`GW > CAP or LW > CAP` 即可。
