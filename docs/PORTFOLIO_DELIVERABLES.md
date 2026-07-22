# Portfolio and GitHub deliverables

本文定义 Math RLVR 正式项目最终必须交付的证据、分析与 GitHub 结构。它是
portfolio 交付合同，不授权模型下载、CUDA、评测或训练，也不改变已经冻结的
实验配置。当前四个 1.5B formal training/checkpoint-validation runs 和 seed-42 Base/PPO/GRPO
final comparison 已完成；GRPO123/PPO123 final evaluation 为 `deferred_not_executed`。
唯一有效 next task 以 `docs/NEXT_TASK.md` 为准。

## 1. 最终项目必须回答的问题

最终叙事必须基于保存的证据回答：

- PPO 和 GRPO 分别如何训练，核心更新机制和模型角色有什么不同？
- 如何保证相同模型、数据、prompt、reward、sampling、LoRA、completion 与
  token 预算下的公平比较？
- 训练是否发生真实参数更新，并产生可验证的学习信号和数学能力变化？
- 两种算法的效果、跨 seed 稳定性和 checkpoint 稳定性如何？
- 相对未训练 base baseline，pass@1 和 pass@4 是否提升或退化？
- GSM8K 与 MATH500 overall、MATH500 Level 1–5 的表现有何差异？
- 模型是学会了输出格式，还是提高了可验证数学正确性？
- 时间、显存、GPU-hours 和人民币成本分别是多少？
- Reward shaping 是否产生预期信号，是否出现 reward hacking？
- 有哪些典型成功、退化、格式失败、验证器拒绝和数学失败案例？
- 结论适用于什么范围；有哪些限制和不能支持的主张？

## 2. 完整训练方法记录

每个正式训练 run 的 resolved config、manifest、报告和可复现文档必须共同记录：

- 精确 model repo、revision、dtype 和 `local_files_only` 状态。
- PPO 与 GRPO 的完整训练配置和算法特有参数。
- 共享 policy LoRA 的 rank、alpha、dropout、target modules 和实际可训练参数量。
- PPO 独立 value model、value LoRA、scalar head、参数量和角色隔离。
- Optimizer、learning rate、scheduler、warmup、weight decay 和数值精度设置。
- Per-device batch、rollout/generation batch、gradient accumulation、group size、
  epoch/minibatch/iteration 和 derived optimizer/global steps。
- 完整 prompt 文本、renderer、version 和 SHA256。
- Reward 公式、所有 component、权重、domain routing、version 和 SHA256。
- Parser 与 GSM8K/MATH canonical verifier 的设计、version、SHA256 和失败语义。
- Dataset repo/revision、原始 split、选取规则、manifest SHA256、数据顺序、
  overlap/leakage 检查和 difficulty/level 分布。
- Completion、generated-token、update、optimizer/global-step 的 expected 与 hard cap。
- Temperature、top-p、max completion length、generation seed 和其他 sampling 参数。
- Checkpoint cadence、adapter/head layout、inventory、SHA256 和 same-run resume 规则。
- 可直接复制的完整命令，包括 config path、离线变量和确认参数。
- Python、PyTorch、Transformers、TRL、Accelerate、PEFT、CUDA、driver 等软件版本。
- GPU 为 H800 80GB，以及计费价格 CNY 8.88/GPU-hour。

PPO 的 policy/value/ref/reward 角色和 optimizer 参数集合必须显式审计；GRPO 的
policy/reward 角色也必须审计。不能用“配置相同”掩盖 PPO value model 带来的算法
必要结构、显存、速度和 checkpoint 差异。

## 3. Base baseline 与最终评测

必须严格区分并在 artifact 中标记：

- Base baseline：固定的未训练 1.5B base model，不加载 adapter。
- PPO checkpoint：仅加载选定 PPO policy adapter；value adapter/head 不参与生成。
- GRPO checkpoint：仅加载选定 GRPO policy adapter。
- Validation checkpoint selection：只使用冻结 validation 数据和预先定义规则。
- Frozen final test：checkpoint 冻结后才运行的 GSM8K test 与 MATH500 评测。

禁止使用 frozen final test 调整 prompt、reward、训练超参数、sampling、数据选择或
checkpoint。任何 test-driven tuning 都会使正式比较失效。

正式指标至少包括：

- Greedy accuracy。
- Sampled pass@1。
- Pass@4。
- GSM8K 指标。
- MATH500 overall 指标。
- MATH500 Level 1–5 分层指标。
- Format accuracy。
- Valid-answer rate。
- Verifier status 分布。
- Generation/completion token length。
- Truncation rate。

三个核心正确率指标定义不可混用：

- **Greedy accuracy**：每题使用确定性 greedy completion 时的 canonical 正确率。
- **Sampled pass@1**：每题采样 completion 的平均 canonical 正确比例，再按题汇总。
- **Pass@4**：每题四次独立冻结采样中至少一次 canonical 正确的问题比例。

报告、CSV 列名、图例和文字结论必须保持这些定义。不能把 sampled pass@1 称为
accuracy，也不能由四条样本平均值替代 pass@4。

## 4. 训练过程指标

PPO 每个 update 至少保存：

- Policy loss、value loss 和 total loss。
- Reward mean/std。
- Advantage 与 return 的可用统计。
- Approximate KL。
- Clip fraction。
- Ratio 及其可靠统计。
- Entropy。
- Policy grad norm 与 value grad norm。
- Explained variance，仅在定义和采集可靠时报告。
- Learning rate。
- Completion length。
- 本 update generated tokens 和 cumulative generated tokens。

GRPO 每个 update 至少保存：

- Loss。
- Reward mean/std。
- 每个 prompt group 的四个 reward。
- Group reward variance。
- Zero-advantage group/rate。
- KL，仅当 beta 启用且指标定义可靠时报告。
- Entropy。
- Grad norm。
- Completion length。
- Canonical `RewardStatus` 分布。
- 本 update generated tokens 和 cumulative generated tokens。

任何缺失、不适用或不可靠指标必须保存标准 JSON `null`/CSV 空值、
`available=false` 和明确原因。禁止将不可用指标伪造为 0；真实 0 必须有原始证据。

### Entropy 与策略坍缩证据

PPO 和 GRPO 正式训练必须尝试保存每个 update 的 policy entropy mean、可靠可得时
的 std、checkpoint 8/16/24/32 的 entropy、entropy 对 cumulative generated tokens、
reward/pass rate 的联合变化，以及 completion duplicate/unique rate、completion
length mean/std、EOS rate 和 truncation rate。

每个 entropy 观察必须同时记录：metric source、原始 metric key、使用 logits 还是
log-probabilities、是否只覆盖 response 轴、是否排除 prompt/PAD/EOS、token mask、
token/sequence/batch aggregation、dtype，以及 TRL/Transformers 版本。优先字段为
`response_token_entropy_mean`，其统一定义只允许复用当前 forward 已存在的 policy
logits，以 detached/no-grad 方式对真实 response token 求 Shannon entropy token
mean；排除 prompt/PAD，PPO/GRPO 使用相同公式和 mask，不增加模型 forward，也不
保存完整 logits。

若统一计算需要额外 forward、明显显存开销或侵入式修改 pinned TRL，则：

- 保留 trainer 原生 entropy mean 和精确定义/原始 key；
- `response_token_entropy_mean` 保存 `null`、`available=false` 和原因；
- 不把定义或 aggregation 不同的 PPO/GRPO entropy 做横向数值优劣比较；
- 原生 std 不可得时同样保存 `null`/unavailable，绝不写成 0。

Completion diversity 使用每个问题四次采样内的 raw-completion 精确字节相等定义，
先计算每组 unique/duplicate rate，再对本 update 的四组取均值。最终分析必须回答：

- entropy 是否持续下降；
- reward 提高是否伴随 entropy 过快下降；
- 是否出现大量重复 completion；
- PPO 是否因 KL/clip 约束保持更稳定；
- GRPO 是否同时出现低 entropy 与 zero-advantage；
- pass 率变化更符合数学能力提升还是输出模式收缩。

## 5. 时间、资源与成本

每个 baseline、training、validation 和 final-evaluation run 必须保存：

- Model-load wall time。
- Train/evaluation wall time。
- Generated tokens/sec；定义分母和计时区间。
- PyTorch peak allocated 与 peak reserved，二者分开。
- `nvidia-smi` peak VRAM。
- GPU utilization 时间序列或可信汇总。
- GPU-hours。
- 按 CNY 8.88/GPU-hour 计算的人民币成本。
- 每个 checkpoint 和整个 adapter/head checkpoint 的大小。
- PPO 与 GRPO 的速度、峰值显存和总资源差异。

资源指标不能代替科学指标。Allocator pre-exit residue 与进程退出后的 GPU release
必须分别报告，不能混为内存泄漏或成功训练证据。

## 6. 最终分析要求

最终报告必须分析：

- 在相同 completion 与 token hard budget 下的效果。
- Sample efficiency：达到验证效果所需的 completions/tokens/updates。
- Cost efficiency：每 GPU-hour、每人民币成本对应的提升或退化。
- 跨 seed 和 checkpoint 的稳定性。
- GSM8K 与 MATH500 的泛化差异。
- MATH500 Level 1–5 的难度差异。
- Format learning 与 canonical correctness 的分离。
- Reward mean/std、group variance 和学习信号。
- PPO value model/value loss/explained variance 的稳定性。
- GRPO zero-advantage group/rate 的影响。
- Entropy、reward/pass rate、generated tokens 和 completion diversity 的联合变化。
- PPO/GRPO entropy 定义不同时只比较各自趋势，不比较原始绝对值。
- Pass 率变化是否伴随模式收缩、EOS/截断或重复率异常。
- 高 shaped reward 但 canonical 错误等 reward hacking 迹象。
- 可复现的 error taxonomy 和类别分布。
- 两种算法可能更适合的条件与工程权衡。
- 实验限制，以及两 seed、小模型、固定数据和预算不能支持的结论。

只允许基于实际保存的 per-problem、per-update 和 resource artifacts 下结论。不能
以 PPO/GRPO loss 的绝对数值大小直接判断算法优劣，不能夸大统计显著性。

## 7. 固定规则案例研究

禁止只挑选好看的结果。案例必须由提交的 deterministic selection 规则从冻结结果
选取，并记录规则、候选数量、稳定排序键和 tie-break。至少覆盖：

- Base、PPO、GRPO 都正确。
- 只有 PPO 正确。
- 只有 GRPO 正确。
- 三者全部失败。
- 格式正确但答案错误。
- 推理合理但计算错误。
- Verifier 拒绝。
- Completion 截断。
- Shaped reward 高但 canonical 错误。
- MATH500 不同 Level 的代表案例。

每个案例保存 problem ID/文本、gold answer、Base/PPO/GRPO completion、token count、
parser/verifier status、reward scalar/components 和简短分析。Gold 仅用于报告，不得
泄漏到模型 prompt。

## 8. 最终 GitHub 结构

最终仓库至少规划并完成：

```text
README.md
RESULTS.md
REPRODUCIBILITY.md
LIMITATIONS.md
INTERVIEW_NOTES.md
reports/formal_1p5b/
  00_experiment_protocol.md
  01_baseline_results.md
  02_grpo_training.md
  03_ppo_training.md
  04_checkpoint_selection.md
  05_final_evaluation.md
  06_efficiency_analysis.md
  07_error_analysis.md
  08_case_studies.md
  09_conclusions.md
  metrics/
  samples/
  figures/
```

现在不创建空的最终结果文件，也不预填或伪造指标。目录和文件只在有真实证据时
生成。

## 9. 图片规则

每张图片必须：

- 可从提交的 CSV/JSON 和版本化 plotting code 完整重建。
- 在 Markdown 中使用 repository-relative link。
- 使用 GitHub 可直接显示的 PNG 或 SVG。
- 具有清晰标题、坐标轴名称、单位、图例和 caption。
- Caption 说明数据范围、聚合方式、误差条或缺失指标语义。

策略坍缩分析固定增加以下可重建 PNG：

- `reports/formal_1p5b/figures/entropy_vs_tokens.png`
- `reports/formal_1p5b/figures/entropy_vs_reward.png`
- `reports/formal_1p5b/figures/completion_diversity.png`

禁止从终端输出手抄数字画图，禁止手动修改图片以改变结果，禁止让图与源 CSV/JSON
不一致。

## 10. GitHub 与 Windows 交付

Git 历史中保存：

- Code、config 和 manifest。
- Markdown 文档。
- Git-safe CSV/JSON。
- PNG/SVG。
- 按固定规则筛选的 completion samples。
- Artifact manifest 和 SHA256。
- Windows 可检出的 UTF-8 文本和相对路径；复现命令同时说明 shell 环境前提。

Windows 用户必须能够直接在 GitHub 网页中查看 Markdown 报告和 PNG/SVG 图片，
无需本地 Linux 路径或专用绘图工具；所有报告链接使用 repository-relative path。

Git 历史中禁止保存：

- Base model 或其他完整模型权重。
- Hugging Face cache。
- 完整 checkpoint。
- 大型原始日志或完整 runtime archive。
- Token、auth、proxy、完整环境变量或其他凭据。

最终 LoRA adapter/checkpoint 单独使用 GitHub Release 或 Hugging Face 发布，并记录
版本、SHA256 和与 Git commit/run ID 的绑定；不得直接写入普通 Git 历史。发布决定
留到实验完成后单独审批。

## 11. 长期执行原则

持续强化并优先完成：

- 训练和评测指标。
- Pass@k 和严格指标定义。
- 时间、显存、GPU-hours 与人民币成本。
- 逐题结果和 completion evidence。
- 可重建图片。
- Markdown 科学分析。
- 失败与退化案例。
- 完整可复现命令。

停止扩展：

- 与正确性无关的非关键 guard。
- 重复 artifact schema。
- PTY capture 细节。
- SVG 尾随空格。
- CSV CRLF。
- Optional telemetry。
- 为罕见情况增加的极端 fallback 逻辑。

只有会实质影响以下事项的问题才能阻塞：

- 训练结果正确性。
- PPO/GRPO 公平性。
- 安全性。
- Checkpoint 恢复与可恢复性。
- 报告真实性。

其他问题只能记录为 warning。任何工程警告都不得被用来隐藏失败 seed、重写历史
证据、补造指标或扩大科学结论。
