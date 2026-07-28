# Math RLVR 项目中文面试问答（200题）

本问答只依据仓库中的冻结配置、代码、CSV/JSON和正式报告。涉及GRPO-v2最终结果时，均指seed 42、Qwen2.5-1.5B和一次性冻结hidden-test协议；不能外推为普遍算法优越性。

## 一、项目目标与实验设计

### Q001：这个项目要解决什么问题？
**答案：** 项目研究在相同模型、提示、奖励、LoRA和生成预算下，PPO与GRPO用于数学RLVR时的样本效率、稳定性、资源开销和泛化表现，并以可审计artifact保存全部证据。

### Q002：什么是RLVR？
**答案：** RLVR是Reinforcement Learning with Verifiable Rewards。模型生成数学答案后，由严格parser和canonical verifier给出可复核奖励，而不是依赖主观偏好模型。

### Q003：为什么选择数学任务？
**答案：** GSM8K和MATH的答案可被结构化解析与验证，适合构建确定性奖励；同时数学推理仍足够困难，可以观察格式遵循、可解析性和正确性的不同瓶颈。

### Q004：artifact-first是什么意思？
**答案：** 训练或评测结论必须能从逐completion、逐update、checkpoint inventory、CSV/JSON和checksum重建，而不是只依赖终端日志或人工抄写的汇总数字。

### Q005：v1阶段比较了什么？
**答案：** v1在Qwen2.5-1.5B上完成匹配的PPO与GRPO训练，并完成seed-42 Base/PPO/GRPO held-out比较；GRPO123和PPO123的final evaluation被明确标为`deferred_not_executed`。

### Q006：v2阶段为什么单独版本化？
**答案：** v2引入新的无泄漏数据划分、completion-only warm-start、更大训练覆盖和新hidden test。单独版本化避免用已观察的v1 test调参或混淆两套test身份。

### Q007：GRPO-v2的主要科学问题是什么？
**答案：** 它区分Base能力、warm-start贡献、warm-start之后GRPO的增量贡献，以及新GRPO-v2相对旧GRPO-v1的提升。

### Q008：GRPO-v2为什么只运行一个seed？
**答案：** 项目预注册为单seed改进实验，以控制成本并完成完整证据链。因此结论只适用于该冻结seed和协议，不能声称多seed稳定性。

### Q009：公平比较合同包含哪些核心项？
**答案：** 固定模型revision、数据manifest、prompt、reward、parser/verifier、LoRA、sampling、completion长度、训练completion/token/update预算以及checkpoint与validation cadence。

### Q010：为什么训练预算要用completion和token约束？
**答案：** PPO与GRPO内部step语义不同；用实际生成completion数和token上限对齐，更能保证计算预算和数据暴露公平。

### Q011：v1 formal训练预算是多少？
**答案：** 每个formal run为32 updates、512 training completions，并受131,072 generated-token hard cap约束；checkpoint和validation位于8/16/24/32。

### Q012：GRPO-v2训练预算是多少？
**答案：** 512个唯一prompt各出现一次，每题4个completion，共128 updates、512 microsteps、2,048 completions，训练token hard cap为524,288。

### Q013：为什么checkpoint选择不能使用hidden test？
**答案：** hidden test用于最终泛化测量。若用它选择checkpoint，就把测试信息反馈到模型选择中，破坏held-out含义并产生乐观偏差。

### Q014：GRPO-v2用什么选择checkpoint？
**答案：** 只用dev-v2，按canonical pass@1、parseable、format、低truncation和更早step的词典序规则选择32/64/96/128之一。

### Q015：最终为什么选checkpoint-96？
**答案：** checkpoint-96的dev canonical pass@1最高，为33/128（25.78125%）；checkpoint-128虽format和parseable更高，但canonical pass降到28/128。

### Q016：hidden test被打开了几次？
**答案：** 冻结设计要求一次性打开。四个模型使用同一题目、candidate key、seed和sampling；结果不能触发重训、调参或重新选择checkpoint。

### Q017：为什么v1与v2结果不能直接相减？
**答案：** v1和v2使用不同held-out test身份。只有在同一v2 hidden test上重新评估Base、旧GRPO-v1、warm-start和GRPO-v2后，配对差异才有效。

### Q018：工程成功和科学成功有什么区别？
**答案：** 工程成功关注预算、证据、checkpoint和进程安全；科学成功还要求结果对应冻结协议。零更新工程失败不进入科学统计，完整证据可恢复的launcher异常也不抹掉科学结果。

### Q019：本项目为什么不用test结果继续迭代？
**答案：** 继续依据test调参会把test变成dev。项目预声明test不能触发新run，从而保留最终paired结果的解释边界。

### Q020：项目最核心的最终结论是什么？
**答案：** 在冻结seed-42 v2 hidden protocol上，GRPO-v2 candidate-0为43/400（10.75%），高于Base的6/400和旧GRPO-v1的17/400；这是单seed配对结果，不是普遍优越性证明。

## 二、PPO与GRPO原理

### Q021：PPO的核心思想是什么？
**答案：** PPO通过概率比率和clip限制policy更新幅度，并使用value model估计回报基线；项目中PPO包含policy loss、value loss及相应稳定性telemetry。

### Q022：GRPO的核心思想是什么？
**答案：** GRPO对同一prompt的多个completion按组内奖励做相对标准化，用组内相对advantage更新policy，不需要独立value model。

### Q023：PPO与GRPO最重要的结构差异是什么？
**答案：** PPO需要独立value adapter和scalar value head；GRPO只训练policy LoRA，并从同组completion构造相对advantage。

### Q024：为什么GRPO每题生成4个completion？
**答案：** 组内多个样本提供相对奖励分布。若四个奖励有差异，就能产生正负相对advantage；单个completion无法形成这种组内基线。

### Q025：什么是zero-advantage group？
**答案：** 一组completion奖励全部相等时，标准化后的相对advantage为零，该组不会提供区分候选优劣的policy学习信号。

### Q026：all-zero group和all-equal group有何区别？
**答案：** all-zero是所有奖励都为零；all-equal包括共同为任意相同值。两者通常都产生零方差和zero-advantage，但诊断含义不同。

### Q027：GRPO-v2有多少个reward group？
**答案：** 512个唯一training prompt，每题4个completion，因此共有512个prompt reward groups；每update处理4个prompt group。

### Q028：GRPO-v2有多少组提供非零方差？
**答案：** 正式summary记录367/512组有非零组内reward variance，145组为zero-advantage。这说明训练并非持续没有学习信号。

### Q029：PPO loss与GRPO loss可以直接比较吗？
**答案：** 不可以。两者目标函数、归一化、value项和batch结构不同；数值尺度不同，直接比较大小会误导。

### Q030：PPO为什么更占显存？
**答案：** PPO同时维护policy、value角色、reference计算以及更多训练状态；GRPO省去value model。v1报告中PPO峰值约53 GiB，GRPO约9–11 GiB。

### Q031：reference policy在PPO中做什么？
**答案：** reference提供相对原始policy的概率基准，用于KL或更新约束。项目冻结合同通过policy禁用adapter形成reference，而不是训练额外reference参数。

### Q032：GRPO是否完全没有baseline？
**答案：** 它没有PPO式独立value baseline，但使用同一prompt组内奖励均值和尺度形成相对基线。

### Q033：clip fraction表示什么？
**答案：** 它反映有多少概率比率落入clip约束区域，是更新幅度诊断。若runtime没有可靠提供，项目保存null、`available=false`和原因，而不是写0。

### Q034：approximate KL表示什么？
**答案：** 它近似衡量新旧policy分布变化，用于观察更新是否过大。不同实现或beta设置下定义可能不同，必须连同raw key和definition保存。

### Q035：ratio及ratio variance有什么用途？
**答案：** ratio是新旧policy对相同行为的概率比；其方差帮助判断更新是否集中或异常。非有限必需指标会阻塞，可选telemetry缺失则只记录不可用。

### Q036：为什么不能因为loss下降就说数学能力提升？
**答案：** loss只说明优化目标变化。数学能力必须由冻结dev或hidden canonical verifier结果证明，且format改善也可能降低loss但不提高正确率。

### Q037：为什么GRPO可能没有学习信号？
**答案：** 如果同一题四个completion的reward全相同，组内方差为零，relative advantage消失。项目逐group保存完整reward列表来诊断这一问题。

### Q038：为什么reward shaping仍需canonical verifier？
**答案：** shaping提供稠密的格式和可解析性信号，但最终正确性仍由canonical verifier决定，避免把格式正确误当成数学正确。

### Q039：PPO中的value loss有什么意义？
**答案：** value loss衡量value预测与目标回报的拟合误差，只用于PPO内部稳定性诊断，不能与GRPO loss对比。

### Q040：项目能否据此断言GRPO永远优于PPO？
**答案：** 不能。模型规模、预算、任务和seed都有限；项目只能报告冻结协议中的结果及资源差异。

## 三、Qwen、LoRA与模型角色

### Q041：正式模型是什么？
**答案：** `Qwen/Qwen2.5-1.5B-Instruct`，revision固定为`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。

### Q042：为什么固定model revision？
**答案：** 仓库名不保证权重和配置永远不变。固定revision和canonical local snapshot才能确保各run使用同一模型身份。

### Q043：为什么使用local-only和offline模式？
**答案：** 防止运行时网络漂移、意外下载新revision或泄漏凭据。正式命令设置`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`和`local_files_only=true`。

### Q044：什么是LoRA？
**答案：** LoRA在冻结base线性层旁加入低秩可训练矩阵，以较少参数完成适配；推理时只需base revision和adapter权重。

### Q045：GRPO policy LoRA配置是什么？
**答案：** r=16、alpha=32、dropout=0，目标模块为q/k/v/o projections；base模型参数保持冻结。

### Q046：warm-start训练多少可训练参数？
**答案：** 冻结合同预期并验证policy LoRA可训练参数为4,358,144，optimizer参数集合必须精确等于这些LoRA trainables。

### Q047：GRPO-v2从什么初始化？
**答案：** 它加载warm-start checkpoint-16中的policy adapter作为初始policy，不从旧GRPO-v1 checkpoint继续。

### Q048：GRPO-v2继承warm-start optimizer吗？
**答案：** 不继承。它只加载policy adapter，并为GRPO创建全新optimizer和scheduler，避免SFT momentum污染RL阶段。

### Q049：为什么optimizer角色集合需要审计？
**答案：** 若optimizer包含冻结base参数或漏掉LoRA参数，会改变训练合同、显存和更新行为，因此parameter-group union必须等于policy LoRA trainables。

### Q050：PPO有哪些模型角色？
**答案：** policy LoRA、独立value adapter、scalar value head、冻结reference语义和参数自由的verifier reward路径。

### Q051：GRPO有哪些模型角色？
**答案：** 只有可训练policy LoRA和冻结base model；没有PPO value adapter、scalar value head或可训练reward model。

### Q052：为什么不能上传完整Qwen基础模型？
**答案：** 权重很大、已有公开revision可重建，也可能受许可和存储约束。公开恢复包只保存精确adapter、config、identity和checksum。

### Q053：adapter-only bundle能独立推理吗？
**答案：** 不能单独替代base权重；它与固定Qwen repo/revision组合后可恢复被评估policy，因此bundle必须记录base身份和adapter SHA。

### Q054：如何验证adapter没有路径逃逸？
**答案：** runtime拒绝symlink、非可信checkpoint根、错误role和路径escape，并对允许文件做大小与SHA inventory。

### Q055：为什么checkpoint-128不能替代96做final test？
**答案：** dev预注册规则选择了96。用128替代会在选择后改变科学合同，即使128是最后一步也不允许。

### Q056：BF16的作用是什么？
**答案：** BF16降低训练显存并保留较大指数范围。项目正式1.5B训练冻结为BF16，而不是QLoRA或bitsandbytes量化。

### Q057：为什么项目不用QLoRA？
**答案：** 公平合同冻结为BF16 LoRA；引入量化会改变优化、显存和数值行为，成为新的实验变量。

### Q058：meta parameter count为什么要记录？
**答案：** 它帮助发现模型是否仍有未物化参数或装配异常；正式运行要求模型角色和device状态可验证。

### Q059：model.eval和inference_mode为何重要？
**答案：** evaluation必须关闭训练行为和梯度构建，确保没有backward、optimizer或dropout漂移，并降低资源占用。

### Q060：评测时会加载optimizer、scheduler或RNG吗？
**答案：** 不会。model-bound evaluator只加载base和对应policy adapter；训练恢复状态与推理无关且不进入公开adapter bundle。

## 四、数据集、划分与泄漏控制

### Q061：项目使用哪些主要数据集？
**答案：** 训练与评测主要使用GSM8K、MATH训练记录和MATH500；Countdown只用于早期verifier/smoke，不属于最终1.5B比较。

### Q062：train-v2有多少题？
**答案：** 512题：GSM8K 256，MATH Level 1/2/3分别64/96/96。

### Q063：warmstart-v2有多少题？
**答案：** 256题，是train-v2的声明子集：GSM8K 128，MATH Level 1/2/3分别32/48/48。

### Q064：dev-v2有多少题？
**答案：** 128题：GSM8K 64，MATH Level 1/2/3分别16/24/24；只用于诊断和checkpoint选择。

### Q065：hidden test有多少题？
**答案：** 400题：GSM8K test 200与MATH500 200。

### Q066：hidden MATH500的Level分布是什么？
**答案：** Level 1–5为3/33/43/59/62，总计200。

### Q067：为什么Level 1只有3题？
**答案：** MATH500排除全部v1已观察记录后，Level 1只剩3条。采用不等比分层是预注册的严格去污染设计，不是结果导向选择。

### Q068：为什么Level-1结果只能诊断？
**答案：** 分母仅3，区间极宽，单题变化就是33.3个百分点，不能代表Level-1总体能力或作为headline结论。

### Q069：shared pass@k subset有多少题？
**答案：** 100题：GSM8K 50，MATH500 50；MATH Level 1–5为3/8/10/14/15。

### Q070：如何防止train/dev/test泄漏？
**答案：** 用source ID、revision、content hash和交叉overlap matrix验证；train/dev/test与全部v1 manifests要求零重叠，声明子集关系单独记录。

### Q071：为什么trusted gold与execution manifest分离？
**答案：** execution ledger不应直接泄漏answer或solution；可信构建路径可访问训练gold做验证，而model-bound evaluation只获得题目和受保护verifier接口。

### Q072：warm-start target从哪里来？
**答案：** 只从官方training solution/gold，通过可信builder生成`<reasoning>...</reasoning><answer>...</answer>`，不使用dev/test solution兜底。

### Q073：数据选择是否使用模型输出？
**答案：** 不使用。test和curriculum选择由固定seed、source ID、revision和hash排序确定，禁止使用gold内容、模型结果或v1表现。

### Q074：curriculum如何冻结？
**答案：** 512题在训练前确定完整position/update/slot顺序，每update恰好4个唯一prompt；运行中不得根据reward或dev动态改变。

### Q075：为什么每个train-v2问题只出现一次？
**答案：** 这样能把更大覆盖与固定completion预算结合，避免少量题被反复采样，也使sample ledger容易验证无遗漏、无重复和无顺序漂移。

### Q076：hidden test能否用于prompt A/B？
**答案：** 不能。prompt、reward、parser、sampling和checkpoint都必须在打开hidden test前冻结。

### Q077：什么是content hash overlap？
**答案：** 即便source ID不同，内容相同也可能泄漏；对规范化题目内容取hash并交叉比较，可捕获复制或别名记录。

### Q078：为什么MATH主指标使用micro-average？
**答案：** 各Level样本数不等，尤其Level 1仅3题。200题micro-average按实际问题数加权，不会让极小层获得不成比例影响。

### Q079：未选择的数据如何处理？
**答案：** data freeze报告记录各层容量、选择数和未选择数；未选择记录不会在运行中动态补入。

### Q080：数据泄漏审计结果如何表达？
**答案：** 对train、warmstart、dev、hidden test和v1 manifests给出机器可读overlap matrix；除声明子集外，要求对应交叉项为0。

## 五、Prompt、Reward、Parser与Verifier

### Q081：正式输出协议是什么？
**答案：** completion必须恰好包含一个`<reasoning>...</reasoning>`，随后是一个终止的`<answer>...</answer>`。

### Q082：为什么严格输出格式重要？
**答案：** verifier需要确定性提取最终答案。缺失、重复或未闭合标签会导致格式或解析失败，即使文本中可能隐含正确数值。

### Q083：prompt如何保证PPO和GRPO一致？
**答案：** 两者调用共享`math_rlvr.prompt` renderer，并在run identity中保存prompt版本、hash和逐问题prompt hash。

### Q084：reward由哪些部分组成？
**答案：** 正式reward包含格式、语义/可解析有效性和canonical正确性；v1 formal权重为0.10、0.10、0.80，具体身份由冻结reward config/hash约束。

### Q085：valid-answer component是什么？
**答案：** 它是canonical evidence定义下的报告/telemetry指标，反映答案是否满足既定有效性映射；它不改变reward scalar或optimizer输入。

### Q086：为什么历史valid-answer曾出现假0？
**答案：** runtime曾读取废弃的`components.valid_answer`，而新RewardEvaluation是扁平结构。H.4修正字段映射，历史原生值保留并由canonical报告更正。

### Q087：parser和verifier有什么区别？
**答案：** parser负责严格提取和规范化模型答案；verifier将解析结果与gold进行canonical数学判定。

### Q088：FORMAT_ERROR是什么意思？
**答案：** completion不满足冻结的输出envelope或格式合同，尚未进入有效表达式/答案判断。

### Q089：INVALID_EXPRESSION是什么意思？
**答案：** 已找到候选答案，但表达式无法由安全解析路径解释为有效数学表达式。

### Q090：INVALID_NUMBER_USAGE是什么？
**答案：** 在要求数字使用约束的任务中，表达式使用了不允许的数字或次数；它与普通wrong answer不同。

### Q091：WRONG_ANSWER是什么意思？
**答案：** completion可解析且格式有效，但canonical结果与gold不等价。

### Q092：VERIFIED_PASS是什么意思？
**答案：** parser和canonical verifier均通过，预测与gold在冻结语义下等价。

### Q093：为什么不用Python eval验证答案？
**答案：** `eval`/`exec`会执行不可信模型文本，存在安全风险且语义不稳定。项目使用AST/Fraction和math-verify等受限路径。

### Q094：Infrastructure error为何不能记reward 0？
**答案：** 解析器或verifier自身故障不是模型错误。把基础设施故障记0会污染训练和评测，因此INFRA_ERROR必须fail closed。

### Q095：format提高是否等于数学能力提高？
**答案：** 不等于。format只说明协议遵循改善；必须看到canonical pass提高，才能描述为该dev/hidden协议上的正确率改善。

### Q096：parseable但wrong answer说明什么？
**答案：** 模型已学会输出可验证表达式，但推理或计算仍错误。这类失败能区分协议瓶颈和数学瓶颈。

### Q097：为什么保存verifier detail？
**答案：** 它让聚合状态可追溯到逐completion判定，支持错误分类、恢复metric finalization和检测汇总与primary evidence矛盾。

### Q098：prompt token是否参与warm-start loss？
**答案：** 不参与。system/user/generation boundary和padding labels均为-100，仅assistant reasoning、answer与EOS为active labels。

### Q099：为什么EOS是active label？
**答案：** EOS属于assistant目标结束行为；监督它有助于模型学习完整结束，而不是只学习内容token。

### Q100：能否为了提高format放宽parser？
**答案：** 不能。放宽parser会改变科学测量标准；v2保持prompt、parser、verifier和reward语义冻结。

## 六、Warm-start与GRPO-v2训练

### Q101：warm-start的目的是什么？
**答案：** 主要教授严格输出协议和answer提取，同时提供有限的可信解题监督，为GRPO提供更可用的初始化。

### Q102：warm-start训练合同是什么？
**答案：** seed 42、256 samples、1 epoch、microbatch 4、gradient accumulation 4、effective batch 16、64 microsteps和16 optimizer/global/scheduler steps。

### Q103：warm-start长度上限是什么？
**答案：** prompt cap 928、target active tokens含EOS cap 640、实际combined sequence cap 1,088；独立cap不能相加成1,568。

### Q104：为什么修订warm-start容量？
**答案：** 真实tokenizer审计发现target p95 363、p99 497、max 609，prompt max 914；新cap保留余量且实际combined max 1,019仍低于1,088。

### Q105：warm-start是否发生截断？
**答案：** 256/256真实target重新审计后prompt、target、combined overflow均为0，truncation为0。

### Q106：warm-start dev结果是多少？
**答案：** Base dev为6/128（4.6875%），warm-start为8/128（6.25%）；差异小，只支持协议遵循改善和可作为RL初始化，不支持显著数学提升。

### Q107：warm-start-only hidden结果是多少？
**答案：** candidate-0为10/400（2.50%），shared subset unbiased pass@1/@4/@10为3.30%/11.03%/19.00%。

### Q108：为什么warm-start-only仍有价值？
**答案：** 它改善format/parseability并为GRPO提供可学习初始化；最终归因需要比较warm-start-only与GRPO-v2，而不能把全部增益都归给SFT。

### Q109：GRPO-v2训练从何开始？
**答案：** 从已验证warm-start checkpoint-16 policy adapter开始，加载adapter SHA后创建fresh GRPO optimizer/scheduler。

### Q110：GRPO-v2每update处理多少样本？
**答案：** 每update 4个prompt，每prompt 4个completion，共16 completions；128 updates合计2,048。

### Q111：GRPO-v2实际生成多少训练token？
**答案：** 正式training summary记录230,675 rollout generated tokens，低于524,288 hard cap。

### Q112：为什么prompt cap后来又修订？
**答案：** 首次GRPO-v2 attempt仍保留旧832 cap，真实curriculum出现914-token prompt。R.1把GRPO prompt cap传播为928、sequence ceiling为1,184。

### Q113：全量prompt preflight审计了什么？
**答案：** 在CUDA和模型加载前用真实pinned tokenizer/renderer检查512 training和128 dev prompts、顺序、hash、cap、combined ceiling及零truncation。

### Q114：为什么optimizer=None曾导致失败？
**答案：** Transformers/TRL采用lazy optimizer lifecycle，Trainer构造后optimizer为None属正常；旧审计过早读取`.state`。

### Q115：optimizer生命周期如何修复？
**答案：** 构造后记录lazy状态，在原生optimizer创建后首步前审计参数角色和fresh空state，首步后验证moment state物化及scheduler推进。

### Q116：GRPO-v2最终训练是否科学成功？
**答案：** 是。run `grpo_v2_seed42_20260726T044303Z`完成128 updates、512 microsteps、2,048 completions和四次dev，状态为`scientific_training_and_dev_success`。

### Q117：训练中format为何重要？
**答案：** 训练早期大量completion可能因envelope失败拿不到有效性/正确性奖励；format曲线能判断reward是否首先改善协议遵循。

### Q118：训练中为什么保存逐completion token IDs？
**答案：** 可核对文本、mask、精确token数、EOS和truncation，支持预算审计和metric恢复，避免只相信汇总。

### Q119：dev completion计入训练预算吗？
**答案：** 不计入。四次dev各128个单candidate completion，token和completion ledger与训练2,048条及524,288 cap严格隔离。

### Q120：训练完成后为什么不能根据hidden结果重跑？
**答案：** 那会把hidden test反馈进训练选择，破坏预注册协议；无论结果正负都必须保留且停止调参。

## 七、训练与诊断指标

### Q121：reward mean表示什么？
**答案：** 它是当前update所有completion scalar reward的平均值，只描述当前采样质量，不单独证明policy长期提升。

### Q122：reward std/variance为何重要？
**答案：** GRPO依赖组内相对差异；总体和逐组方差能显示是否存在可区分候选的学习信号。

### Q123：canonical pass rate是什么？
**答案：** 当前证据中`VERIFIED_PASS` completion比例，由严格canonical verifier决定，是比format或parseable更接近最终正确性的指标。

### Q124：format rate是什么？
**答案：** 满足冻结`<reasoning>`与`<answer>` envelope的比例；它是必要但非充分条件。

### Q125：parseable rate是什么？
**答案：** completion能被安全parser转换为可验证答案/表达式的比例；parseable仍可能是wrong answer。

### Q126：accuracy_given_parseable如何计算？
**答案：** 分子为canonical pass，分母为parseable completion数；若分母为0，必须保存null、`available=false`和`zero_denominator`。

### Q127：entropy反映什么？
**答案：** entropy描述policy输出分布的不确定性或集中程度，可用于观察过早坍缩，但必须结合实现的raw metric key和定义。

### Q128：为什么PPO与GRPO native entropy不能直接比较？
**答案：** TRL路径可能在不同token mask、batch或目标上聚合entropy，定义和尺度不一致；只有同算法同定义的趋势可安全比较。

### Q129：grad norm反映什么？
**答案：** 它反映可训练参数梯度整体尺度，用于发现爆炸、消失或异常更新；不是能力指标。

### Q130：grad norm缺失时如何处理？
**答案：** 作为optional telemetry保存value=null、available=false、实际reason和raw key；不得伪造为0，也不得阻塞checkpoint或训练。

### Q131：grad norm为NaN/Inf时如何处理？
**答案：** 与“缺失”不同，非有限梯度可能表示训练正确性问题，应fail closed。

### Q132：learning rate为何逐step保存？
**答案：** 可验证scheduler是否按冻结合同推进，并支持恢复后检测重复或跳步。

### Q133：EOS rate表示什么？
**答案：** completion在上限前正常生成EOS的比例；低EOS常与过长输出或truncation有关。

### Q134：truncation rate表示什么？
**答案：** 达到max completion length而未正常结束的比例；截断可能破坏closing tag、format和最终答案。

### Q135：duplicate rate为何重要？
**答案：** 同一prompt多个候选若高度重复，会降低有效探索和pass@k收益；项目保存组内或候选级重复率。

### Q136：tokens/sec如何解释？
**答案：** 它是资源吞吐指标，需注明训练或generation范围；不同prompt长度和candidate batch不能仅凭tokens/sec判断算法质量。

### Q137：peak allocated和peak reserved有何区别？
**答案：** allocated是活跃tensor占用，reserved包含PyTorch allocator缓存。nvidia-smi显存又是进程级设备观测，三者不能混写。

### Q138：optional metric为什么不能写0？
**答案：** 0是一个有效测量值，缺失是未知。混写会制造假结论，例如历史valid-answer假0和nullable telemetry问题。

### Q139：哪些数值问题必须阻塞？
**答案：** 必需loss/reward/gradient的NaN/Inf、预算越界、optimizer角色错误、证据错位、checkpoint损坏等影响训练正确性或真实性的问题。

### Q140：哪些通常只记warning？
**答案：** optional telemetry缺失、PTY、CSV CRLF、SVG空格、进程退出前allocator residue但退出后GPU已释放，以及可从CSV/JSON重建的图表暂时失败。

## 八、pass@k与统计解释

### Q141：candidate-0 accuracy是什么？
**答案：** 对全部400题各取冻结candidate index 0的二元canonical结果求准确率，是四模型paired主指标。

### Q142：unbiased pass@1是什么？
**答案：** 在固定100题shared n=10池上，先根据每题10个候选中的正确数c计算pass_hat(1)，再对问题平均。

### Q143：为什么candidate-0 accuracy与unbiased pass@1不同？
**答案：** 前者用400题的某一个固定候选；后者用100题、每题10个exchangeable draws估计随机抽1次成功概率。问题集合和估计方式都不同。

### Q144：无偏pass@k公式是什么？
**答案：** 对n=10、正确候选数c，`pass_hat(k)=1-C(10-c,k)/C(10,k)`，组合数用精确整数计算后再转浮点。

### Q145：该公式的直觉是什么？
**答案：** `C(10-c,k)/C(10,k)`是从10个候选中无放回选k个且全部错误的概率；1减它就是至少一个正确的概率。

### Q146：为什么同一n=10池能计算pass@1、pass@4和pass@10？
**答案：** 三个估计只需要同一题的n=10和正确数c，分别代入k=1、4、10；无需为不同k重新采样。

### Q147：为什么不能只看前k个candidate？
**答案：** 那是特定顺序候选的observed success，不是基于n=10 exchangeable pool的无偏pass@k估计，且浪费其余候选信息。

### Q148：n=10、c=1时三个值是多少？
**答案：** pass@1=0.1，pass@4=0.4，pass@10=1。

### Q149：n=10、c=2时pass@4是多少？
**答案：** `1-C(8,4)/C(10,4)=1-70/210=2/3`。

### Q150：pass@10为什么只要c>0就是1？
**答案：** k=n=10时选中全部候选；只要池中至少一个正确，十选十必然包含正确候选。

### Q151：为什么逐题pass@1≤pass@4≤pass@10？
**答案：** 从同一候选池抽取更多候选，至少一个正确的概率不会下降；项目同时验证逐题和aggregate单调性。

### Q152：为什么shared subset每题必须恰好10条？
**答案：** 公式冻结n=10。缺失或重复candidate会改变n或c，不能把缺失当错误或静默缩小样本池。

### Q153：bootstrap CI如何计算？
**答案：** 以问题为重采样单位，对paired差异反复bootstrap，取冻结分位数形成95%区间；不能把candidate当独立问题重采样。

### Q154：McNemar检验适用于什么？
**答案：** 适用于同一400题candidate-0二元结果，基于一个模型对而另一个错的discordant pairs做精确检验。

### Q155：为什么pass@k差异不用McNemar？
**答案：** 每题pass_hat(k)是连续估计值而非单个二元结果，项目对其采用problem-level paired bootstrap。

### Q156：Base的最终pass指标是多少？
**答案：** candidate-0为6/400（1.50%）；shared subset unbiased pass@1/@4/@10为1.70%/6.03%/12.00%。

### Q157：旧GRPO-v1的最终pass指标是多少？
**答案：** candidate-0为17/400（4.25%）；shared subset为5.60%/15.87%/25.00%。

### Q158：GRPO-v2的最终pass指标是多少？
**答案：** candidate-0为43/400（10.75%）；shared subset为14.40%/31.14%/42.00%。

### Q159：Base到GRPO-v2的paired结果是什么？
**答案：** +9.25个百分点，38题改善、1题退化，bootstrap 95% CI为[+6.50,+12.25]，McNemar p=1.46e-10。

### Q160：统计显著是否等于普遍有效？
**答案：** 不等于。paired结果对该400题和seed很强，但单seed、单模型和特定协议限制外部有效性。

## 九、工程、checkpoint、恢复与成本

### Q161：checkpoint为什么要保存inventory？
**答案：** inventory记录文件名、大小和SHA，可验证adapter与恢复状态完整、无symlink/path escape，并拒绝完整base权重。

### Q162：可信GRPO checkpoint包含什么？
**答案：** policy adapter、optimizer/scheduler、RNG、runtime counters、curriculum cursor、completion/metric prefix、identity及SHA inventory。

### Q163：公开adapter bundle应包含什么？
**答案：** 只包含推理所需adapter weights/config、模型revision、checkpoint/adapter SHA、角色说明和bundle checksum；不含optimizer或base权重。

### Q164：为什么optimizer/RNG不公开上传？
**答案：** 它们只用于精确训练resume，体积较大且不需要推理；可另存私有持久存储，但公开Release默认最小化。

### Q165：什么是incremental evidence？
**答案：** 每个update完成后、checkpoint callback前原子保存completion、metrics、ledger和counters，使后续validation/finalization失败也不丢已完成训练证据。

### Q166：历史失败run为什么不可修改？
**答案：** 它们是工程审计证据。覆盖或改写会破坏时间线、checksum和失败根因真实性。

### Q167：Base hidden run为何需要metric recovery？
**答案：** 1,300条generation evidence完整，但finalizer误用了128-row dev aggregator。CPU恢复从不可变primary evidence重算指标，不重新generation。

### Q168：Base原run与恢复状态如何区分？
**答案：** 原状态保留`engineering_failure_after_generation_during_metric_finalization`，补充composite为`scientifically_complete_with_recovered_metric_finalization`。

### Q169：launcher IPC失败为何不一定使训练作废？
**答案：** 若worker已原子写完科学artifact、checksum和checkpoint，parent只是在传输大对象时卡死，科学证据仍完整；需独立验证后保留成功状态。

### Q170：hidden evaluator如何避免大IPC对象？
**答案：** worker把1,300条completion写入磁盘，IPC只返回小型primitive状态、路径、计数和失败原因，parent从summary文件读取。

### Q171：完整基础模型权重如何被拒绝？
**答案：** checkpoint allowlist和inventory禁止`pytorch_model.bin`或完整base `model.safetensors`，只允许明确role的adapter及恢复状态。

### Q172：GPU释放如何验证？
**答案：** worker退出后由非CUDA parent检查PID、nvidia-smi compute process和显存基线；进程退出前allocator缓存仅为warning。

### Q173：GRPO-v2训练only成本是多少？
**答案：** warm-start加GRPO-v2训练only telemetry为0.299755 GPU-hours、约¥2.6618；不把checkpoint dev和IPC idle伪拆成精确数值。

### Q174：四模型hidden评测成本是多少？
**答案：** 5,200 completions、606,487 tokens、1.980286 GPU-hours、约¥17.5849，峰值显存5,321 MiB。

### Q175：为什么成本账目有minimum和full instance两种？
**答案：** minimum只加不重叠且可确认组件；full instance用完整launcher占用替代重叠training-only，避免重复计费。

### Q176：checkpoint-dev-only成本是多少？
**答案：** 不可用。现有artifact未单独持久化四次checkpoint dev的GPU telemetry，它包含在full launcher wall中，不能伪造精确值。

### Q177：IPC idle-only成本是多少？
**答案：** 不可用。full launcher记录合并了dev、finalization和IPC等待，无法可靠拆分，故保存null与原因。

### Q178：PPO与GRPO显存实测差异是什么？
**答案：** v1正式运行中PPO峰值约53 GiB，GRPO约9–11 GiB；差异主要来自PPO的value角色和更重训练状态。

### Q179：灾难恢复为什么不能只写AutoDL路径？
**答案：** 当前服务器和`/root/autodl-fs`可能同时销毁；只有上传到GitHub Release或其他独立持久存储并重新下载验SHA，才算异地保存。

### Q180：如何证明archive可恢复？
**答案：** 记录源文件SHA、bundle SHA和inventory，从独立位置重新下载，执行checksum、tar listing和解压检查，并扫描禁传文件。

## 十、结果、限制、简历与未来工作

### Q181：四模型candidate-0结果是什么？
**答案：** Base 6/400（1.50%）、旧GRPO-v1 17/400（4.25%）、warm-start-only 10/400（2.50%）、selected GRPO-v2 43/400（10.75%）。

### Q182：旧GRPO-v1到GRPO-v2提升多少？
**答案：** +6.50个百分点，31题改善、5题退化，bootstrap 95% CI为[+3.75,+9.50]，McNemar p=1.29e-5。

### Q183：warm-start到GRPO-v2提升多少？
**答案：** +8.25个百分点，37题改善、4题退化，95% CI为[+5.25,+11.25]，McNemar p=1.03e-7。

### Q184：Base到warm-start结果如何解释？
**答案：** candidate-0从1.50%到2.50%，+1.00个百分点，McNemar p=0.125；不能据此声称确定数学提升。

### Q185：为什么v2比旧v1好？
**答案：** 可观察到更大训练覆盖、completion-only warm-start、确定性curriculum和dev-only选择共同出现；本实验没有逐项消融，不能断言单一原因。

### Q186：warm-start具体贡献是什么？
**答案：** 它主要提高协议遵循和可解析性，并提供GRPO初始化；warm-start-only正确率有限，GRPO相对它的增量才支持RLVR贡献。

### Q187：GRPO-v2 format和parseable是多少？
**答案：** final candidate-0上format为58.75%，parseable为48.50%，高于Base的7.75%和7.00%。

### Q188：GRPO-v2 truncation是多少？
**答案：** candidate-0 truncation为8.50%；completion cap仍是256，没有因为结果修改。

### Q189：error analysis如何避免挑好案例？
**答案：** 按固定transition类别和稳定排序机械选取，包括改善、退化、格式失败、解析失败、parseable wrong和truncation，不只展示支持算法的案例。

### Q190：GSM8K与MATH结果如何使用？
**答案：** 按冻结domain分别报告分子/分母和rate，用于诊断任务差异；不能事后改变domain混合或训练选择。

### Q191：MATH Level结果最大的限制是什么？
**答案：** Level样本不均，尤其Level 1只有3题。主要MATH结论使用200题micro-average，Level图只作诊断。

### Q192：单seed结果的主要风险是什么？
**答案：** 可能受初始化、采样和数据顺序偶然性影响；需要预注册的更多seed才能评估方差和稳定性。

### Q193：如果继续研究，最优先做什么？
**答案：** 在不再查看当前hidden test的前提下，用新冻结test和多seed复现实验，并做warm-start、数据覆盖和curriculum的受控消融。

### Q194：能否继续用当前hidden test调reward？
**答案：** 不能。当前test已打开，任何基于其结果的reward、prompt或checkpoint调整都会污染后续结论。

### Q195：简历中如何一句话描述项目？
**答案：** 构建artifact-first数学RLVR系统，在Qwen2.5-1.5B上以预注册数据划分和无偏pass@k比较PPO、GRPO及warm-start GRPO-v2。

### Q196：最重要的量化简历bullet是什么？
**答案：** 在冻结400题paired hidden test上，将candidate-0从Base 1.50%和旧GRPO-v1 4.25%提升到GRPO-v2 10.75%。

### Q197：第二条量化bullet可以写什么？
**答案：** 在共享n=10、100题池上实现14.40%/31.14%/42.00%的unbiased pass@1/@4/@10，并保存逐候选可重算证据。

### Q198：第三条量化bullet可以写什么？
**答案：** 完成128个RL updates、2,048训练completions和5,200最终评测completions，使用adapter-only checkpoint与SHA inventory保证可审计恢复。

### Q199：面试时最应避免的夸大是什么？
**答案：** 不说“GRPO普遍优于PPO”、不把单seed当稳定性证明、不把Level-1三题外推，也不把format提升自动称为数学能力提升。

### Q200：项目最终体现了什么能力？
**答案：** 它同时体现实验设计、RL训练、严格评测、统计解释、GPU工程、失败恢复、证据治理和可复现发布，而不是只展示一个最高准确率。

## 自动验证

运行以下CPU-only检查可验证编号、题目、答案和README链接：

```bash
python scripts/validate_interview_200_qa.py
```

冻结要求：Q001至Q200连续；问题数200；答案数200；重复ID为0；README必须链接本文件。
