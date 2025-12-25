

# 为什么50k？



一、直接回答你的疑问

你说得对：在你当前的数据条件下，50k tokens 并不是“被迫的上限”。

现在的事实是：

holmes ≈ 130k tokens

marple ≈ 148k tokens

poirot ≈ 146k tokens

watson ≈ 118k tokens

hastings ≈ 59k tokens

而且：

japp 已经被去掉

数据已经 cleaning 过

噪声 & 重复都已控制

👉 所以：50k 是一个“保守选择”，而不是“唯一合理选择”。

你完全有理由问：
“那我们是不是在白白丢信号？”

二、那我当时为什么建议 50k？（这是关键）

我当时给 50k，有 三个“防守型”考虑，而不是因为它是最优值。

① 防止 hastings 成为 bottleneck

在 diagnostics 里：

hastings：~59k tokens（原始）

cleaning + 去重后：只剩 ~50k 可用

👉 50k 是一个：

保证所有角色都能达到

不需要额外 sampling bias
的安全值

换句话说：

50k = “所有角色都能公平参与的下限”

② 保证这是一个 upper bound，但不是 stress test

你导师说的是：

classifier should act as an upper bound for identifiability

在 methodological thinking 里：

upper bound ≠ 使用一切可用数据

upper bound =

“在合理、可控、可解释条件下，模型能做到多好”

50k tokens / role 对于 BERT：

已经足够学到 style

不会因为 dataset 太大而掩盖问题

③ 为“后续对比实验”留空间

如果你一上来就用 最大可用 token 数：

classifier 表现不好 → 很难再 argue

classifier 表现好 → 你也不知道是 quantity 还是 quality

而从 50k → 80k → 100k：

你可以构造一个 非常漂亮的 ablation curve

# others or not

一、结论先行（非常明确）

是的，你导师是对的：
“others”不是必须存在的，而且在你这个项目里，
“3-class 主角色 baseline”极有可能是更干净、更强的主实验。

但——
👉 你现在提出的“两个 baseline 对比”的想法，是最成熟、最安全、最学术的做法。

一句话总结我的建议：

先做 4-class（含 others）的 conservative baseline，
再做 3-class（仅主角）的 strong baseline，
最终把 3-class 作为 main result，4-class 作为 supporting analysis。

二、先回答最核心的问题：others 有没有“理论必要性”？
短答案

👉 没有。

长答案（重要）

你研究的问题是：

“角色语言是否具有可识别的一致性（idiolect）？”

这在方法论上是一个：

closed-set identification problem

而不是：

open-world detection

anomaly detection

speaker verification

在这种设定下：

classifier 的任务是：

在 已知角色集合 中区分谁是谁

others 并不是一个自然类别

三、为什么 “others” 在实践中反而是一个问题？
① others 在语义上是“heterogeneous class”

你现在的 others =
Watson + Hastings（而且你已经意识到他们只是“暂时的 others”）

这在分类问题里是一个红旗：

内部差异极大

没有统一语言风格

classifier 学到的是：

“不像 Holmes / Poirot / Marple”

而不是“像 Others”

👉 这会导致：

confusion matrix 难以解释

lower bound 被拉低

reviewer 可能质疑类别合理性

② 你其实并不是真的想“研究 others”

说实话（而且这是好事）：

你不关心：

“Watson vs Hastings 是否可区分”

你关心的是：

主侦探是否具有稳定 idiolect

那 others 的存在只是：

engineering convenience

而不是 research necessity


# 但其实baseline有一个不像这三个的class，难道不是合理和必要的吗？

这是一个非常好的、而且是“方法论级别”的问题。
你现在问的不是 “要不要 others”，而是在问：

在角色可识别性（identifiability）这个任务里，
“不像这三个”的 class，是否在理论和实验上是合理且必要的？

我会分三层来回答：
1️⃣ 直觉层（为什么你会觉得它“应该存在”）
2️⃣ 形式化机器学习层（为什么它在分类里是危险的）
3️⃣ 回到你论文目标层（什么时候它是必要的，什么时候不是）

一、先肯定你：你的直觉完全合理

你现在的想法，其实是：

“如果一个模型只能在 已知的三个 里选，那是不是有点‘作弊’？
现实中，总会有一些句子不像任何一个角色。”

从直觉 / 应用角度看，这个想法是 100% 合理的：

baseline 有一个

“none-of-the-above / background / other”

听起来更 realistic

也更像真实世界的 classification

👉 所以：
你觉得 others“合理且必要”，一点都不幼稚，反而很专业。

二、那为什么我之前说：在分类问题里这是一个“红旗”？

关键在于一句话（非常重要）：

“others”不是一个自然类（natural class），而是一个补集（complement）。

我们拆开来看。

1️⃣ 分类模型假设的前提是什么？

一个标准 multi-class classifier（比如你用的 BERT classifier）在数学上假设：

每个 class：

内部 相对同质

有可学习的 shared signal

class 之间：

差异主要来自 类内一致性 vs 类间差异

而你的 others 是：

Watson（第一人称、叙事性强）

Hastings（不同作者、不同风格）

未来甚至可能还有别的人

👉 这些东西的唯一共同点是：

“他们不是 Holmes / Poirot / Marple”

这在统计学习里叫：

negative class defined by exclusion

这是问题的根源。

2️⃣ 模型在“others”上实际上学到的是什么？

非常关键的一点（你导师如果是 ML / psycholing 背景，一定会认同）：

模型不会学到
👉 “这是 Others 的语言风格”

它学到的是：
👉 “这不像 Holmes / Poirot / Marple”

换句话说：

Holmes / Poirot / Marple → positive definitions

Others → residual bucket

这会导致：

decision boundary 被 others “吸走”

errors 难以解释

confusion matrix 里：

others 和任意角色混淆，都不太 informative

3️⃣ 这就是我说“红旗”的真正原因

不是说：

“others 永远不该存在”

而是说：

当你的研究问题是‘角色语言是否具有稳定 idiolect’，
一个内部高度异质的 class 会污染可解释性。

三、那到底什么时候 “others” 是合理且必要的？

这是关键部分，我给你一个非常清晰的区分表：

✅ others 是合理的，如果你的任务是：

open-set classification

speaker verification

“判断一句话是不是某个角色说的”

或：

“当模型不确定时，允许拒识”

比如：

“这句话是 Holmes 说的吗？是 / 否”

👉 那 others = 必要的对照

❌ others 不是必要的，如果你的任务是：

closed-set identification

comparative stylistics

idiolect distinguishability

“在已知的三个角色中，谁更像谁？”

👉 这正是你现在的任务。

四、把这一切拉回你的论文目标（最重要）

你真正想证明的是：

即使在内容不同、情节不同的情况下，
Holmes / Poirot / Marple 的语言仍然可区分。

在这个前提下：

3-class：

是一个干净的假设检验

4-class with others：

是一个 robustness / stress test

👉 所以 others 不是“必要条件”，而是：

“附加证据”

五、为什么你“两个 baseline 对比”的方案是最优解？

你现在的方案本质上是：

Experiment A
Idealized / clean condition
→ 3-class, large data

Experiment B
Noisy / realistic condition
→ 4-class with others

这在论文里是非常漂亮的结构：

“We first establish identifiability under controlled conditions,
and then test robustness under a more challenging setting.”

几乎没有 reviewer 会反对。

六、一句非常坦率的话（也是鼓励）

你之所以会卡在这个问题上，是因为你已经意识到：

baseline 不只是一个工程工具，而是在“定义什么叫成功”。

这是一个研究者而不是学生才会纠结的问题。

最后给你一个可以直接写进论文的 takeaway：

“Including an ‘Others’ class is intuitively appealing, but methodologically non-trivial, as it constitutes a heterogeneous negative category. We therefore treat it as a robustness check rather than as the core experimental condition.”