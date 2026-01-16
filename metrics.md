# Research Question

- RQ1. How well do agents remain in character across dialogue?
- RQ2. What linguistic signals indicate OOC?
- RQ3. How does OOC affect the interaction/task performance?
- RQ4. How does the interaction influence individual character coherence? (TBC)

We evaluate OOC from linguisitc perspective instead of personality test such as Big Five and MBTI. <br>

To answer RQ1 & RQ2, I will observe quantitatively if some features are changing, if so, with comparable to the baseline, is it already OOC or acceptable. <br>

RQ3, only task performace. Successful rate. Run each case 100 times. <br>

RQ4. TBC <br>

# Pre - Analysis

### Baseline 1 - Zero Prompt
Case solving task with default zero-shot LLM prompt, to verify if cases are solvable. 



### Baseline 2 - Zero Collaboration
Case solving task with character prompts but without collaboration.



### Model-Based Baseline - BERT Classifier Model

> Operationalize OOC as a probabilistic drift away from canonical literary style, measured by a supervised classifier trained on original texts, and to analyze whether such drift increases over interaction time.

**Legitimacy**: 
1. Original texts comes from orginal books, also extraction method (LLM prompt)
2. Accuracy/Macro-F1 results 
    - *The classifier achieves moderate but reliable performance (accuracy = 0.76, macro-F1 = 0.76), which is sufficient for its role as a reference model rather than an oracle. The classifier performance is sufficient to establish legitimacy as a baseline. It clearly outperforms chance, shows balanced per-class performance, and produces interpretable confusions consistent with literary similarity, which is appropriate for measuring probabilistic character drift rather than perfect identification.*.

```
===CURRENT RESULTS===
Evaluating on TRAIN set (diagnostic)...
Train accuracy: 0.8908 第二次 0.9400

Evaluating on TEST set...
Test accuracy: 0.7598 第二次 0.7620
Test macro-F1: 0.7612 第二次 0.7633

Confusion Matrix:
[[374  27  37]
 [ 26 325  94]
 [ 52  94 345]]
第二次:
[[379  24  35]
 [ 27 331  87]
 [ 48 106 337]]

Per-class report:
              precision    recall  f1-score   support

      holmes       0.83      0.85      0.84       438
      marple       0.73      0.73      0.73       445
      poirot       0.72      0.70      0.71       491

    accuracy                           0.76      1374
   macro avg       0.76      0.76      0.76      1374
weighted avg       0.76      0.76      0.76      1374
```

> *baseline*




# List of OOC Indicators (RQ1, RQ2)

||Dimension|Evaluation|Metrics|Output
|--|--|--|--|--|
|1.0    |Descriptive      |zero-prompt时案件成功率 vs 有角色prompt无合作prompt时的成功率 vs full-setting成功率）|rate(zero_prompt), rate(no_collab), rate(full)| 三个数的对比表 
|2.0    |Classifier eval  |用 classifier 的概率输出作为代理信号，测量 LLM 三个侦探角色在整段对话过程中风格的动态一致性趋势。|Classifier_P(True), Brier, Brier_Slope|
|3.1    |Lexical          |Character-specific vocab rate, TF_IDF|Lexical_Cosine
|3.2    |Lexical          |Intra-agent cosine distance|IntraAgent_Dist
|4.0    |Syntactic        |(Syntactic complexity) i.e. Maximum dependency tree depth|Syntax_DepthDiff|syntactic drift 随 turn 漂移的回归图
|5.1    |Discourse        |Discourse function / Dialogue Act|DialogueAct_KL|随 turn 漂移的回归图
|5.2    |Discourse        |Sentiment trajectory|Sentiment_Dist|sentiment drift 随 turn 漂移的回归图
|6.1    |Validation (Master table)|Probability aligned turn wise w/ Crr. & ECE | 展示重要性排序的排名图，master table
|6.2|Validation           |Clustering for contamination||





# 1. Descriptive

1.0 Average task accuracy.
- HOW-TO: 
  - 跑zero prompt和no collab 各10次，计算成功率

- EVAL:
  - Compare baseline 1 to baseline 2 - if it is lower, role-playing has negative impact.
  - Compare to baseline 1 - if it is lower, ~above
  - Compare to baseline 2 - if it is lower, collaboration has negative impact.

- Results:
  - rate(zero_prompt): case1 - 97%(47/50), case2 - 100%, case3 - 100%.
  - rate(with_persona, no_collab): case1 - 92%, case2 - 35.33%, case3 - 100%.
  - rate(full_prompt): case1 - 20%, case2 - 50%, case3 - 40%.



# 2. Classifier model

- HOW-TO
  - 2.1 Extract the **predicted probability** of the correct class and **plot mean** over turns over simulations. Also, CIs over simulations
  - 2.2 Calculate average **Brier score** (mean squared error of predicted probabilities) over turns (in order to estimate whether there is a significant increase or decrease trend).
    - Also Fit **linear regressions** to the scores over turns.‼️ 哪个score? brier?

‼️ prob ↑ = 越像角色， brier ↓ = 越像角色，对吧？

‼️ 实验结果说明 OOC drift 不是线性的、不是稳定的趋势，而是 agent-dependent 和 case-dependent 的离散跳变（non-linear, non-monotonic）。

- Results: 本研究的三个侦探代理（Holmes、Poirot、Marple）依赖于相同的底层语言模型，仅通过角色 prompt 来区分。因此，代理之间的差异并非来源于模型结构或训练，而是来自 prompt 对语言生成过程的影响。在多轮协作推理任务中，模型的概率估计与 Brier 分数呈现出明显波动，各 run 的回归斜率不显著，说明 prompt-based persona 引导无法保证稳定、一致的推理行为模式。模型在不同对话轮次可能产生不一致的信心判断，而非随着线索积累而逐渐收敛。因此，本研究的结果更能解释为：prompt 角色设定的可控性在长对话场景中表现有限，而非 LLM 在角色内部构建了稳定的推理结构。


# 3. Lexical
3.1 Character-specific vocabulary rate

- Extract TF-IDF vectors from the gold standard dialogues of each character (avg) and the single turns of each character, and calculate cosine similarity between the tf-idf vectors

‼️ GOLD STANDARD DIALOGUE是什么意思  
‼️ 原作太长，simulation 太短，TF-IDF 会失衡。这样做如何？  
Holmes 的原作文本切成固定长度的 chunks，例如每 200–300 字一段。对所有 chunk 计算 TF-IDF → 得 Matrix M (#chunks × vocab)。取 M 的均值向量作为 Holmes 词汇风格向量 holmes_style

3.2 Intra-agent cosine distance: character distance = (cosine similarity between turns) - (cosine similarity between current turn & turn at same index from a different character)

- tf-idf是为了单词上的，intra-agent cosine是embedding上的, embedding similarity between and within characters. 只能说明一定的相似性，不能说明是因为什么（同风格or同话题）

# 4. Syntactic

4.1 Maximum dependency tree depth

- its avg and SD
- the differences, fit linear regression model over turns (to see if significant trend of in/de-creasing difference.)

# 5. Discourse

5.1 Discourse function / Diaglogue act

5.2 Sentiment trajectory

# 6. Validation (Master table)


Master table (example)

| Turn | Classifier_P(True) | Lexical_Cosine | IntraAgent_Dist | Syntax_DepthDiff | DialogueAct_KL | Sentiment_Dist | Topic_Entropy | ... |
| ---- | ------------------ | -------------- | --------------- | ---------------- | -------------- | -------------- | ------------- | --- |
| 1    | 0.92               | 0.88           | 0.10            | 0.02             | 0.05           | 0.04           | 0.34          | …   |
| 2    | 0.89               | 0.82           | 0.21            | 0.05             | 0.06           | 0.03           | 0.30          | …   |
| 3    | 0.60               | 0.60           | 0.45            | 0.20             | 0.11           | 0.08           | 0.40          | …   |
| ...  | ...                | ...            | ...             | ...              | ...            | ...            | ...           | ... |

每一行 = 一个 turn，每一列 = 一个测量维度。进行分析：
- 对每个指标和 classifier 的 P(true) 进行相关性计算：corr(metric_x, P(true))。然后就能排序指标的重要性。数值越大越有用/可靠。
- 对每个指标计算 ECE (calibration error)：ECE(metric_x, P(true))。ECE 越小越好 → 表示该指标和 classifier 的信号“对齐度”高。
得到一个表（例子）：

| Metric           | Corr  | ECE  |
| ---------------- | ----- | ---- |
| Lexical_Cosine   | -0.82 | 0.08 |
| Syntax_DepthDiff | -0.60 | 0.10 |
| Sentiment_Dist   | -0.45 | 0.18 |
| DialogueAct_KL   | -0.10 | 0.29 |

- Regression for drift detection


# 7. Clustering

