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
> *tests > test_allcases.py*

- Before running the multi-agent collaborative experiments, we conducted a zero-shot solvability verification using a single, non-role-based LLM. The model was provided with the complete case description and a minimal analytical instruction, without examples or interaction. The purpose of this step was to ensure that each case admitted a coherent, reasoned solution independent of character constraints or dialogue dynamics.

目前不是百分百的正确率.(需要能100%正确解决吗？还是证明有一定正确率即可？是的话多少正确率？)
而且这里用的是gpt，collab用的是llama，说实话llama正确率是要比gpt低一点

### Baseline 2 - Zero Collaboration
Case solving task with character prompts but without collaboration, to verify WHAT???
> *tests > test_allcases_character_controlled.py*

同样，不是百分百的正确率。



### Model-Based Baseline - BERT Classifier Model

> Operationalize OOC as a probabilistic drift away from canonical literary style, measured by a supervised classifier trained on original texts, and to analyze whether such drift increases over interaction time.

**Legitimacy**: 
1. Original texts comes from orginal books, also extraction method (LLM prompt)
2. Accuracy/Macro-F1 results 
    - *The classifier achieves moderate but reliable performance (accuracy = 0.76, macro-F1 = 0.76), which is sufficient for its role as a reference model rather than an oracle. The classifier performance is sufficient to establish legitimacy as a baseline. It clearly outperforms chance, shows balanced per-class performance, and produces interpretable confusions consistent with literary similarity, which is appropriate for measuring probabilistic character drift rather than perfect identification.*.

```
===CURRENT RESULTS===
Evaluating on TRAIN set (diagnostic)...
Train accuracy: 0.8908

Evaluating on TEST set...
Test accuracy: 0.7598
Test macro-F1: 0.7612

Confusion Matrix:
[[374  27  37]
 [ 26 325  94]
 [ 52  94 345]]


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

|Dimension|Metrics|
|--|--|
Desciptive|Average task accuracy & CI
Model|Predicted probability & CI; Average Brier Score & its Linear Regression
Lexical|




# 1. Descriptive

1.1 Average task accuracy.（我需要跑case多少次？我记得我们说需要调整）
- Compare baseline 1 to baseline 2 - if it is lower, role-playing has negative impact.
- Compare to baseline 1 - if it is lower, ~above
- Compare to baseline 2 - if it is lower, collaboration has negative impact.

1.2 Its Confidence Interval (CI)
- Use for ...?

# 2. Classifier model

2.1 Extract the **predicted probability** of the correct class and **plot mean** over turns over simulations.

- CIs over simulations

2.2 Calculate average **Brier score** (mean squared error of predicted probabilities) over turns (in order to estimate whether there is a significant increase or decrease trend).
- Fit **linear regressions** to the scores over turns.

哪个score? brier?

# 3. Lexical
3.1 