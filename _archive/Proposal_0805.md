# Master's Thesis Proposal

**Title:**  
**Roleplay as Reasoning: Investigating Persona Consistency and Goal-Oriented Collaboration in Multi-Agent LLM Dialogue**

**Student:** Yifei Chen  
**Program:** M.A. in Computational Linguistics  
**Advisor:** Dr. Polina Tsvilodub 💓   
**Date:** 08 May, 2025

---

## 1. Background and Motivation

Large Language Models (LLMs) like GPT-4 have demonstrated impressive abilities in dialogue generation, including role-based conversation and multi-turn interaction. However, their ability to maintain consistent personas and effectively collaborate in goal-oriented tasks within multi-agent roleplay settings remains underexplored, especially from a discourse and linguistic interaction perspective.

This thesis focuses on simulating multi-agent dialogue in roleplay scenarios, where each agent represents a unique persona with its own communication style, motivations, and preferences. The goal is to analyze how LLMs handle persona consistency and cooperative behavior during task-oriented conversations.

---

## 2. Questions (Exploratory Study)

`To be defined in details. Literature needed.`

**Persona Consistency** 
- How do LLM agents express and maintain persona traits across multi-turn interactions in cooperative, multi-agent roleplay scenarios?

**Collaboration Strategy** 
- What patterns of collaboration emerge in multi-agent LLM dialogues, and how are task roles, planning, and decision-making linguistically negotiated between agents?

❓❓❓`tbc, if needed` How do different prompt engineering strategies (e.g., explicit vs implicit persona instructions) affect the consistency and quality of the dialogue?

---

## 3. Scope and Limitations

**Scope:**
- Up to **3 simulated agents**, each with a clearly defined persona.
- Tasks will be simple, narrative-driven roleplay scenarios (inspired by narrative-based detective game settings).
- Focus on **language use, dialogue structure, and cooperation** rather than full game logic or dynamic environments, model architectures or training schemes/algorithms.

**Limitations:**
- No large scale **human evaluation** involved. (Maybe some hands on analysis)
- Focus on **text-based analysis** only, without multimodal input/output.

---

## 4. Methodology

### 4.1 Scenario and Persona Design

- Design 2–3 narrative scenarios involving cooperation (e.g., "decide how to rescue a trapped ally", "distribute limited resources during an escape", etc.).
- Create detailed persona descriptions for each agent:
  - *Example*: A risk-taking rogue, a cautious cleric, a pragmatic leader.
- Include personality traits (e.g., emotional tone, preferred strategies, relationship tendencies).

### 4.2 Dialogue Simulation

- Use GPT-4 (or similar) to simulate dialogues:
  - Rotate turns between agents by manually feeding their previous history and persona info.
  - Each agent only “sees” what it should logically know based on the context.
- Run each scenario multiple times to assess variation and consistency.

### 4.3 Analysis Metrics

| Aspect               | Description                                           | Method                                              |
|----------------------|-------------------------------------------------------|-----------------------------------------------------|
| Persona Consistency  | Does language match persona traits?                  | Lexical analysis, style markers, qualitative coding |
| Cooperation Strategy | Do agents show negotiation, planning, etc.?          | Discourse act tagging, turn-taking patterns         |
| Goal Completion      | Is the task successfully completed?                  | Manual evaluation based on scenario criteria        |
| Prompting Impact     | Does prompt detail affect quality/consistency?       | Comparison across different prompt styles           |

### 4.4 Tools

- **Model**: GPT-4 via OpenAI API (or open-source alternative)
- **Text Analysis**: spaCy, LIWC, custom Python scripts
- **Annotation**: Manual tagging?

---

## 5. Expected Outcomes

- A better understanding of how LLMs maintain character identity in extended dialogue.
- Insights into emergent cooperation behavior between AI agents.
- A small but robust case-study framework for analyzing multi-agent interaction linguistically.

---

## 6. Timeline

| Timeframe           | Task                                                  |
|---------------------|--------------------------------------------------------|
| Month 1 (May)       | Environment Setup, Scenario Design    |
| Month 2 (June)      | Run simulations, refine prompting strategies           |
| Month 3 (July)      | Data analysis, metric annotation                      |
| Month 4 (August)    | Write thesis draft, incorporate feedback                       |
| Month 5 (September) |   Finalize and submit             |
---

## 7. References (TBC)

- Zhang, S., Dinan, E., Urbanek, J., et al. (2018). *Personalizing Dialogue Agents: I have a dog, do you have pets too?*  
