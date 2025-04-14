# Collaborative Reasoning with Multi-Agent LLMs: A Sandbox Murder Mystery Framework



## Motivation

Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding, generation, and even multi-turn dialogue. However, coordinating multiple LLM agents to collaborate effectively in open-ended, dynamic environments remains a largely unsolved challenge. Real-world tasks often involve incomplete information, evolving beliefs, and the need for strategic communication—conditions rarely captured in current benchmark settings.

We introduce a novel testbed—Who’s the Murderer, a cooperative murder mystery game—in which multiple LLM-based agents interact in a sandbox environment to collectively identify a hidden culprit. The scenario simulates a detective-style investigation involving free-form conversation, evidence gathering, belief updates, deception resistence, and multi-stage reasoning. This environment enables us to study how different LLM architectures and communication strategies influence the agents' ability to coordinate, reason, and succeed under uncertainty.

(In short: The integration of LLMs in collaborative, multi-agent environments presents a unique opportunity to evaluate their reasoning and communication abilities in realistic, dynamic contexts. This research investigates why and how multi-agent systems need to communicate and collaborate effectively, especially when navigating uncertain or incomplete information.)


## Research Question

This thesis seeks to address the following core question:

**How do different LLM architectures and training paradigms affect the collaborative reasoning performance of multi-agent systems in an open-ended, dynamic, and communication-intensive environment?**

Sub questions include:
- Can the prevailing architecture maintain reasoning consistency across multi-stage dialogue?
- Does any prevailing architecture yield better alignment between internal beliefs and communicated plans, and have better deception resistence?
- Does communication between agents measurably improve final task performance (e.g., correctly identifying the murderer)?
- How robust are agent collaborations to missing or misleading evidence?

## Game Framework
We implement a multi-agent framework for Who’s the Murderer, consisting of:
- **Suspect agents (LLMs)**, each given a unique persona and partial access to the game world.
- **A player detective (human or LLM)**, who interviews agents and leads the investigation.
- **Two reasoning stages**, with new information introduced between them, testing the agents' ability to adapt their beliefs and dialogue accordingly.
- **One interview stage**: allow for focused information extraction and belief probing, which introduces a different communication dynamic compared to open group discussions.   
_(We could explore: 1/ Which LLM architectures are more "interrogable" — i.e., more likely to reveal clues under questioning? 2/ Which agents lie or resist interrogation? (Contributing to research on deception detection) 3/ Does multi-agent belief consistency break down during private interactions?)_
- **Game Flow and Stages**:
    - **Prologue**: Players (agents) are introduced and given background information.
    - **1st Round of Crime Scene Investigation**: Agents search for evidence to form hypotheses.
    - **Group Deduction**: Agents discuss findings and collaborate.
    - **2nd Round of Crime Scene Investigation**: New clues are introduced, further complicating the reasoning process.
    - **1-on-1 Detective Interviews**: Agents privately question suspects.
    - **Final Group Deduction and Voting**: A final discussion and vote are held to identify the murderer.
- **Information Dynamics**: Partial observability and evolving evidence challenge agents to reason collaboratively under uncertainty.


## Methodology, Experiment Setup

### Architecture to be compared:
- GPT-style decoder-only models (e.g., GPT-2 / GPT-NeoX)
- BART-style encoder-decoder models (e.g., R2C2 or fine-tuned BART)
- etc..

### Tasks and objectives:
- Identifying the correct murderer
- Reasoning consistency across rounds
- Communication effectiveness

### Evaliation metrics:
Automatic metrics only. No resources for quantitative human evaluation.
- **Task accuracy/Winning rate**: Was the murderer correctly identified?
- **Dialogue quality (fluency, alignment, informativeness)**: Are statements consistent with evidence and internal beliefs?
- **Reasoning coherence**: Does the agent’s belief state evolve plausibly over time?
- **Communication quality**: Assessed via human or automatic metrics (e.g., fluency, informativeness)
- **Belief alignment/coherence** before and after the interview.
- Deception detection
- ? Maybe do a light ablation study? w/o intent grounding, w/o interviews (reinforced deduction, disturbing), w/o dialogue.

### Optional sub-tasks:
- Filtering models to prevent hallucination or contradiction.

## Expected Contributions
- A novel game-based testbed for evaluating multi-agent LLM communication under uncertainty.
- Empirical comparison of LLM architectures in multi-stage reasoning tasks.
- Insights into the effectiveness of communication strategies for belief alignment and coordination.
- Foundations for designing future LLM agents capable of collaborative problem-solving in open environments.

## Related Work
- LLM agents: one vs. multiple
- Test-bed: zero-sum, pure language vs. scarce combination of dynamic environment + strategic reasoning + communication, ...
- Scarce multi-dimentional comparisons between LLMs in complicated gaming environment 


## *NOTES TO MYSELF
- Core: Multi-agent LLM collaboration under uncertainty in communication games
- Sub-tasks:

| Main topics | Sub tasks |
|:---|:---|
| Agent architecture |	GPT vs BART |
| Reasoning alignment|	consistency under multi-stage evidence
| Communication effect|	performance with/without dialogue
| Dialogue control	| with/without intent grounding
| Interaction format|	Free group chat vs 1-on-1 interviews


