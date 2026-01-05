# Abstract

# Content (Skip)

# 1. Introduction

# 2. Background

# 3. Method

## 3.1 Model

## 3.2 Data

## 3.3 Agent Construction

## 3.4 Simulation

## 3.5 Baseline

# 4. Results

# 5. Discussion

# Reference

# Appendix A Case Data

# Appendix B Tables

# Appendix C Prompts



3. Methods / Experimental Setup
3.1 Multi-Agent Role-Play Framework

This study adopts a controlled multi-agent dialogue framework in which three large language model (LLM) agents collaboratively solve fictional murder cases through natural language interaction. Each agent is explicitly instructed to role-play a well-established fictional detective: Sherlock Holmes, Miss Marple, and Hercule Poirot.

The agents interact exclusively through turn-based dialogue. No external coordination mechanism, shared scratchpad, or symbolic planner is provided. All collaboration emerges solely from linguistic interaction. This design ensures that both task performance and character consistency can be evaluated based on observable language use alone.

Each experimental run consists of a fixed maximum number of dialogue rounds (10 turns), with early termination if all three agents converge on the same suspect. At each turn, agents speak in a randomized order to avoid positional bias or fixed conversational dominance.

3.2 Information Asymmetry and Collaborative Constraint

To encourage genuine collaboration rather than independent parallel reasoning, information asymmetry is deliberately introduced. While all agents are given access to global collaboration rules, each agent receives only a partial subset of the case information:

Holmes receives crime scene descriptions and forensic evidence.

Marple receives detailed social and interpersonal information about suspects.

Poirot receives a structured timeline of events.

No agent has direct access to the complete case. Consequently, successful task completion requires information sharing and synthesis through dialogue. This design mirrors real-world collaborative problem-solving and allows analysis of how character-specific reasoning styles affect both cooperation and accuracy.

3.3 Prompt Design and Role Protection

Each agent’s prompt consists of four major components:

Global Collaboration Rules, shared across all agents, specifying conversational constraints (e.g., turn-taking, coherence).

Role-Play Guidelines, defining the canonical reasoning style, personality traits, and investigative methods associated with each detective.

Protective Guidelines, designed to discourage generic assistant behavior, meta-commentary, or stylistic convergence toward a neutral LLM voice.

Task Description, incorporating the agent’s partial case information.

Agents are instructed to respond in well-formed paragraphs, limited in length, and to explicitly state a hypothesis at the end of each turn using a fixed linguistic template (“I believe the murderer is XXX”). This constraint enables automatic extraction of beliefs while preserving naturalistic reasoning in the main body of the response.

3.4 Dialogue Memory and Logging

All surface-level dialogue utterances are stored in a shared conversation history and provided verbatim to agents at each turn. System prompts and internal prompt construction steps are excluded from this memory, ensuring that agents respond only to what other agents explicitly say.

For each experimental run, the system logs:

A complete dialogue transcript with turn number, speaker identity, utterance text, and extracted belief.

A full prompt log for each agent at each turn, enabling prompt-level reproducibility and post-hoc analysis of behavioral drift.

All logs are timestamped and stored in structured formats (CSV and plain text) to support both qualitative inspection and quantitative linguistic analysis.

3.5 Task Completion and Stopping Criterion

After each round, the system checks whether all three agents converge on the same suspect. If unanimous agreement is reached, the simulation terminates early. Otherwise, the dialogue proceeds until the maximum number of turns is reached.

This stopping criterion operationalizes collaborative success as explicit linguistic convergence, rather than latent internal agreement, aligning task evaluation with observable language behavior.

4. Baseline Model: Design and Justification
4.1 Purpose of the Baseline

The baseline model is introduced to distinguish character-driven variation from task-driven or model-intrinsic variation. Specifically, it serves as a reference point for evaluating whether deviations in agent behavior constitute out-of-character (OOC) phenomena or merely reflect generic LLM reasoning under uncertainty.

Without such a baseline, stylistic or reasoning shifts observed in role-playing agents could be misattributed to OOC behavior when they are in fact common across non-role-conditioned models.

4.2 Baseline Model Design

The baseline model uses the same underlying LLM architecture as the role-playing agents but operates without any character conditioning. It is instructed to act as a neutral, cooperative detective assistant whose sole objective is to solve the case efficiently.

Key characteristics of the baseline include:

Access to the same partial information structure as the role-playing agents.

Identical collaboration rules and response length constraints.

No role-play instructions, stylistic constraints, or personality cues.

The baseline participates in the same dialogue environment and produces hypotheses using the same explicit belief template, enabling direct comparison at both linguistic and task-performance levels.

4.3 Justification for Baseline Choice

The baseline is designed to control for three critical factors:

Model Capability Control
By using the same LLM backbone, differences in performance or language use cannot be attributed to model capacity or training data alone.

Task Structure Control
Because the baseline follows the same task rules and information constraints, any observed divergence reflects the presence or absence of role conditioning rather than task complexity.

Linguistic Reference Point
The baseline provides an empirical estimate of how a “default” LLM reasons and communicates in this collaborative detective task, serving as a neutral anchor for identifying character-specific deviations.

4.4 Baseline as an OOC Reference

Out-of-character behavior is operationalized not merely as deviation from a predefined character description, but as convergence toward baseline-like behavior. For example, when a role-playing agent’s language becomes increasingly similar to the baseline in terms of lexical choice, discourse structure, or reasoning patterns, this is interpreted as potential character collapse.

Thus, the baseline does not define correctness or optimality; instead, it defines character-neutrality, against which character consistency can be measured.

5. Relation to OOC Analysis (Bridge Paragraph)

Taken together, this experimental setup allows OOC to be analyzed as a dynamic phenomenon emerging during interaction. By combining controlled role conditioning, information asymmetry, and a neutral baseline reference, the study isolates how character fidelity influences both collaborative dynamics and task outcomes over time.