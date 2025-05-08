# When LLMs Roleplay: A Dialogue-Centric Study of LLM-Based Multi-Agent Narrative Drift
*Text-based Roleplay as Dialogue Simulation: Investigating Narrative Deviation in Multi-Agent LLM Systems*


# Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities in simulating human-like dialogue, role-based interaction, and even narrative construction. Recent advances in multi-agent systems composed of multiple LLMs show potential for emergent, complex conversational dynamics. These systems mimic social interactions in virtual environments, raising questions about coherence, agency, and control in multi-agent storytelling.

Inspired by game NPCs and interactive fiction, this project investigates how LLM agents with predefined personas and goals interact with one another, and how the introduction of a human-like player agent might cause divergence from the originally intended narrative or goals. By adopting a text-based framework, this research centers linguistic analysis over game mechanics, aligning with the goals of computational linguistics.



# Hypothesis

1. How do LLM agents collaboratively construct and maintain a shared narrative space through dialogue?

2. To what extent does the introduction of a player agent (human or LLM-controlled) cause narrative deviation?

3. What linguistic strategies (e.g., speech acts, implicatures, framing) contribute to changes in agent behavior or task reorientation?

4. How consistent are agents' personas across multi-turn dialogues, especially under narrative perturbation?

# Experiment Setup

No visual environment needed - all interaction occurs through turn-based textual dialogue (like a chatroom, forum, MUD-style RPG)

# Methodology and Experiment Design

### System Architecture



- Agents: 2–3 LLM-based NPC agents with assigned personas, background stories, and goals (e.g., a suspicious guard, an ambitious merchant, a secretive healer).

- Player Agent: A fourth participant representing a human-like player, controlled either manually or via another LLM.

- Interaction Mode: Turn-based text dialogue in a controlled environment, simulating a forum/chat-like conversation. All communication occurs through natural language only.

### Experimental Design

Two main conditions:

1. Baseline (Control): NPC agents interact among themselves to pursue their goals without any external influence.

2. Player Intervention (Experimental): A player agent is introduced after the NPC narrative is initialized, and the interaction continues.

Each session will be logged and analysed for:

- Narrative coherence and divergence

- Goal shifts and task reassignments

- Persona consistency

- Discourse structure

### Data Collection & Annotation

All dialogues will be recorded and annotated for:

- Speech act types (e.g., assertion, request, proposal, threat)

- Discourse moves (e.g., elaboration, contradiction, question-response pairs)

- Goal statements and changes

- Linguistic triggers of deviation

Automated tagging (via LLMs or rule-based heuristics) will be supplemented by manual analysis on selected sessions.




# Related Work