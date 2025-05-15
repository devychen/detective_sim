# Notes For Thesis

- Paper access from [Overleaf](https://www.overleaf.com/read/fbhjtzxsnzjb#78ecdc) (View only).
- Proposal access from [Notion](https://www.notion.so/Master-Thesis-Proposal-1505-1ed738761bc880fba492ef0200a6e7dc)(permission required)

[ChatDev](https://github.com/OpenBMB/ChatDev?tab=readme-ov-file)  

[AgentSims](https://github.com/py499372727/AgentSims)

# Meeting Notes 150425

- Game environment/testbed: **Jubensha**
    - Code available [here](https://github.com/jackwu502/ThinkThrice)
- Research Question(s): Influence of LLM-based agent architecture on gaming performance
- Evaluation:
    1. Ablation study
    2. Automatic metrics:
        - On game: task performance/win rate
        - On agent: 
            - Strategy quality (concealment, persuasiveness, etc..,)
            - Language quality (fluency, alignment, informativeness)
            - Reasoning coherence: Does the agent’s belief state evolve plausibly over time?
            - Character consistency: 
            - Belief cohenrence: before and after the interview.
            - Dialogue coherence: Are statements consistent with evidence and internal beliefs?
            - 
- TBC: Build own agent architecture or comparing exisiting works?

### Next steps
1. Finish proposal
2. Read the codes (and according paper), confirm plausibility of intergrating customised agent, vanilla agent (e.g. ReAct)


# 010425 To Polina: About the Papers
1. [Xu et al., 2024](https://arxiv.org/abs/2403.10249) and [Guo et al., 2024](https://arxiv.org/abs/2402.01680) are the two review papers that I found most useful in this field (i.e. LLM-based multi-agent interactive system). There might be other ones but they are sufficient so far. They summaried the workflow, framework, architecture, purposes of most representative previous works. They also have Git repos where all works mentioned could be accessed. For your convenient here're the links: [Xu.ea](https://github.com/BAAI-Agents/GPA-LM), [Guo.ea](https://github.com/taichengguo/LLM_MultiAgents_Survey_Papers) 
2. Three possibly most relevant games and respective works are listed below. They are all communicative games with high requirements on cooperation, collaboration, competition and heavily rely on natural languages - a key feature of LLM agents. 
- Avalon: [Paper 1](https://arxiv.org/pdf/2310.14985), [paper 2](https://arxiv.org/pdf/2310.05036), [paper 3](https://arxiv.org/pdf/2310.01320)
- Diplomacy: [Paper 1](https://arxiv.org/abs/2310.08901), [Paper 2](https://www.science.org/doi/pdf/10.1126/science.ade9097?casa_token=AB3PXQnKr8YAAAAA:pJO8TUkmbEUH77IhRcn-4r9PpxQc0jRgKokE3ElhmFvAhyTdjjS8aHOgJ_ViH_BnJwMDtTqdMmJgug) 
- Werewolves: [Paper 1](https://arxiv.org/abs/2310.18940), [Paper 2](https://arxiv.org/abs/2309.04658v1)

### Questions

Many systems were designed in different frameworks, architectures and for different purposes. So how about evaluating them in one setting which have most elements covered. Then to discover which is the best and why. These elements might include:
- Deceptive environment.
- Input from both environment and agent interactions. **Question** Too hard???
- Evaluation of success rate, and dialogue analysis.
- A more dynamic environment where agents are allowed to interrupt each other. (**Question** Or focus only on this? I rarely find any work on this, they are all sequential - complete thinking then act).
- Agents have characters, have different stands (rivals).

#### Wondering...
1. Too big???
2. Choice of games
- Rare research on LLM-based multi-agent using Cluedo.
- Can we use a novel setting, not an existing one?
- Game should be used for specific purpose.
3. Choises of baseline
4. Word count
5. I can't tell the clear difference between previous works, how to categories them in terms of underlying architecture etc.

**Variable：Architecture**
- Transformer decoder-only
- Transformer encoder-decoder
- Non Transformer（e.g SSM）

**Variable：Training Paradigm**
- pre-trained + supervised fine-tuning
- w/o reinforcement learning (self-play, RLHF)
- online interaction training

**Variable：agent set up**
- single-agent vs multi-agent
- shared model vs specialized agent


# Meeting Notes 160125

### Todo

- Find a specific task !!!!! brainstorming some tasks / benchmarks
- What would be interesting within the architecture of the agents.
- Read some papers in details. 5-10 different ones.
- Proposal!!!!! (to Prof. Franke)

### Expected timeline
- Proposal in March
- Progress April, May, June
- Submit Jul
- Oral exam Jul

