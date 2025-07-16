
# Three Detectives in a Room: <br> Investigating Character Consistency and Interactions in Multi-Agent LLM Dialogue

Master thesis for M.A. in Computational Linguitstics at University of Tuebingen.

Proposal [Quick Access](https://github.com/devychen/Thesis_CL/blob/main/Proposal_0601.md).

**Current Status**:

03 Jun, Proposal approved.<br>
28 Jun, Profile and Prompt generated. <br>
01 Jul, Profile and Prompt (with examples) completed.



## Literature Review

The overlaps of: RPLA (Role Play Language Agents) and Multi-agent simulations.

What have they done so far. 

Why intriguing.

Currently rare studies in combination. Also rare studies do evaluations from a linguistic-based perspective.

## Workflow - Character Profile & Prompt Generation

Inpsired by _de Lima et al (2025)_, I use GPT API to generate the **character profile**, covering (1) vocabulary, (2) lexical features, (3) syntactic features, (4) discourse patterns, (5) investigative method. One paragraph for each aspect. Using a designated **profile prompt**.

Using this profile, have GPT generated the **role-play prompt**. Two sentences for each paragraph (each aspect), with one example for each sentence mannually extracted from the original text. 

For both aotomated generation process using remote API, the temperature  is set up at 0.0 to reduce variability and promote deterministic outputs. 

Then using an open-source LLM (llama) to do a _Validation Via Reverse identification_, to validate these profiles and prompts are valid representation of the respective character. (The LLM can tell who is who from the input)


✅ **Step 1**: confirm brew up-to-date: 
`brew update` and install necessary pkgs in terminal:`pip install -r requirements.txt`  

✅ **Step 2**: Run [gen_profile] for profile generation. Profile generation done. 

✅ **Step 3**: Run [gen_agent_prompt.py] to generate the prompts for agents role-playing. 

✅ **Step 4**: Run [reverse_id.py] to do the reverse identification check, model used llama-3.3-70b-Instruct. Identity proved. 

✅  **Step 5** Extract examples from original works. Run the series of [gen_examples.py]. API not working well, mannual extraction in the end.

**`=== STAGE CLOSE. 1 JUL 2025 ===`**

## Workflow - Agent Construction

> LLM-based Assistants [Course Webbook](https://maxschmaltz.github.io/Course-LLM-based-Assistants/) might be helpful, especially the [Multi-agent Environment](https://maxschmaltz.github.io/Course-LLM-based-Assistants/sessions/block2_core_topics/pt1_business/2705/2705.html) part.

✅ **Step 1**: Build the foundation. The struction, do a trial simuation to make sure the basics work.   
**agents** -> base_agent, individual_agents * 3.   
**prompts** -> individual prompts files * 3.   
**tools** (if any) -> ask_other_questions.   
**tasks** task description files.  

❓ Undecided - for collaboration task, should or not including `ReAct` to allow questions between each other? if so, how many questions allowed? 2 each turn?

**`=== STAGE CLOSE. 15 JUL 2025 ===`**

## Workflow - Task Preparation


**Step 1**: Build real cases and extract the clues. Build the debate task description too.

**Step 2**: Design the case report template.

**`=== STAGE CLOSE. 20 JUL 2025 ===`**

## Workflow - Finalise Setting & Run Simulations

**Step 1**: Based on the task descriptions, finalise adjustments on dialogue and agent coding files.

**Step 2**: Run the simulations and get the results.

**`=== (Estimated) STAGE CLOSE. 31 JUL 2025 ===`**

## Workflow - Baseline

Establish baseline with BERT classification model.

**`=== (Estimated) STAGE CLOSE. 07 AUG 2025 ===`**

## Workflow - Data Analysis, the Metrics.

Do the evaluation.

**`=== (Estimated) STAGE CLOSE. 15 AUG 2025 ===`**

## Workflow - Ablation Study (if needed)

## Workflow - Writing Wrap-Up!

**`=== (Estimated) STAGE CLOSE. 31 AUG 2025 ===`**




# Notes

✅ Poirot [complete pdf](https://dn721604.ca.archive.org/0/items/MegaAgatha/hercule%20poirot-complete%20short%20stories%20-%20agatha%20christie.pdf) and [The murder of Roger Ackroyd](https://github.com/GITenberg/The-murder-of-Roger-Ackroyd_69087/tree/master)

✅ Agatha [complete](https://github.com/oliviachang29/the-complete-works-of-agatha-christie). Have used _Nemesis_.

✅ Holmes [complete txt](https://sherlock-holm.es/html/)

✅ Miss Marple [complete pdf](https://ia801809.us.archive.org/19/items/AgathaBundle/Miss%20Marple_%20The%20Complete%20Short%20-%20Agatha%20Christie.pdf)

MM [A Murder is Announced](https://archive.org/stream/amurderisannounced_201908/A%20Murder%20Is%20Announced_djvu.txt)

[Nvidia models](https://build.nvidia.com/)

OpenAI [API Reference](https://platform.openai.com/docs/api-reference/chat/create)

SH dataset, on [kaggle](https://www.kaggle.com/datasets/bharatkumar0925/sherlock-holmes-collection).

BBC'S SH script, on [BBC web](https://www.bbc.co.uk/writers/scripts/tv-drama/sherlock/).

SH txt book, on [github](https://github.com/lucko515/rnn-sherlock-holmes-book/blob/master/datasets/holmes.txt).

HP video drama, [script](https://www.otrr.org/FILES/Scripts_pdf/Hercule%20Poirot/Hercule%20Poirot%2045-02-22%20Case%20of%20Careless%20Client.pdf)

**Detective Selection** from *de Lima et al (2025)*:
- SH (Private investigator): Arthur Conan Doyle, 1887; TV series Sherlock, 2010
    - exemplifies the application of logical reasoning over detailed observation 
    - refined his observational acuity through a broad range of disciplines, focusing only on knowledge essential for his role as a “consulting detective” 
- HP (...): emphasizes psychological profiling
- MM (Amateur detective): Agatha Christie, 1927; TV series Miss Marple, 1984
    - relies on her skill to find parallels between human behavior in her small village and current criminal actions. She frequently makes discoveries through gossip, a daily occupation that suits her temperament and sharp observational skills.
- *The Second Half*, a [blog](https://ysymyth.github.io/The-Second-Half/) article emphasises evaluation > training is the future trending.


# Reference (APA)

de Lima, E. S., Casanova, M. A., Feijó, B., & Furtado, A. L. (2025). Characterizing the Investigative Methods of Fictional Detectives with Large Language Models. arXiv preprint arXiv:2505.07601.  
[Quick Access](https://arxiv.org/abs/2505.07601)