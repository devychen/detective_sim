
# Three Detectives in a Room: <br> Investigating Character Consistency and Interactions in Multi-Agent LLM Dialogue

Master thesis for M.A. in Computational Linguitstics at University of Tuebingen.

Proposal [Quick Access](https://github.com/devychen/Thesis_CL/blob/main/Proposal_0601.md).

**Current Status**:

03 Jun, Proposal approved.<br>
01 Jul, Profile and Prompt (with examples) completed. <br>
01 Aug, Two tasks prepared.  <br>
28 Aug, Finalise simulation codes. <br>




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


✅ **Step 1**: Create `venv`. Confirm brew up-to-date: 
`brew update` and install necessary pkgs in terminal:`pip install -r requirements.txt`  

> 🍎 *requirements.txt, _build_venv.txt*.

✅ **Step 2**: Run *prep > gen_profile.py* for character profile generation. Profile generation done. 

> 🍎 Output saved to *profiles* > three character profile yaml files.

✅ **Step 3**: Run *prep > gen_agent_prompt.py* to generate the prompts for agents role-playing. 

> 🍎 Output saved to *prompts > holmes.yaml, marple.yaml, poirot.yaml*.

✅ **Step 4**: Run *prep > reverse_id.py* to do the reverse identification check, model used llama-3.3-70b-Instruct. 

> 🍎 Identity proved. 

✅  **Step 5** Get example lines. Tried using LLM to extract example lines from original works (the series of *gen_examples.py*). But API not working well - duplicate lines and only extremely famous quote. So in the end I mannual read the books and extracted some example lines.

> 🍎 Mannually added example lines into the three prompts at *prompts*.

**`=== STAGE CLOSE. 1 JUL 2025 ===`**

## Workflow - Agent Construction

> LLM-based Assistants [Course Webbook](https://maxschmaltz.github.io/Course-LLM-based-Assistants/) might be helpful, especially the [Multi-agent Environment](https://maxschmaltz.github.io/Course-LLM-based-Assistants/sessions/block2_core_topics/pt1_business/2705/2705.html) part.

✅ **Step 1**: Build the base agent (*agents > base_agent.py*) and the basic construction, do a trial simuation to make sure the basics work.   
> 🍎 *agents > base_agent.py*, output saved as *holmes_agent.py, marple_agent.py, poirot_agent.py*   
> 🍎 ~~*tools* (if any).~~  

[No need]~~Undecided - for collaboration task, should or not including `ReAct` to allow questions between each other? if so, how many questions allowed? 2 each turn?~~

**`=== STAGE CLOSE. 15 JUL 2025 ===`**

## Workflow - Task Preparation, Finalise Setting & Run Simulations

**Step 1**: Use the script from a previous work - [Player](https://github.com/alickzhu/PLAYER/tree/main) - which has mystery game scripts. And mannually extract the information based on my case template.   
选取的三个案件：绝命阳光号（凶手张大副-Zack），罪恶（凶手王村长-William），未完结的爱（凶手苏阳-John Saar）
> 🍎 *cases > case1.yaml, case2.yaml, case3.yaml*

**Step 2**: use GPT to prove that the case extracted works - namely it is solvable
> 🍎 *tests > test_allcases.py*. the results shows it is not only solvable but also murders are correctly identified.

**Step 3**: Based on inspiring works ([Player](https://github.com/alickzhu/PLAYER/tree/main)), build the rule prompt. It contains the common intro and common rules which is to be shared across agents.
>  🍎 *prompts > rule_collab.yaml*.

**Step 4** Run the simulations and get the results, each time stamped. Meanwhile, print out the system prompt, make sure it's not too chaotic.
> 🍎 Output dialogue each turn saved to *data > case1, case2, case3*, within each run a *dialogue.csv* and each agent's prompt output are saved.

~~**======For Debate Task======**~~
~~Step 1: Write a more detailed description on their investigation methods. Use the summarised investigative traits from Lima et al (2025) directly (but rephrased to imperative sentences).~~  
~~> 🍎 prompts > holmes_methods.yaml, poirot_methods.yaml, marple_methods.yaml~~
~~**Step 2**: Complete task prompt~~
~~> 🍎 rules > rule_debate.yaml~~


**`=== (Estimated) STAGE CLOSE. 05 AUG 2025 ===`**

## Workflow - Baseline

**Summary**:
- Establish baseline with a small BERT classification model
- Supervised classifier.
- Four classes: sh, mm, hp, others. (Others: `watson` from Conan Doyle, `japp` and `hastings` from Agatha Christie) **OR** three classes: sh, mm, hp. Whether to include others or not is discussed in *baselin/others_or_not.md*.
- Datasize: holmes ≈ 130k tokens, marple ≈ 148k tokens, poirot ≈ 146k tokens, watson ≈ 118k tokens, hastings ≈ 59k tokens, Japp = 32k.

**Step 1** From original books, using regex to extract the quotes.  
Resources: [sherlock](https://sherlock-holm.es/ascii/), [marple & poirot](https://github.com/oliviachang29/the-complete-works-of-agatha-christie)
> 🍎 *_novels* > contains all the novels in txt format.   
> 🍎 *lines > regex_extract.py*, results as 6 characters' csv in *lines/regex_data* folder.

_**OR [CURRENT]**_: From original books, using LLM (`meta/llama-3.2-3b-instruct`) to extract lines.
> 🍎 *lines > llama_extract.py*.  
> 🍎 *lines > llm_data > each character has a folder containing its lines in csv format, named by source book title.*

**Step 1.5** Performs systematic diagnostics on LLM-extracted dialogue data before any classifier training is attempted.   
- Japp is noisy, delete from the data. Only using the other five.

> 🍎 *baseline > data_diagnosis.py*

**Step 2** Prepare the dataset, preprocessing with cleaning and balancing. It is parameterised to adapt two situation: including "others" or not.

> 🍎 *baseline > prepare_dataset.py*

**Step 3** Use [BERT-cased](https://huggingface.co/google-bert/bert-base-cased). Train the baseline. Parameterised too for two situation accordingly.  
> 🍎 *baselines > train.csv*.  
> 🍎 *baselines > terminal_3class.txt, terminal_4class.txt*.   
> 🍎 Baseline models saved to *models*.  


**`=== STAGE CLOSE. -- 31 Dec 2025 ===`**

## Workflow - Data Analysis, Metrics.

**Step 1** Walk-through of the metrics, confirm its rationale.
> metrics.md


## Timeline
 
31 Dec - Baseline.  
31 Jan - Eval.  
28 Feb - Writing.   
30 Mar - Oral.  


# Notes

[Nvidia models](https://build.nvidia.com/)  
OpenAI [API Reference](https://platform.openai.com/docs/api-reference/chat/create).  
SH dataset, on [kaggle](https://www.kaggle.com/datasets/bharatkumar0925/. sherlock-holmes-collection).  
BBC'S SH script, on [BBC web](https://www.bbc.co.uk/writers/scripts/tv-drama/sherlock/).  
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