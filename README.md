
# Three Detectives in a Room: <br> Investigating Character Consistency and Interactions in Multi-Agent LLM Dialogue

Master thesis for M.A. in Computational Linguitstics at University of Tuebingen.

Proposal [Quick Access](https://github.com/devychen/Thesis_CL/blob/main/Proposal_0601.md).

**Current Status**:

03 Jun, Proposal approved.<br>
28 Jun, Profile and Prompt generated.


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


**Step 1**: confirm brew up-to-date: ✅ 
```
brew --version
brew update
```
**Step 2**: Then run [gen_profile] for profile generation. Profile generation done. ✅

**Step 3**: Using the [gen_agent_prompt.py] to generate the prompts for agents role-playing. ✅

**Step 4**: Use [reverse_id.py] to do the reverse identification check, model used llama-3.3-70b-Instruct. Identity proved. ✅
```
Detective Identification Results:

detective1: Sherlock Holmes
detective2: Hercule Poirot
detective3: Miss Marple
```




## Workflow - Agent Construction

LLM-based Assistants [Course Webbook](https://maxschmaltz.github.io/Course-LLM-based-Assistants/) might be helpful, especially the [Multi-agent Environment](https://maxschmaltz.github.io/Course-LLM-based-Assistants/sessions/block2_core_topics/pt1_business/2705/2705.html) part.







# Notes

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