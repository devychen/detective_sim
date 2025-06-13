
# Three Detectives in a Room: <br> Investigating Character Consistency and Interactions in Multi-Agent LLM Dialogue

Master thesis for M.A. in Computational Linguitstics at University of Tuebingen.

Proposal [Quick Access](https://github.com/devychen/Thesis_CL/blob/main/Proposal_0601.md).

**Current Status**:

03 Jun Proposal approved.<br>
16 Jun Profile and Prompt generated.


## Literature Review

The overlaps of: RPLA (Role Play Language Agents) and Multi-agent simulations.

What have they done so far. 

Why intriguing.

Currently rare studies in combination. Also rare studies do evaluations from a linguistic perspective.


## Profile Generation

## Prompt Generation

```
Extract the key traits that identify the character in the following text and produce a single, cohesive prompt suitable for LLM character simulation or prompt engineering. The output must consist of exactly eight sentences, with two sentences each dedicated to the character’s vocabulary, sentence structure, discourse pattern, and personality.
Do not include any bullet points, lists, formatting, or section headings.
The content of each sentence must be based strictly on the original text; do not invent or infer beyond what is explicitly stated.
Begin the prompt with: "You are {name}. You are {occupation}."
End the prompt with: "Stay in character at all times."
```

## Agent Construction


# Notes

SH dataset, on [kaggle](https://www.kaggle.com/datasets/bharatkumar0925/sherlock-holmes-collection).

BBC'S SH script, on [BBC web](https://www.bbc.co.uk/writers/scripts/tv-drama/sherlock/).

SH txt book, on [github](https://github.com/lucko515/rnn-sherlock-holmes-book/blob/master/datasets/holmes.txt).

HP video drama, [script](https://www.otrr.org/FILES/Scripts_pdf/Hercule%20Poirot/Hercule%20Poirot%2045-02-22%20Case%20of%20Careless%20Client.pdf)

---
**Detective Selection** from *lima et al (2025)*:
- SH (Private investigator): Arthur Conan Doyle, 1887; TV series Sherlock, 2010
    - exemplifies the application of logical reasoning over detailed observation 
    - refined his observational acuity through a broad range of disciplines, focusing only on knowledge essential for his role as a “consulting detective” 
- HP (...): emphasizes psychological profiling
- MM (Amateur detective): Agatha Christie, 1927; TV series Miss Marple, 1984
    - relies on her skill to find parallels between human behavior in her small village and current criminal actions. She frequently makes discoveries through gossip, a daily occupation that suits her temperament and sharp observational skills.
- *The Second Half*, [blog](https://ysymyth.github.io/The-Second-Half/) article emphasise evaluation.