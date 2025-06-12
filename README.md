
Proposal available [here]().

## Character Profile Generation 

- Use designed prompt to generate character profiles. A method employed from *Lima et al (2025)*.
    - API required
```
======= INVESTIGATION METHOD =======

Prompt. Task: Generate a concise and formal description to distinguishing characteristics of the investigative method used by the fictional detective D_name.

Requirements:
- Base your description of the investigative method used on stories where D_name is the protagonist or a principal investigator.
- Focus solely on the investigative approach, strategies, and distinguishing features that define this detective's method of solving cases.
- Consider only the distinguishing characteristics that set the investigative method of D_name apart from those of other fictional detectives.
- Do not include biographical details, story summaries, or references to specific cases.
- Structure the response as a single paragraph, without bullet points or numbered lists.
- Do not include any introductory or concluding sentences outside of the description itself.
- Limit the response to a maximum of 5 sentences.
```
Sherlock Holmes’s **investigative method** is distinguished by its rigorous application of deductive reasoning, in which general conclusions are drawn from specific observations through logical inference. He systematically gathers empirical data through meticulous attention to physical detail, often relying on minute, seemingly trivial evidence overlooked by others. Holmes integrates scientific experimentation and specialized knowledge—particularly in chemistry, anatomy, and handwriting analysis—to test hypotheses and eliminate improbabilities. He emphasizes the importance of objective observation free from preconception, often reconstructing events through inference before engaging with witnesses or suspects. This methodical, evidence-first approach contrasts sharply with intuition-driven or confession-based techniques employed by many other fictional detectives.
```
======= VOCABULARY ======= 

Prompt. Task: Generate a concise and formal description to distinguishing characteristics of the vocabulary used by the fictional detective D_name.

Requirements:
- Base your description of the vocabulary used in stories where D_name is the protagonist or a principal investigator.
- Avoid discourse-level, focus solely on the lexical-level and dinsinguishing features that would be able to define this detective's personality from the linguistic perspective.
- Consider only the distinguishing characteristics that set the vocabulary of D_name apart from those of other fictional detectives.
- Do not include biographical details, story summaries, or references to specific cases.
- Structure the response as a single paragraph, without bullet points or numbered lists.
- Do not include any introductory or concluding sentences outside of the description itself.
- Limit the response to a maximum of 5 sentences.
```

Sherlock Holmes’s **vocabulary** is characterized by a precise, elevated lexicon that reflects his analytical rigor and scientific orientation, often incorporating terms from disciplines such as chemistry, medicine, criminology, and classical music. His word choice favors Latinate and polysyllabic constructions, contributing to a formal and authoritative tone. He frequently employs rare or archaic words, distinguishing his speech with a marked erudition uncommon among fictional detectives. The vocabulary is functionally instrumental, prioritizing clarity in logical exposition while avoiding emotional or colloquial expressions. This lexical precision, coupled with technical specificity, reinforces his identity as a detached and methodical reasoner.

```
======= SENTENCES ======= 

Prompt. Task: Generate a concise and formal description to distinguishing characteristics of the sentences used by the fictional detective D_name.

Requirements:
- Base your description of the sentences used in stories where D_name is the protagonist or a principal investigator.
- Avoid lexical or discourse level. Focus on syntactic, structural and distinguishing features that define this detective's personality from the linguistic perspective.
- Consider only the distinguishing characteristics that set the sentences spoken by D_name apart from those of other fictional detectives.
- Do not include biographical details, story summaries, or references to specific cases.
- Structure the response as a single paragraph, without bullet points or numbered lists.
- Do not include any introductory or concluding sentences outside of the description itself.
- Limit the response to a maximum of 5 sentences.

```

The **sentences** spoken by Sherlock Holmes are structurally complex, often featuring multiple subordinate clauses that mirror the layered nature of his reasoning. He frequently employs hypotactic constructions, allowing him to express causality, condition, and inference within a single, tightly controlled syntactic unit. His use of declarative sentences predominates, conveying authority and finality, while interrogatives are typically rhetorical, serving to guide or emphasize rather than solicit information. Passivization appears selectively, often to foreground actions or outcomes over agents, reinforcing his analytical detachment. This syntactic precision and control set his speech apart from that of other fictional detectives who tend toward more conversational, fragmented, or emotionally inflected sentence structures.

```
======= DISCOURSE PATTERNS ======= 

Prompt. Task: Generate a concise and formal description to distinguishing characteristics of the discourse pattern of the fictional detective D_name.

Requirements:
- Base your description of the discourse pattern used in stories where D_name is the protagonist or a principal investigator.
- Avoid lexical-level, focus on discourse patterns and distinguishing features that would be able to define this detective's personality from the linguistic perspective.
- Consider only the distinguishing characteristics that set the discourse patterns of D_name apart from those of other fictional detectives.
- Do not include biographical details, story summaries, or references to specific cases.
- Structure the response as a single paragraph, without bullet points or numbered lists.
- Do not include any introductory or concluding sentences outside of the description itself.
- Limit the response to a maximum of 5 sentences.
```

Sherlock Holmes’s **discourse pattern** is marked by a didactic and monologic structure, frequently unfolding as extended explanatory sequences in which he reconstructs chains of reasoning with minimal interruption. His speech exhibits a high degree of coherence and logical progression, often building arguments through successive deductive steps that guide the listener toward an inescapable conclusion. He frequently withholds conclusions until a climactic reveal, creating a rhetorical asymmetry that emphasizes his intellectual dominance. Turn-taking in dialogue is typically controlled by Holmes, who limits others’ contributions to questions or brief confirmations, reinforcing his role as the primary epistemic authority. This pattern of structured exposition, strategic withholding, and asymmetric dialogue sets him apart from more conversational or collaboratively styled fictional detectives.

```
======= PERSONAL TRAITS ======= 

Prompt. Task: Generate a concise and formal description to distinguishing characteristics of the personal traits of the fictional detective D_name.

Requirements:
- Base your description of the personality traits used in stories where D_name is the protagonist or a principal investigator.
- Focus on individual and distinguishing features that would be able to define this detective's personality.
- Consider only the distinguishing personal traits that set the discourse patterns of D_name apart from those of other fictional detectives.
- Do not include biographical details, story summaries, or references to specific cases.
- Structure the response as a single paragraph, without bullet points or numbered lists.
- Do not include any introductory or concluding sentences outside of the description itself.
- Limit the response to a maximum of 5 sentences.
```

Sherlock Holmes exhibits a **personality** defined by intellectual detachment, intense rationalism, and a marked preference for solitude, distinguishing him from more emotionally driven or socially engaged fictional detectives. He demonstrates a compulsive need for mental stimulation, often appearing indifferent or irritable when faced with matters lacking analytical challenge. His interpersonal manner is typically terse, aloof, and pragmatic, with limited tolerance for emotional expression or social conventions. Holmes's demeanor conveys self-assuredness and precision, often bordering on arrogance, reinforced by a consistent focus on logic over empathy. These traits collectively construct a persona centered on cognitive mastery and emotional restraint, setting him apart through an almost clinical approach to both human behavior and investigative work.

# Character Prompt

```
Extract the key traits that identify the character in the following text and produce a single, cohesive prompt suitable for LLM character simulation or prompt engineering. The output must consist of exactly eight sentences, with two sentences each dedicated to the character’s vocabulary, sentence structure, discourse pattern, and personality.
Do not include any bullet points, lists, formatting, or section headings.
The content of each sentence must be based strictly on the original text; do not invent or infer beyond what is explicitly stated.
Begin the prompt with: "You are {name}. You are {occupation}."
End the prompt with: "Stay in character at all times."
```


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