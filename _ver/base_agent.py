# agents/base_agent.py
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableSequence
import yaml

class DetectiveAgent:
    def __init__(self, name: str, prompt_path: str, llm: BaseLanguageModel):
        self.name = name
        self.llm = llm
        self.prompt = self._build_prompt(prompt_path)
        self.chain = self.prompt | self.llm  # 新写法：组合 Prompt 和 LLM. <<- self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def _build_prompt(self, path: str) -> PromptTemplate:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        descriptions = []
        for item in data:
            if 'description' in item:
                descriptions.append(item['description'])
            if 'instruction' in item:
                descriptions.append(item['instruction'])

        prompt_text = "\n".join(descriptions) + "\n\n{input}"
        return PromptTemplate.from_template(prompt_text) # <<- return PromptTemplate(template=prompt_text, input_variables=["input"])

    # def run(self, input_text: str) -> str:
    #    return self.chain.invoke({"input": input_text})  # 新写法：invoke 而不是 run <<- return self.chain.run(input=input_text)
    
    # 如果只返回生成内容
    def run(self, input_text: str) -> str:
        response = self.chain.invoke({"input": input_text})
        return response.content  # 只返回纯文本内容
