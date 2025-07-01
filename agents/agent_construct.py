from langchain.agents import Agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import yaml

# Load the character prompts
def load_character_prompt(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Create agent constructor function
def create_detective_agent(character_name, llm):
    # Load the appropriate YAML file
    if character_name.lower() == "sherlock holmes":
        prompt_data = load_character_prompt("sh_prompt.yaml")
    elif character_name.lower() == "hercule poirot":
        prompt_data = load_character_prompt("hp_prompt.yaml")
    elif character_name.lower() == "miss marple":
        prompt_data = load_character_prompt("mm_prompt.yaml")
    else:
        raise ValueError("Unknown character")
    
    # Build the system message from prompt data
    system_message = "\n".join([item['description'] for item in prompt_data if 'description' in item])
    examples = "\n".join([item['example'] for item in prompt_data if 'example' in item])
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        input_variables=["input", "chat_history"],
        template=f"""System: You are {character_name}. Stay in character at all times.
{system_message}
Examples of your speech:
{examples}

Chat History:
{{chat_history}}

Human: {{input}}

{character_name}:"""
    )
    
    # Create memory for the agent
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    # Create LLM chain
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        verbose=True
    )
    
    # Create the agent
    agent = Agent(
        llm_chain=llm_chain,
        tools=[],  # Add tools if needed for investigation
        verbose=True
    )
    
    return agent