# debate.py
# 用于构建每个角色的 system prompt

import yaml
from pathlib import Path

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_system_prompt(agent_name: str, template_path: Path) -> str:
    # Lowercase version for file naming
    agent_name_lower = agent_name.lower()

    # Load agent prompt (e.g., prompts/holmes_prompt.yaml)
    character_path = Path(f"prompts/{agent_name_lower}_prompt.yaml")
    character_data = load_yaml(character_path)

    # Load agent methods (e.g., prompts/holmes_methods.yaml)
    methods_path = Path(f"prompts/{agent_name_lower}_methods.yaml")
    agent_methods = load_yaml(methods_path)
    agent_methods_summary = yaml.dump(agent_methods, default_flow_style=False, allow_unicode=True)

    # Load protective constraints (e.g., protective_holmes)
    protective_data = load_yaml(Path("prompts/protective.yaml"))
    protective_key = f"protective_{agent_name_lower}"
    protective_rules = protective_data.get(protective_key, [])

    # Get all other agents for prompt
    all_agents = ["Holmes", "Poirot", "Marple"]
    other_agents = [a for a in all_agents if a != agent_name]
    other_agent_1, other_agent_2 = other_agents

    # Load system prompt template
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # Prepare context for rendering
    from jinja2 import Template
    jinja_template = Template(template)

    system_prompt = jinja_template.render(
        agent_name=agent_name,
        agent_name_lower=agent_name_lower,
        other_agent_1=other_agent_1,
        other_agent_2=other_agent_2,
        character_traits=character_data,
        protective_rules=protective_rules,
        agent_methods_summary=agent_methods_summary
    )

    return system_prompt
