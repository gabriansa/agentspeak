# AgentSpeak: Integrating ABMs and LLMs

AgentSpeak is a framework that integrates Agent-Based Models (ABMs) with Large Language Models (LLMs), offering a novel approach to simulating complex social dynamics and human decision-making processes.

## Installation
To set up AgentSpeak, ensure you have Python installed along with the necessary libraries.
```
pip install -r requirements.txt
```

Additionally, place the ```.gguf``` model file inside a folder named ```models``` in your main project directory. This file is essential for the LLM component of the simulation.

## Usage
Run simulations using provided configuration and prompt files. Modify these files to tailor the simulation to your specific research needs.
```
python main.py --config_path='path/to/config.json' --prompt_path='path/to/prompt.txt'
```

## Features
- **Integration of LLMs**: Enhances agent decision-making with advanced cognitive capabilities.
- **Flexible Agent Architecture**: Supports diverse and dynamic agent behaviors.
- **In-depth Analysis**: Facilitates comprehensive policy analysis and simulation of complex social interactions.


## Templates/Tutorial
Examples of ```config.json``` and ```prompt.txt``` can be found under the ```examples``` folder
### config.json
The ```config.json``` contains detailed agent definitions, including variables, actions, and states. Each variable can have an initial value and an update rule. Here is a template:

```
{
  "simulation_parameters": {
    "simulation_steps": [Number of steps],
    "update_type": "[synchronous or asynchronous]",
    "seed": [Random seed number],
    "verbose": [true or false],
    "run_name": "[Name of the simulation run]"
  },
  "llm_parameters": {
    "model_name": "[Name of the LLM model]",
    "temperature": [Temperature value],
    "output_parser": "[Type of output parser]"
  },
  "agents_definition": {
    "[Agent Type]": {
      "num_agents": [Number of agents],
      "variables": {
        "[Variable 1]": {
          "initial_value": [Initial value or distribution],
          "update_rule": "[Update rule or null]"
        },
        // More variables...
      },
      "actions": {
        "[Action 1]": "[Description]",
        // More actions...
      },
      "states": {
        "[State 1]": "[Description]",
        // More states...
      }
    },
    // More agent types...
  }
}
```

### Prompt File
In the ```prompt.txt``, it is possible to dynamically reference agent variables and states using a special syntax. For example:
- ```"@self.variable_1@"```: Directly references the ```variable_1``` variable of the agent.
- Conditional statements like ```"@if (self.variable_1<=2): 'excellent'; ...@"``` dynamically change the prompt's content based on the agent's ```variable_1```.

