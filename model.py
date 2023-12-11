import csv
import json
import os
import re
import shutil
from collections import defaultdict
from pprint import pprint
import argparse
import random

from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np
from langchain.llms import LlamaCpp
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from llama_cpp.llama import LlamaGrammar

from gbnf_compiler import *
from utils import generate_prompt_template, get_initial_value, string_to_python_function

# Global tracker for categorical distributions
categorical_distribution_tracker = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

class MyAgent(Agent):
    def __init__(self, unique_id, model, agent_data, agent_name):
        super().__init__(unique_id, model)

        self.name = agent_name
        self.unique_id = unique_id

        self.current_action = None
        self.current_reasoning = None
        self.past_action = None
        self.past_reasoning = None

        # Generate initial variables values for an agent
        self.agent_variables = self.generate_initial_values(agent_data["variables"])
            
        # Generate initial state values for an agent
        self.agent_states = {key: value.lower() if isinstance(value, str) else value for key, value in agent_data["states"].items()}

        # Get the possible actions
        self.agent_actions = {key: value.lower() if isinstance(value, str) else value for key, value in agent_data["actions"].items()}

        # Initialize all variables
        agent_vars = {var: self.agent_variables[var]['value'] for var in self.agent_variables}
        agent_vars_init = {var+"_init_value": self.agent_variables[var]['init_value'] for var in self.agent_variables}
        external_vars =  {'agent_id': self.unique_id}

        self.all_vars = {**agent_vars, **agent_vars_init, **self.agent_actions, **self.agent_states, **external_vars}
        self.all_vars["current_step"] = self.model.schedule.steps
        self.all_vars["current_action"] = self.current_action
        self.all_vars["current_reasoning"] = self.current_reasoning
        self.all_vars["past_action"] = self.past_action
        self.all_vars["past_reasoning"] = self.past_reasoning

        # self.save_step_data()

    def generate_initial_values(self, data_dict):
        values = {}
        for var, properties in data_dict.items():
            if isinstance(properties["initial_value"],(int, float, str)):
                init_value = properties["initial_value"]
            elif properties["initial_value"]["distribution_type"] == "categorical":
                init_value = get_initial_value(properties["initial_value"], var, self.name, categorical_distribution_tracker)
            else:
                distribution_properties = properties["initial_value"]
                init_value = get_initial_value(distribution_properties)
            values[var] = {"value": init_value, "init_value": init_value, "update_rule": properties["update_rule"]}
        return values

    def update_variables(self):
        def update_values(source_dict):
            class_vars = defaultdict(list)
            for agent in self.model.schedule.agents:
                for var in agent.all_vars:
                    class_vars[agent.name + "_" + var].append(agent.all_vars[var])

            new_values = {}
            for var in source_dict:
                update_rule = source_dict[var]['update_rule']
                if update_rule:
                    string_func = string_to_python_function(self, update_rule)
                    merged_vars_class = {**self.all_vars, **class_vars}
                    exec(string_func, merged_vars_class)
                    new_values[var] = merged_vars_class['func']()

            for var in new_values:
                source_dict[var]['value'] = new_values[var]

        update_values(self.agent_variables)

        agent_vars = {var: self.agent_variables[var]['value'] for var in self.agent_variables}
        agent_vars_init = {var+"_init_value": self.agent_variables[var]['init_value'] for var in self.agent_variables}
        external_vars =  {'agent_id': self.unique_id}

        self.all_vars = {**agent_vars, **agent_vars_init, **self.agent_actions, **self.agent_states, **external_vars}  
        self.all_vars["current_step"] = self.model.schedule.steps    

    def generate_prompt(self):
        with open(self.model.prompt_path, 'r') as file:
            user_prompt = file.read()
        
        class_vars = defaultdict(list)
        for agent in self.model.schedule.agents:
            for var in agent.all_vars:
                class_vars[agent.name + "_" + var].append(agent.all_vars[var])

        # Find all patterns that match @...@ and replace them
        matches = re.findall(r'\@(.*?)\@', user_prompt)
        for key in matches:
            string_func = string_to_python_function(self, key)
            merged_vars_class = {**self.all_vars, **class_vars}
            exec(string_func, merged_vars_class)
            var_value = merged_vars_class['func']()
            if isinstance(var_value, float):
                var_value = round(var_value, 2)
            user_prompt = user_prompt.replace(f"@{key}@", str(var_value))

        self.user_prompt = user_prompt

    def ask_llm(self):
        # Extract the actions from the dictionary
        action_list = list(self.agent_actions.values())

        # Format the actions into a string with 'or' before the last action
        action_options = ', '.join(action_list[:-1]) + ', or ' + action_list[-1] if len(action_list) > 1 else action_list[0]
        
        sys_prompt = (
            "Carefully analyze the query to understand its context and requirements. "
            f"Choose the most appropriate action from the following options: {action_options}. "
            "In your response, clearly specify the chosen action and provide a detailed step-by-step explanation. "
            "Your explanation should be thoughtful and based on the information available in the query, focusing on accuracy and relevance. "
            "Make sure to address the specific nuances of the query, offering insights that are both precise and informative. "
        )

        if self.model.output_parser == "grammar":
            _input = generate_prompt_template(model=self.model, sys_prompt=sys_prompt, user_prompt=self.user_prompt)

            output_template = "- Action to take: {{action}}"

            valid_actions = [action.lower() for action in [action for action in self.agent_actions.values()]]
            actions = MultipleChoice('action', valid_actions)
            # c = GBNFCompiler(output_template, { 'action': actions, 'reason': StringRule() })
            c = GBNFCompiler(output_template, { 'action': actions })

            done = False
            while not done:
                output = self.model.llm(prompt=_input.to_string(), grammar=LlamaGrammar.from_string(c.grammar(), verbose=False))
                try:
                    result = c.parse(output)
                    action = result["action"]
                    reasoning = "r" #result["reason"]
                    done = True
                except Exception as e:
                    print(f"An exception occurred: {e}")

        elif self.model.output_parser == "schema":
            response_schemas = [
                ResponseSchema(name="explanation", description="explanation for chosen action"),
                ResponseSchema(name="action", description="chosen action")
            ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()

            _input = generate_prompt_template(model=self.model, sys_prompt=sys_prompt, user_prompt=self.user_prompt, format_instructions=format_instructions)
            
            done = False
            valid_actions = [action.lower() for action in [action for action in self.agent_actions.values()]]
            while not done:
                output = self.model.llm(prompt=_input.to_string())
                try:
                    result = output_parser.parse(output)
                    reasoning = result["explanation"]
                    action = next((action.lower() for action in valid_actions if action.lower() in result["action"].lower()), None)
                    if action is None:
                        raise ValueError("No matching option found in the output.")
                    done = True
                except Exception as e:
                    print(f"An exception occurred: {e}")


        if self.model.verbose:
            print(_input.to_string())
            print(output)

        return action, reasoning

    def step(self):
        self.generate_prompt()
        self.current_action, self.current_reasoning = self.ask_llm()

        self.all_vars["current_step"] = self.model.schedule.steps
        self.all_vars["current_action"] = self.current_action
        self.all_vars["current_reasoning"] = self.current_reasoning

        self.past_action = self.current_action
        self.past_reasoning = self.current_reasoning
        self.all_vars["past_action"] = self.past_action
        self.all_vars["past_reasoning"] = self.past_reasoning

        self.save_step_data()

        if self.model.update_type == "asynchronous":
            self.update_variables()     

    def save_step_data(self):
        keys_to_save = ['current_step', 'agent_id', 'current_action']
        keys_to_save.extend([key for key in self.agent_variables])
        keys_to_save.append('current_reasoning')
        csv_file = self.model.data_dir + self.model.run_name + "_" + self.name + '_data.csv'
        with open(csv_file, 'a', newline='') as file:  # 'a' for append mode
            writer = csv.writer(file)

            if os.stat(csv_file).st_size == 0:
                writer.writerow(keys_to_save)

            # Write data in the same order as the headers
            writer.writerow([self.all_vars.get(key, '') for key in keys_to_save])

    
class Simulation(Model):
    def __init__(self, config_path, prompt_path, seed=None):
        super().__init__(seed)
        
        self.prompt_path = prompt_path

        # Load config data
        with open(config_path, 'r') as file:
            self.config_data = json.load(file)

        # Initialize simulation
        self.simulation_steps = self.config_data["simulation_parameters"]["simulation_steps"]
        self.seed = self.config_data["simulation_parameters"]["seed"]
        self.update_type = self.config_data["simulation_parameters"]["update_type"]
        self.verbose = self.config_data["simulation_parameters"]["verbose"]
        self.run_name = self.config_data["simulation_parameters"]["run_name"]
        self.schedule = RandomActivation(self)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Initialize LLM
        self.model_name = self.config_data["llm_parameters"]["model_name"]
        self.temperature = self.config_data["llm_parameters"]["temperature"]
        self.template = self.config_data["llm_parameters"]["template"]
        self.output_parser = self.config_data["llm_parameters"]["output_parser"]
        if self.model_name == "GPT":
            self.output_parser = "schema"
            self.llm = OpenAI(temperature=self.temperature) # = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)
        else:
            self.llm = LlamaCpp(
                model_path="models/"+self.model_name,
                n_gpu_layers=-1,
                n_batch=2^9,
                temperature=self.temperature,
                verbose=False,
                n_ctx=800,
            )
        
        # Initialize data collection
        self.data_dir = 'data/'
        os.makedirs(self.data_dir, exist_ok=True)

        agents = self.config_data["agents_definition"]
        for agent_name in agents:
            file_name = self.data_dir + self.run_name + "_" + agent_name + '_data.csv'
            with open(file_name, 'w') as file:
                pass
      
        # Initialize agents
        self._initialize_agents()

    def _initialize_agents(self):
        # Initialize categorical counts for each categorical variable
        for agent_name, agent_data in self.config_data["agents_definition"].items():
            total_agents = agent_data["num_agents"]
            for variable_name, variable_data in agent_data["variables"].items():
                if not isinstance(variable_data["initial_value"],(int, float, str)):
                    # Check if distribution_type is defined and is categorical
                    if "distribution_type" in variable_data["initial_value"] and variable_data["initial_value"]["distribution_type"] == "categorical":
                        categories = variable_data["initial_value"]["categories"]
                        # Check if the sum of probabilities is 1
                        total_probability = sum(categories.values())
                        if not np.isclose(total_probability, 1.0):
                            raise ValueError(f"The sum of probabilities for the categorical distribution of '{variable_name}' in '{agent_name}' does not equal 1.0. Found sum: {total_probability}")
                        for category, percentage in categories.items():
                            count = round(percentage * total_agents)
                            categorical_distribution_tracker[agent_name][variable_name][category] = count
        
        agents = self.config_data["agents_definition"]
        for agent_name in agents:
            agent_data = agents[agent_name]
            num_agents = agent_data["num_agents"]
            for i in range(num_agents):
                unique_id = agent_name + "_" + str(i)
                agent = MyAgent(unique_id, self, agent_data, agent_name)
                self.schedule.add(agent)

    def step(self):
        self.schedule.step()
        if self.update_type == "synchronous":
            for agent in self.schedule.agents:
                agent.update_variables()


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run the simulation model.")
    # Arguments
    parser.add_argument('--config_path', type=str, help='Path to the config.json file')
    parser.add_argument('--prompt_path', type=str, help='Path to the .txt prompt file')
    # Parse arguments
    args = parser.parse_args()

    # Load config data
    with open(args.config_path, 'r') as file:
        config_data = json.load(file)
    seed = config_data["simulation_parameters"]["seed"]
    
    # Create the Simulation object with the config file path
    sim = Simulation(args.config_path, args.prompt_path, seed=seed)
    for _ in range(sim.simulation_steps):
        sim.step()
