from langchain.prompts import PromptTemplate
import numpy as np
from collections import defaultdict


def validate_distribution_properties(props):
    if "distribution_type" not in props:
        raise ValueError("Distribution type is required")

    dist = props["distribution_type"]
    data_type = props.get("data_type", None)

    if dist not in ["normal", "uniform", "categorical", "gamma"]:
        raise ValueError(f"Invalid distribution type: {dist}")

    if dist == "normal":
        if "mean" not in props:
            raise ValueError("'mean' is required for normal distribution")
        if data_type not in ["float", "int"]:
            raise ValueError("Data type for normal distribution must be 'float' or 'int'")
        
    if dist == "gamma":
        if "mean" not in props:
            raise ValueError("'mean' is required for gamma distribution")
        if data_type not in ["float", "int"]:
            raise ValueError("Data type for normal distribution must be 'float' or 'int'")

    if dist == "uniform":
        if "low" not in props and "high" not in props and "options" not in props:
            raise ValueError("'low', 'high' or 'options' are required for uniform distribution")
        if "options" not in props and data_type not in ["float", "int"]:
            raise ValueError("Data type for uniform distribution without options must be 'float' or 'int'")
    
    if dist == "categorical":
        if "categories" not in props:
            raise ValueError("'categories' are required for categorical distribution")
        
def get_initial_value(distribution_properties, variable_name=None, group=None, categorical_distribution_tracker=None):
        validate_distribution_properties(distribution_properties)

        distribution_type = distribution_properties["distribution_type"]
        data_type =  distribution_properties.get("data_type", None)
        mean =  distribution_properties.get("mean", None)
        std =  distribution_properties.get("std", 1)
        options =  distribution_properties.get("options", None)
        low =  distribution_properties.get("low", None)
        high =  distribution_properties.get("high", None)
        categories = distribution_properties.get("categories", None)
        
        # Normal distribution
        if distribution_type == "normal":
            return np.random.normal(mean, std) if data_type == "float" else int(np.random.normal(mean, std))

        # Uniform distribution
        if distribution_type == "uniform":
            return np.random.choice(options) if options else (
                np.random.uniform(low, high) if data_type == "float" else np.random.randint(low, high)
            )
        
        # Gamma distribution
        if distribution_type == "gamma":
            shape = mean**2 / std**2
            scale = std**2 / mean
            return np.random.gamma(shape, scale) if data_type == "float" else int(np.random.gamma(shape, scale))

        # Categorical distribution
        if distribution_type == "categorical":
            # Initialize the tracker for the group-variable combination if not already done
            tracker = categorical_distribution_tracker
            # Get the remaining counts for each category
            remaining_counts = {category: tracker[group][variable_name][category] for category in categories}
            categories_with_remaining_counts = [cat for cat, count in remaining_counts.items() if count > 0]

            # If all counts are exhausted, choose randomly from all categories
            if not categories_with_remaining_counts:
                chosen_category = np.random.choice(list(categories.keys()))
            else:
                # Choose randomly, weighted by the remaining counts
                weights = [remaining_counts[cat] for cat in categories_with_remaining_counts]
                chosen_category = np.random.choice(categories_with_remaining_counts, p=np.array(weights) / np.sum(weights))

            # Decrement the count for the chosen category
            tracker[group][variable_name][chosen_category] -= 1

            return chosen_category

# def string_to_python_function(self, expression):
#         # Restructure variables
#         expression = expression.replace("self.", "")
#         expression = expression.replace("class", self.name)
#         for agent in self.model.schedule.agents:
#             expression = expression.replace(agent.name+".", agent.name+"_")

#         # For "if" statements
#         expression = expression.replace(")", "):\n   return")
#         expression = expression.replace("{", "  ")
#         expression = expression.replace("}", "\n")
#         expression = expression.replace("else", "else:\n   return")

#         # For "avg", "sum", "min", "max", "count" operators
#         expression = expression.replace("avg", "np.mean")
#         expression = expression.replace("sum", "np.sum")
#         expression = expression.replace("min", "np.min")
#         expression = expression.replace("max", "np.max")
#         expression = expression.replace("count", "len")
#         expression = expression.replace("[", "(")
#         expression = expression.replace("]", ")")

#         # For single line operator
#         if "return" not in expression:
#             expression = "return " + expression

#         return "def func():\n" + " import numpy as np\n" + " " + expression

def string_to_python_function(self, expression):
    # Restructure variables
    expression = expression.replace("self.", "")
    expression = expression.replace("class", self.name)
    for agent in self.model.schedule.agents:
        expression = expression.replace(agent.name+".", agent.name+"_")
    expression = expression.replace(".init_value", "_init_value")

    # For "if" statements
    expression = expression.replace(";", "\n")
    expression = expression.replace("):", "):\n   return")
    expression = expression.replace("else:", "else:\n   return")

    # For single line operator
    if "return" not in expression:
        expression = "return " + expression

    return "def func():\n" + " import numpy as np\n" + " from math import exp\n" + " " + expression

def generate_prompt_template(model, sys_prompt, user_prompt, format_instructions=""):

    if model.model_name == "GPT":
        prompt = PromptTemplate.from_template(
            "{sys_prompt} {format_instructions}\n\nQuery:\n{user_prompt}"
        )
        _input = prompt.format_prompt(sys_prompt=sys_prompt, user_prompt=user_prompt, format_instructions=format_instructions)
    else:
        # Initialize template values
        start_sys = model.template["start_sys"]
        end_sys = model.template["end_sys"]
        start_user = model.template["start_user"]
        end_user = model.template["end_user"]
        start_ai = model.template["start_ai"]
        end_ai = model.template["end_ai"]

        # Check if start_sys and end_sys are None
        if start_sys is not None and end_sys is not None and start_ai is not None:
            prompt = PromptTemplate.from_template(
                "{start_sys}\n{sys_prompt} {format_instructions}\n{end_sys}\n{start_user}\n{user_prompt}\n{end_user}\n{start_ai}"
            )
            _input = prompt.format_prompt(start_sys=start_sys, sys_prompt=sys_prompt, end_sys=end_sys, start_user=start_user, user_prompt=user_prompt, end_user= end_user, start_ai=start_ai, format_instructions=format_instructions)

        if start_sys is not None and end_sys is not None and start_ai is None:
            prompt = PromptTemplate.from_template(
                "{start_sys}\n{sys_prompt} {format_instructions}\n{end_sys}\n{start_user}\n{user_prompt}\n{end_user}"
            )
            _input = prompt.format_prompt(start_sys=start_sys, sys_prompt=sys_prompt, end_sys=end_sys, start_user=start_user, user_prompt=user_prompt, end_user= end_user, format_instructions=format_instructions)

        if start_sys is None and end_sys is None and start_ai is not None:
            prompt = PromptTemplate.from_template(
                "{start_user}\n{sys_prompt} {format_instructions}\n\nQuery:\n{user_prompt}\n{end_user}\n{start_ai}"
            )
            _input = prompt.format_prompt(sys_prompt=sys_prompt, start_user=start_user, user_prompt=user_prompt, end_user= end_user, start_ai=start_ai, format_instructions=format_instructions)

        if start_sys is None and end_sys is None and start_ai is None:
            prompt = PromptTemplate.from_template(
                "{start_user}\n{sys_prompt} {format_instructions}\n\nQuery:\n{user_prompt}\n{end_user}"
            )
            _input = prompt.format_prompt(sys_prompt=sys_prompt, start_user=start_user, user_prompt=user_prompt, end_user= end_user, format_instructions=format_instructions)

        if start_sys is None and end_sys is None and start_ai is None and end_ai is None and start_user is None and end_user is None:
            prompt = PromptTemplate.from_template(
                "{sys_prompt} {format_instructions}\n\nQuery:\n{user_prompt}"
            )
            _input = prompt.format_prompt(sys_prompt=sys_prompt, user_prompt=user_prompt, format_instructions=format_instructions)

    return _input