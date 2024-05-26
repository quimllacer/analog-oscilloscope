
# config.py
current_function = None
current_function_params = None
current_effects = None


def update_function(function_name, params):
    global current_function, current_function_params
    current_function = function_name
    current_function_params = params
    print(f"Generative function set to {current_function} with params {current_function_params}")
    test()

def apply_effect(effect_name, parameter_value):
    global current_effects
    current_effects[effect_name] = parameter_value
    print(f"Effect {effect_name} applied with parameter {parameter_value}")

def test():
    print(current_function)