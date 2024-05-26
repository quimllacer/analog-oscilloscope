# user_input.py
import re
from effects import functions

def start_user_input_thread():
    while True:
        try:
            user_input = input("Enter command as <function param value>:\n")
            function_name, parameter_name, parameter_value = re.split(r'\s+', user_input)
            if parameter_name not in ['status', 'name']:
                parameter_value = float(parameter_value)
            functions[function_name][parameter_name] = parameter_value
            m_functions = {
            key: {sub_key: str(sub_value)[:4] for sub_key, sub_value in value.items() if sub_key not in ['f', 'order']}
            for key, value in functions.items()
            }
            [print(a) for a in m_functions.items()]
        except Exception as e: print(e)