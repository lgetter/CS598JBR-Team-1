import json
import jsonlines
import sys
import torch
import subprocess
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def extract_function_signature(prompt_content):
    """
    Extracts the function signature (including imports/type hints) from the prompt.
    The extraction stops exactly at the colon that ends the function definition line(s).
    """

    # Find the end of the function signature
    end_index = prompt_content.find(':\n')
    
    return prompt_content[0:end_index+1]

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")

    # TODO: download the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # TODO: load the model with quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config,
    )
    
    test_info = json.load(open("selected_humaneval_tests_all.json", "r"))

    results = []

    i = 1

    for entry in dataset:
        task_id = entry["task_id"]
        task_number = task_id.split('/')[-1]
        all_tests = test_info[task_id]
        selected_test = all_tests.pop()
        input = selected_test["input"]
        output = selected_test["output"]

        function_signature = extract_function_signature(entry['prompt'])

        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create
        #       the prompt including prompt, canonical_solution, test, etc.
        if vanilla:
            prompt = (
f"""
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science.
For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instructions:
Generate a pytest test suite for the following code.
Only write unit tests in the response and nothing else.

{function_signature}
{entry['canonical_solution']}
### Response:
""")
        else:
            # selected_example = all_tests.pop()
            # example_input = selected_example["input"]
            # example_output = selected_example["output"]

            prompt = (
f"""
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science.
For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instructions:
Generate a comprehensive pytest test suite for the following code with maximum code coverage. 

Requirements:
1. Generate at least 15-20 test cases to cover all possible execution paths
2. Include tests for:
   - Normal/typical inputs
   - Edge cases (empty inputs, single elements, maximum values)
   - Boundary conditions
   - Different data types where applicable
   - All conditional branches (if/else statements)
   - Loop iterations (empty, single, multiple)
   - Error cases and exceptions
3. Ensure every line of code is executed by at least one test
4. Test all return value possibilities
5. Only write unit tests in the output and nothing else

{function_signature}
{entry['canonical_solution']}

### Response:

import pytest

""")

        print(f"({i}/20) Prompt for Task_ID {task_id}:\n{prompt}")

        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"Response for Task_ID {task_id}:\n{response}\n")

        # Extract only the test code from response
        # Look for test functions and clean up the response
        test_code = "import pytest\n\n"
        
        # Add the function under test
        test_code += entry['prompt'] + entry['canonical_solution'] + "\n\n"
        
        # Add the generated tests
        test_code += response

        # Create directory for temporary test files
        temp_test_dir = "Tests"
        os.makedirs(temp_test_dir, exist_ok=True)
        
        # Save the code under test to a file
        code_file = os.path.join(temp_test_dir, f"{task_number}.py")
        code_content = entry['prompt'] + entry['canonical_solution']
        save_file(code_content, code_file)
        
        # Save the test suite to a file
        test_file = os.path.join(temp_test_dir, f"{task_number}_test.py")
        save_file(test_code, test_file)

        # Run pytest with coverage
        coverage_type = "vanilla" if vanilla else "crafted"
        coverage_dir = "Coverage"
        os.makedirs(coverage_dir, exist_ok=True)
        coverage_file = os.path.join(coverage_dir, f"{task_number}_test_{coverage_type}.json")
        
        try:
            # Run pytest with coverage
            cmd = [
                "pytest", 
                test_file, 
                "--cov", task_number,
                "--cov-report", f"json:{coverage_file}",
                "-v"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=temp_test_dir  # Run pytest from the temp directory
            )
            
            print(f"Pytest output:\n{result.stdout}\n")
            if result.stderr:
                print(f"Pytest errors:\n{result.stderr}\n")
            
            # Read coverage report
            coverage = ""
            if os.path.exists(coverage_file):
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    # Extract coverage percentage
                    if 'totals' in coverage_data:
                        coverage = f"{coverage_data['totals'].get('percent_covered', 0):.2f}%"
                    else:
                        coverage = "Coverage data not available"
            else:
                coverage = "Coverage file not generated"
                
        except subprocess.TimeoutExpired:
            coverage = "Test execution timeout"
            print(f"Test execution timed out for {task_id}")
        except Exception as e:
            coverage = f"Error: {str(e)}"
            print(f"Error running tests for {task_id}: {str(e)}")
        
        # Clean up temporary files
        try:
            if os.path.exists(code_file):
                os.remove(code_file)
            if os.path.exists(test_file):
                os.remove(test_file)
        except Exception as e:
            print(f"Error cleaning up files: {str(e)}")

        print(f"Task_ID {task_id}:\ncoverage: {coverage}")
        print("========================================\n")
        
        results.append({
            "task_id": task_id,
            "prompt": prompt,
            "response": response,
            "coverage": coverage
        })
        
        i += 1
        
    return results

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader: 
            dataset.append(line)
    return dataset

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 task_2.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_vanilla>: Set to 'True' or 'False' to enable vanilla prompt
    
    Outputs:
    - You can check <output_file> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_vanilla = args[3] # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    vanilla = True if if_vanilla == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)
