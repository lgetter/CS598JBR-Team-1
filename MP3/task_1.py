import jsonlines
import sys
import torch
import subprocess
import tempfile
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP3/task_1;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def extract_java_code(response):
    """
    Extract Java code from the model response.
    The response typically contains code in ```java ... ``` blocks.
    """
    # Remove common explanatory prefixes
    response = re.sub(r'^.*?(?:Here is|Here\'s).*?(?:Java|translation|version|code|implementation|method).*?:?\s*\n', '', response, flags=re.IGNORECASE)

    # Try to find code between ```java and ``` (closed blocks)
    pattern = r'```java\s*(.*?)\s*```'
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        # Return the first code block found
        return matches[0].strip()

    # Try to find code starting with ```java but not closed (truncated response)
    pattern_open = r'```java\s*(.*?)$'
    matches = re.findall(pattern_open, response, re.DOTALL)

    if matches:
        # Return the code even if block isn't closed
        return matches[0].strip()
    
    # If no code blocks found, try to extract anything that looks like a method
    # Look for public/private method declarations
    method_pattern = r'((?:public|private|protected|static|\s)+[\w<>\[\]]+\s+\w+\s*\([^\)]*\)\s*\{[\s\S]*?\n\})'
    matches = re.findall(method_pattern, response, re.MULTILINE)

    if matches:
        return '\n'.join(matches)

    # Last resort: return the response as-is
    return response.strip()

def load_java_dataset(seed):
    """
    Load the Java dataset to get the test code and method signatures.
    """
    java_dataset_file = f"selected_humanevalx_java_{seed}.jsonl"
    java_dataset = {}

    try:
        with jsonlines.open(java_dataset_file) as reader:
            for line in reader:
                java_dataset[line['task_id']] = line
    except FileNotFoundError:
        print(f"Warning: Java dataset file {java_dataset_file} not found!")
        return {}

    return java_dataset

def create_java_test_file(java_entry, translated_code):
    """
    Create a complete Java file with the translated code and test.
    """
    # Extract the method signature from the prompt
    prompt = java_entry['prompt']

    # The prompt contains the class declaration and method signature
    # We need to extract everything before the method body starts
    # Typically ends with "public ReturnType methodName(params) {"

    # Get the declaration (imports + class + method signature)
    declaration = java_entry['declaration']

    # Extract just the method body from translated_code
    # Remove any class declarations or method signatures the model might have added
    method_body = translated_code

    # Remove any lines that contain import statements
    # This handles various import patterns:
    # - import java.util.*;
    # - import java.util.ArrayList;
    # - import static java.lang.Math.*;
    lines = method_body.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that start with 'import' (case-insensitive)
        # Also check if 'import' appears after whitespace at start of line
        if not (stripped.startswith('import ') or 
                stripped.startswith('import\t') or
                re.match(r'^\s*import\s+', line, re.IGNORECASE)):
            filtered_lines.append(line)
    method_body = '\n'.join(filtered_lines)

    # Remove any standalone class declarations
    method_body = re.sub(r'class\s+\w+\s*\{[\s\S]*?\}', '', method_body)

    # Remove any method signatures (we'll use the one from the dataset)
    method_body = re.sub(r'(public|private|protected|static|\s)+[\w<>\[\]]+\s+\w+\s*\([^\)]*\)\s*\{', '', method_body)

    # Ensure the method body doesn't start with }
    method_body = method_body.lstrip()
    if method_body.startswith('}'):
        method_body = method_body[1:].lstrip()

    # Build the complete Java file
    java_code = declaration + "\n" + method_body + "\n}\n"

    # Add the test code
    test_code = java_entry['test']

    complete_code = java_code + "\n" + test_code

    #print(f"Complete Java code for {java_entry['task_id']}:\n{complete_code}\n")
    return complete_code

def run_java_test(java_code, task_id):
    """
    Compile and run the Java code to test if it's correct.
    Returns True if tests pass, False otherwise.
    """
    # Create a temporary directory for Java files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the Java code to a file
        # The test code typically has a Main class
        java_file = os.path.join(temp_dir, "Main.java")

        try:
            with open(java_file, 'w') as f:
                f.write(java_code)

            # Compile the Java file
            compile_result = subprocess.run(
                ['javac', java_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            if compile_result.returncode != 0:
                print(f"Compilation failed for {task_id}:")
                print(compile_result.stderr)
                return False

            # Run the Java program
            run_result = subprocess.run(
                ['java', '-cp', temp_dir, 'Main'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if run_result.returncode != 0:
                print(f"Execution failed for {task_id}:")
                print(run_result.stderr)
                return False

            # If we get here, tests passed!
            return True

        except subprocess.TimeoutExpired:
            print(f"Timeout for {task_id}")
            return False
        except Exception as e:
            print(f"Error testing {task_id}: {e}")
            return False

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")

    # Download the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load the model with quantization
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

    # Load the Java dataset for testing
    # Extract seed from the first entry's file name pattern
    # The Python dataset should have been generated with the seed in the filename
    seed = "237879371724955854448207014936885343769"  # Your team's seed
    java_dataset = load_java_dataset(seed)

    results = []
    i = 1
    for entry in dataset:
        print(f"\n({i}/20) Processing Task_ID {entry['task_id']}...")
        i += 1
        declaration = java_dataset.get(entry['task_id'])['declaration']
        # Create prompt for the model
        # The task is to translate Python code to Java
        if vanilla:
            # Vanilla prompt - simple instruction-based approach
            prompt = (
                "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, "
                "and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, "
                "and other non-computer science questions, you will refuse to answer.\n"
                "### Instruction:\n"
                f"Translate the following Python function to Java:\n\n"
                f"{entry['prompt']}\n"
                f"{entry['canonical_solution']}\n"
                "Provide only the Java method implementation (the body of the method).\n"
                "### Response:\n"
            )
        else:
            # Crafted prompt - enhanced with type hints while keeping DeepSeek format
            prompt = (
                f"You are an expert programmer in both Python and Java languages. \n"
                "### Instruction:\n"                
                f"Translate the following Python function to Java.\n\n"
                f"Python code:\n"
                f"{entry['prompt']}\n"
                f"{entry['canonical_solution']}\n\n"
                f"Important conversions:\n"
                f"- list → ArrayList<Type>: use .get(i), .add(x), .size()\n"
                f"- dict → HashMap<K,V>: use .get(k), .put(k,v), .containsKey(k)\n"
                f"- str[i:j] → str.substring(i, j)\n"
                f"- str[::-1] → new StringBuilder(str).reverse().toString()\n"
                f"- len() → .length() for strings, .size() for collections\n"
                f"- List comprehensions → use loops or streams\n"
                f"- zip(a, b) → iterate with index: for(int i=0; i<a.size(); i++)\n"
                f"- enumerate() → use for loop with index variable\n"
                f"- ''.join(list) → String.join(\"\", list) or StringBuilder\n\n"
                f"Expected declaration and method signature in Java:\n"
                f"{declaration}\n\n"
                "Provide only the Java method implementation (the body of the method).\n"
                "### Response:\n"
            )

        #print(f"Prompt:\n{prompt}\n")

        # Prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens to avoid repeating the prompt
        input_length = inputs.input_ids.shape[1]
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"Response:\n{response}\n")

        # Process the response - ACTUALLY TEST THE CODE
        verdict = False

        # Get the corresponding Java entry for testing
        java_entry = java_dataset.get(entry['task_id'])

        if java_entry:
            # Extract the Java code from the response
            java_code = extract_java_code(response)
            print(f"Extracted Java code for {entry['task_id']}:\n{java_code}\n")

            try:
                # Create a complete Java test file
                complete_java_file = create_java_test_file(java_entry, java_code)

                # Run the Java tests
                verdict = run_java_test(complete_java_file, entry['task_id'])

                if verdict:
                    print(f"✓ Tests PASSED for {entry['task_id']}")
                else:
                    print(f"✗ Tests FAILED for {entry['task_id']}")

            except Exception as e:
                print(f"Error processing {entry['task_id']}: {e}")
                verdict = False
        else:
            print(f"Warning: No Java test found for {entry['task_id']}")
            # Fallback to basic check
            verdict = False

        print(f"is_correct: {verdict}")
        print("========================================")

        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
        })

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
    This Python script is to run prompt LLMs for code translation.
    Usage:
    `python3 task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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
