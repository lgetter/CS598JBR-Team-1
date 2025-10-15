import jsonlines
import random
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

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
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config
    )

    results = []
    for entry in dataset:
        input, output = extract_random_test(entry["test"])
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        prompt = """You are an AI programming assistant. You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek 
        Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, 
        and other non-computer science questions, you will refuse to answer. 
        ### Instruction:
        If the input is """ + input + """, what will the following code return? 
        Return only the return value. For example if the return value is True, just return True.
        Reason step by step to solve the problem. \n""" + entry["prompt"]
        
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=500,
            do_sample=False,
            temperature=0.0
        )

        # TODO: process the response and save it to results
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        verdict = False
        if output in response:
            verdict = True
        
        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_correct:\n{verdict}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
        })
        
    return results

def extract_random_test(entry):
    tests = entry.strip().split("\n")
    tests = [test for test in tests if test.strip().startswith("assert ")]
    test = random.choice(tests)
    test = test.strip()
    if test.startswith("assert not candidate("): 
        # assert not candidate(input)
        start = test.find("candidate(") + len("candidate(")
        end = test.rfind(")")
        input_val = test[start:end].strip()
        return input_val, "False"
    elif test.startswith("assert candidate("): 
        # assert candidate(input) == output
        if "==" in test:
            left, right = test.split("==", 1)
            left = left.strip()
            right = right.strip()
            start = left.find("candidate(") + len("candidate(")
            end = left.rfind(")")
            input_val = left[start:end].strip()
            output_val = right
            return input_val, output_val
        else: 
            # assert candidate(input)
            start = test.find("candidate(") + len("candidate(")
            end = test.rfind(")")
            input_val = test[start:end].strip()
            return input_val, "True"
    return None, None


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
    `python3 Task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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
