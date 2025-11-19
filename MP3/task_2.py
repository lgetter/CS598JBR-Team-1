import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP3/task_2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    
     # TODO: download the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # # TODO: load the model with quantization
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
    
    results = []
    for entry in dataset:
        # Create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        if vanilla:
            prompt = f"""
            You are an AI programming assistant. You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
            
            ### Instruction:
            {entry['declaration']}{entry['buggy_solution']}
            "Is the above code buggy or correct? Please explain your step by step reasoning. The prediction should be enclosed within <start> and <end> tags. For example: <start>Buggy<end> or <start>Correct<end>" \
            ### Response:
            """
        else:
            prompt = (
                f"""
                You are an expert Python programmer. Your job is to analyze whether the implementation below is BUGGY or CORRECT.

                Follow this exact reasoning process:
                1. Read the **specification** (docstring + example tests).
                2. Determine the intended behavior from the examples.
                3. Examine the provided **implementation**.
                4. Identify if it matches the intended behavior.
                5. At the final line, output ONLY:
                <start>Buggy<end>
                or 
                <start>Correct<end>
                
                ### Instructions:
                Determine if the Python Function works as described by the Function Description. 
                Write out your reasoning step by step, and then provide your final verdict enclosed between <start> and <end> tags.

                ### Specification:
                {entry['prompt']}
                
                ### Example Tests
                {entry['example_test']}

                ### Implementation Under Review
                {entry['declaration']}{entry['buggy_solution']}

                ### Response:
                """)
        
        # Prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        input_length = inputs.input_ids.shape[1]
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"Processed response for Task_ID {entry['task_id']}:\n{response}")
        print("========================================\n")

        parsed = response.split("<start>")[-1].split("<end>")[0].strip().lower()

        # Process the response and save it to results
        verdict = (parsed == "buggy")

        print(f"Expected: Buggy\nParsed: {parsed}\nIs correct: {verdict}\n")

        #print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_correct:\n{verdict}")
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
    This Python script is to run prompt LLMs for bug detection.
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
