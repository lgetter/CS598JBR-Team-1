import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP3/task_1;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

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

    results = []
    i = 1
    for entry in dataset:
        print(f"\n({i}/20) Processing Task_ID {entry['task_id']}...")
        i += 1

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
            # Crafted prompt - more detailed with context and examples
            prompt = (
                f"You are an expert programmer specializing in code translation between Python and Java. "
                f"Your task is to translate the given Python function to Java while maintaining the same functionality.\n\n"
                f"### Instructions:\n"
                f"Translate the following Python function to Java. Pay careful attention to:\n"
                f"1. Data type conversions:\n"
                f"   - Python lists/tuples → Java List<Type> or arrays\n"
                f"   - Python strings → Java String\n"
                f"   - Python None → Java null or Optional\n"
                f"   - Python dict → Java Map or HashMap\n"
                f"   - Python set → Java Set or HashSet\n"
                f"2. String operations:\n"
                f"   - Python string methods → Java String methods\n"
                f"   - Use StringBuilder for string concatenation in loops\n"
                f"3. List operations:\n"
                f"   - Python list comprehensions → Java streams or loops\n"
                f"   - Python slicing → Java subList() or manual loops\n"
                f"   - Python append() → Java add()\n"
                f"4. Common patterns:\n"
                f"   - Python for x in list → Java for (Type x : list)\n"
                f"   - Python enumerate() → Java for loop with index\n"
                f"   - Python zip() → Java parallel iteration\n"
                f"   - Python sorted() → Java Collections.sort() or stream().sorted()\n"
                f"   - Python max()/min() → Java Collections.max()/min() or Math.max()/min()\n"
                f"5. Return types:\n"
                f"   - Match the expected Java return type\n"
                f"   - Use List<Integer>, List<String>, etc. for lists\n\n"
                f"### Python Function:\n"
                f"{entry['prompt']}\n"
                f"{entry['canonical_solution']}\n\n"
                f"### Requirements:\n"
                f"- Provide ONLY the method body (implementation inside the method)\n"
                f"- Do NOT include the class definition or method signature\n"
                f"- Do NOT include import statements\n"
                f"- Ensure the code compiles and runs correctly in Java\n"
                f"- Maintain the exact same logic and behavior as the Python version\n\n"
                f"### Response:\n"
            )

        # Prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens to avoid repeating the prompt
        input_length = inputs.input_ids.shape[1]
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"Response:\n{response}\n")

        # Process the response
        # For this task, we're translating code, so we don't have a simple True/False verdict
        # The verdict will be determined by running the tests (which happens in validation)
        # For now, we check if the response contains Java-like code
        verdict = False
        if "return" in response.lower() or "{" in response or ";" in response:
            verdict = True

        print(f"Contains Java-like code: {verdict}")
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
