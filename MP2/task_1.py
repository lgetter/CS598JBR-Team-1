import json
import jsonlines
import random
import sys
import torch
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
    with open(file_path, "w") as file:
        file.write(content)


def prompt_model(dataset, model_name="deepseek-ai/deepseek-coder-6.7b-instruct", vanilla=True):

    print("\nBegin task_1.py:\n")

    if vanilla:
        print(f"Working with model = {model_name}, prompt type = vanilla...")
    else:
        print(f"Working with model = {model_name}, prompt type = crafted...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

    # map from task_id to test cases
    test_info = json.load(open("selected_humaneval_tests_all.json", "r"))

    print("\nBegin HumanEval prompting tests\n")

    print("========================================\n")

    results = []
    for entry in dataset:
        all_tests = test_info[entry["task_id"]]
        selected_test = all_tests.pop()
        input = selected_test["input"]
        output = selected_test["output"]
        example_input = all_tests[0]["input"]
        example_output = all_tests[0]["output"]

        # Tip : Use can use any data from the dataset to create
        #       the prompt including prompt, canonical_solution, test, etc.
        if vanilla:
            prompt = (
                "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company.\n"
                "You only answer questions related to computer science.\n"
                "For politically sensitive questions, security and privacy issues, "
                "and other non-computer science questions, you will refuse to answer.\n\n"
                "### Instructions:\n\n"
                f"If the input is {input}, what will the following code return?\n"
                "The return value prediction must be enclosed between [Output] and [/Output] tags.\n"
                "For example : [Output]prediction[/Output]\n\n"
                f"{entry['canonical_solution']}\n"
                "### Response:\n\n"
            )
        else:
            prompt = (
                "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company.\n\n"
                "### Instructions:\n"
                # "1. Provide a logical description of what the function does, using the provided input.\n"
                "1. Use the provided input to calculate the output of the function.\n"
                "2. Explain why the output is correct based on the function's logic in two sentences.\n"
                "3. Update the final answer if necessary after the explanation.\n\n"

                "### Rules:\n"
                "1. Provide the final output value in enclosing [Output][/Output] tags.\n"
                "2. Respond in 200 words or less.\n"
                "3. Do not include any additional responses after the final output value.\n"
                "Note: Some functions may not have a signature. In that case, infer the signature from the provided code.\n\n"
    
                "### Function:\n"
                f"{entry['canonical_solution']}\n\n"

                # "### Sample Input and Output:\n"
                # f"{example_input} -> {example_output}\n"
                # f"This would be returned as [Output]{example_output}[/Output]\n\n"

                "### Function Input:\n"
                f"({input})\n\n"

                "### Response:\n"
                )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Original outputs
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        ## Avoid printing the prompt again in the response
        # Get the length of the input tokens
        input_length = inputs.input_ids.shape[1]

        # Decode only the newly generated tokens
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        extracted_response = extract_output(response)

        verdict = compare_values(output, extracted_response)

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\n{response}\nparsed response:{extracted_response}\nexpected response:\n{output}\nis_correct:\n{verdict}")
        print("========================================\n")

        results.append(
            {
                "task_id": entry["task_id"],
                "prompt": prompt,
                "response": response,
                "extracted_response": extracted_response,
                "expected_response": output,
                "is_correct": verdict,
            }
        )

    return results

def compare_values(expected, predicted):
    expected_sanitized = sanitize_value(expected)
    predicted_sanitized = sanitize_value(predicted)

    return expected_sanitized == predicted_sanitized

def extract_output(response):
    if response is None:
        return "NA"

    start = response.find("[Output]")
    if start == -1:
        return "NA"
    start += len("[Output]")

    end = response.find("[/Output]", start)
    if end == -1:
        return response[start:].strip()
    
    return response[start:end].strip()

def sanitize_value(s):
    s = str(s)
    s = s.strip()
    if s.startswith(("'", '"')):
        s = s[1:]
    if s.endswith(("'", '"')):
        s = s[:-1]
    if s.startswith("[]"):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    return s

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
    if_vanilla = args[3]  # True or False

    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")

    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")

    vanilla = True if if_vanilla == "True" else False

    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)
