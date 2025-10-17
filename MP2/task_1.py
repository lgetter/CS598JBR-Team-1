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

        function_signature = extract_function_signature(entry['prompt'])

        # TODO: create prompt for the model
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
                # f"{function_signature}\n"
                f"{entry['canonical_solution']}\n"
                "### Response:\n\n"
            )
        else:
            prompt = (
                # "You are an expert Python programmer. Help solve the following question.\n\n"
                # "### Instructions:\n\n"
                # "You are provided with a Python function description, the implementation of this function, and an example input-output pair.\n"
                # "You are provided with a Python function and an example input and output.\n"
                # "Your task is to determine the expected output of the function with the given input.\n"
                # "You must return the expected output of the provided function in enclosing [Output] and [/Output] tags as the final output.\n"
                # "For example, if the expected output is '1234', you should return [Output]'1234'[/Output].\n\n" 
                # "### Function description:\n"
                # f"{entry['prompt']}\n"
                # "### Python Function:\n"
                # f"{entry['canonical_solution']}\n\n"
                "### Instructions:\n"
                "You are provided with a Python function and a list of sample input-output pairs.\n" \
                "Your task is to:\n" \
                "1. List out the execution steps as consicely as possible using the provided input.\n" \
                "2. Return the expected output of the provided function in enclosing [Output] and [/Output] tags as the final output.\n" \
                "For example, if the expected output is '1234', output [Output]'1234'[/Output].\n\n"
                ""
                "### Function:\n" 
                f"{entry['canonical_solution']}\n\n"
                "### Sample Input-Output Pairs:\n"

                

                f"What is the output of this Python function with this input: {input}\n"
                f"{function_signature}\n"
                
                # "Here is an example input and output formatted in the requested response type:\n\n"
                )
            
            for test in all_tests[:3]:
                    prompt += f"Input: {test['input']} -> Output: [Output]{test['output']}[/Output]\n"

            prompt += "\n### Response:\n"

            # prompt += "\n### Question:\n\n"
            # prompt += (f"Now, given the function input: {input}, what is the expected output?\n\n")
            # prompt += "### Response:\n\n"

            # prompt += "Reason about your answer silently.\nReturn your answer like this [Output]<your_answer>[/Output].\n"

        print(f"Prompt for Task_ID {entry['task_id']}:\n\n{prompt}")

        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Original outputs
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # TODO: process the response and save it to results

        # Original response
        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        ## Avoid printing the prompt again in the response
        # Get the length of the input tokens
        input_length = inputs.input_ids.shape[1]

        # Decode only the newly generated tokens
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"Processed response for Task_ID {entry['task_id']}:\n{response}")
        print("========================================\n")

        parsed_response = response.split("[Output]")[-1].split("[/Output]")[0].strip()

        verdict = False
        if output in parsed_response:
            verdict = True

        print(
            f"Expected output: {output}\n"
            f"Actual output: {parsed_response}\n"
            f"Is correct: {verdict}\n"
        )

        # print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nexpected response:\n{output}\nis_correct:\n{verdict}")
        results.append(
            {
                "task_id": entry["task_id"],
                "prompt": prompt,
                "response": response,
                "parsed_response": parsed_response,
                "expected_output": output,
                "is_correct": verdict,
            }
        )

    return results


def extract_random_test(tests):
    selected_test = random.choice(tests)
    return selected_test["input"], selected_test["output"]


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
