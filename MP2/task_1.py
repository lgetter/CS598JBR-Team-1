import json
import jsonlines
import random
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################


def save_file(content, file_path):
    with open(file_path, "w") as file:
        file.write(content)


def prompt_model(
    dataset, model_name="deepseek-ai/deepseek-coder-6.7b-instruct", vanilla=True
):
    print(f"Working with {model_name} prompt type {vanilla}...")

    # TODO: download the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # # TODO: load the model with quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
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

    results = []
    for entry in dataset:
        all_tests = test_info[entry["task_id"]]
        selected_test = all_tests.pop()
        input = selected_test["input"]
        output = selected_test["output"]
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create
        #       the prompt including prompt, canonical_solution, test, etc.
        if vanilla:
            prompt = (
                "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company.\n"
                "You only answer questions related to computer science. For politically sensitive questions, security and privacy issues, "
                "and other non-computer science questions, you will refuse to answer.\n"
                "### Instruction:\n"
                f"If the input is ({input}), what will the following code return?\n"
                "The return value prediction must be enclosed between [Output] and [/Output] tags. For example : [Output]prediction[/Output]\n"
                f"{entry['canonical_solution']}\n\n"
                "### Response:\n"
            )
        else:
            # v1
            # prompt = (
            #     "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company.\n"
            #     "You only answer questions related to computer science. For politically sensitive questions, security and privacy issues, "
            #     "and other non-computer science questions, you will refuse to answer.\n"
            #     "### Instruction:\n"
            #     "You are provided with a function description, the implementation of this function, and a sample input.\n"
            #     "Your task is to determine the expected output of the function when given the sample input.\n"
            #     "Reason through the problem step by step to arrive at the correct output.\n"
            #     "You must return the expected output of the provided function in enclosing [Output] and [/Output] tags as the final output.\n"
            #     "For example, if the expected output is True, return [Output]True[/Output].\n"
            #     "Reason step by step to solve the problem.\n"
            #     "Function description:\n"
            #     f"'{entry['prompt']}'\n"
            #     "Function implementation:\n"
            #     f"{entry['canonical_solution']}"
            #     "Here are some example inputs and their expected outputs:\n")

            # Crafted prompt v3 - Sean
            # prompt = (
            #     "You are acting as Guido Van Rossum, the legendary genius creator behind Python.\n"
            #     "Your task is to analyze the provided Python function and determine the expected output with the given input.\n\n"
            #     "### Instructions:\n\n"
            #     "Using the given Python code function, perform the following steps:\n"
            #     "1. Read through the entire Python function to fully understand it.\n"
            #     "2. Analyze the provided function input parameter(s). Take note of the type, number of parameters, and order the parameters are passed to the function.\n"
            #     "3. Reason through the execution logic of the function body step by step, showing your thoughts and steps in your output.\n"
            #     "4. Examine the example inputs and outputs and logically think through how the code executes to arrive at the example answers, revising your reasoning from the previous step if your reasoned answers are misaligned from the examples.\n"
            #     "4. Determine the expected output of the function given the input.\n"
            #     "5. Return the answer enclosed in [Output] and [/Output] tags, like this: [Output]answer[/Output]\n\n"
            #     "### Function:\n\n"
            #     f"{entry['canonical_solution']}\n\n"
            #     "### Examples:\n\n"
            #     "The example inputs are enclosed in [Input] and [/Input] tags [Input]parameters[/Input].\n"
            #     "Here are some example inputs and their outputs:\n\n"
            # )

            # Sean - V5
            prompt = (
                # "You are acting as Guido Van Rossum, the legendary genius creator behind Python.\n"
                # "Your task is to analyze the provided Python function and determine the expected output with the given input.\n\n"
                # "### Instructions:\n\n"
                # "Using the given Python code function, perform the following steps:\n"
                # "1. Read through the entire Python function to fully understand it.\n"
                # "2. Analyze the provided function input parameter(s). The example inputs are enclosed in [Input] and [/Input] tags, like this: [Input]parameters[/Input]\n"
                # "3. Determine the expected output of the function given the input parameters.\n"
                # "5. Return your answer for the expected output enclosed in [Output] and [/Output] tags, like this: [Output]answer[/Output]\n\n"
                # "### Function:\n\n"
                # f"{entry['canonical_solution']}\n\n"
                # "### Examples:\n\n"
                # "The example inputs are enclosed in [Input] and [/Input] tags [Input]parameters[/Input].\n"
                # "Here are some example inputs and their outputs:\n\n"
            )
            # Sean V6
            prompt = f"""
                You are a meticulous Python interpreter.
                Given a function and an input, determine the exact return value.

                ### Rules:
                - Do any reasoning silently.
                - Output ONLY the final value wrapped in [Output]...[/Output].
                - No quotes around strings unless the function actually returns a quoted string.
                - Use Python bools True/False and exact list/tuple/dict formatting.
                - Do not print or explain.

                ### Function:
                ```python
                {entry['canonical_solution']}
                ```

                ### Examples:
                """

            # for test in all_tests:
            #     prompt += f"Input: [Input]{test['input']}[/Input] => Output: [Output]{test['output']}[/Output]\n"

            for test in all_tests:
                prompt += f"[Input]{test['input']}[/Input] -> Output: [Output]{test['output']}[/Output]\n\n"

            prompt += "### Question:\n"
            prompt += (
                f"Given [Input]{input}[/Input], return [Output]...[/Output] only.\n\n"
            )

            # prompt += f"\nNow, given the input parameter(s): [Input]{input}[/Input], what is the expected output?\n\n"

            prompt += "### Response:\n"

        print(f"Prompt for Task_ID {entry['task_id']}:\n\n{prompt}")

        # TODO: prompt the model and get the response

        # Original version
        # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # V2 utilizing message template
        messages = [
            {
                "role": "system",
                "content": "You are a meticulous Python interpreter. Think privately. Return only the final answer.",
            },
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

        # TODO: process the response and save it to results
        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Avoid printing the prompt again in the response

        # Get the length of the input tokens
        input_length = inputs.input_ids.shape[1]

        # Decode only the newly generated tokens
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"Processed response for Task_ID {entry['task_id']}:\n\n{response}")
        print("========================================\n")

        response = response.split("[Output]")[-1].split("[/Output]")[0].strip()

        verdict = False
        if output in response:
            verdict = True

        print(
            f"Expected output: {output}\n"
            f"Actual output: {response}\n"
            f"Is correct: {verdict}\n"
        )

        # print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nexpected response:\n{output}\nis_correct:\n{verdict}")
        results.append(
            {
                "task_id": entry["task_id"],
                "prompt": prompt,
                "response": response,
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
