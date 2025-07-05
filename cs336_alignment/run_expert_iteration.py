import re
import json
import torch
import wandb
import random
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from argparse import ArgumentParser

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from cs336_alignment.utilities import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from cs336_alignment.evaluate_vllm import evaluate_vllm

SEED = 69
torch.manual_seed(SEED)
random.seed(SEED)

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")


def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU sepearate from the policy
    """
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    
def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_prompt_with_template(question: str, template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(question=question)

def get_batch(formatted_train_prompts: list[str], train_data, batch_size: int) -> list[str]:
    batch_indices = random.sample(
        range(len(formatted_train_prompts)), batch_size
    )
    return [formatted_train_prompts[i] for i in batch_indices], [train_data[i] for i in batch_indices]

def get_sft_batch(
    tokenized_train_data: dict[str, torch.Tensor], batch_size: int, device: str
) -> dict[str, torch.Tensor]:
    batch_indices = random.sample(
        range(len(tokenized_train_data["input_ids"])), batch_size
    )
    return {k: v[batch_indices].to(device) for k, v in tokenized_train_data.items()}

def to_float(val):
    if isinstance(val, torch.Tensor):
        return val.float().item()
    return float(val)

def SFT(
    args,
    sft_data,
    model,
    tokenizer,
    optimizer,
    vllm,
    test_data,
    formatted_test_prompts,
    device_SFT,
    global_step: int = 0,
):
    SFT_num_epochs = args.SFT_num_epochs

    amp_ctx = torch.amp.autocast(
        device_type=device_SFT,
        dtype=torch.bfloat16,
    )

    tokenized_sft_data = tokenize_prompt_and_output(
        [data["prompt"] for data in sft_data],
        [data["response"] for data in sft_data],
        tokenizer
    )
    # print(tokenized_sft_data)
    n_grad_accum_steps = 8
    micro_batch_size = 2
    
    n_sft_steps = len(sft_data) * SFT_num_epochs // (n_grad_accum_steps * micro_batch_size)
    eval_steps = 10
    print(f"n_sft_steps{n_sft_steps} = len(sft_data){len(sft_data)} * SFT_num_epochs{SFT_num_epochs} // (n_grad_accum_steps{n_grad_accum_steps} * micro_batch_size{micro_batch_size})")
    train_batch = get_sft_batch(tokenized_sft_data, micro_batch_size, device_SFT)
    input_ids = train_batch["input_ids"].to(device_SFT)
    labels = train_batch["labels"].to(device_SFT)
    response_mask = train_batch["response_mask"].to(device_SFT)

    for i_sft_step in range(n_sft_steps):
        for j_grad_accum_step in range(n_grad_accum_steps):
            with amp_ctx:
                response_log_probs = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                log_probs = response_log_probs["log_probs"]
                entropy = response_log_probs["token_entropy"]

                next_batch = get_sft_batch(tokenized_sft_data, micro_batch_size, device_SFT)

                loss, _ = sft_microbatch_train_step(log_probs, response_mask, n_grad_accum_steps)

                if j_grad_accum_step == n_grad_accum_steps - 1:
                    # i think we need to do something here
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    optimizer.zero_grad()
                    # log train generation
                    global_step += 1
                    print(f"\n📊 Training Summary at Step {global_step}:")
                    print(f"Loss: {loss:.6f}")
                    print(f"Entropy: {entropy.mean().item():.6f}")
                    wandb.log({
                        "train/loss": to_float(loss),
                        "train/entropy": to_float(entropy.mean()),
                        "train_step": global_step,
                    })

            train_batch = next_batch
            input_ids = train_batch["input_ids"].to(device_SFT)
            labels = train_batch["labels"].to(device_SFT)
            response_mask = train_batch["response_mask"].to(device_SFT)

        if (i_sft_step + 1) % eval_steps == 0 or i_sft_step == 0:
            # use vllm to eval 0.0
            load_policy_into_vllm_instance(model, vllm)

            sampling_params =  SamplingParams(
                temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
            )
            counts, format_errors, answer_errors = evaluate_vllm(
                vllm_model=vllm,
                reward_fn=r1_zero_reward_fn,
                data=test_data,
                prompts=formatted_test_prompts,
                eval_sampling_params=sampling_params
            )

            # log eval generation
            accuracy = counts['correct'] / len(formatted_test_prompts)
            print(f"\n📊 Evaluation Summary at Step {global_step}:")
            print(f"Correct (format + answer): {counts['correct']}")
            print(f"Wrong answer (but correct format): {counts['wrong_answer']}")
            print(f"Wrong format: {counts['wrong_format']}")
            print(f"Accuracy: {accuracy}")
            wandb.log({
                "eval/correct": counts["correct"],
                "eval/wrong_answer": counts["wrong_answer"],
                "eval/wrong_format": counts["wrong_format"],
                "eval/accuracy": accuracy,
                "eval_step": global_step,
            })
            
    if n_sft_steps % eval_steps != 0:
        # one last time
        load_policy_into_vllm_instance(model, vllm)

        sampling_params =  SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
        )
        counts, format_errors, answer_errors = evaluate_vllm(
            vllm_model=vllm,
            reward_fn=r1_zero_reward_fn,
            data=test_data,
            prompts=formatted_test_prompts,
            eval_sampling_params=sampling_params
        )

        # log eval generation
        accuracy = counts['correct'] / len(formatted_test_prompts)
        print(f"\n📊 Evaluation Summary at Step {global_step}:")
        print(f"Correct (format + answer): {counts['correct']}")
        print(f"Wrong answer (but correct format): {counts['wrong_answer']}")
        print(f"Wrong format: {counts['wrong_format']}")
        print(f"Accuracy: {accuracy}")
        wandb.log({
            "eval/correct": counts["correct"],
            "eval/wrong_answer": counts["wrong_answer"],
            "eval/wrong_format": counts["wrong_format"],
            "eval/accuracy": accuracy,
            "eval_step": global_step,
        })
    
    return model, global_step

def main(args):
    global_step = 0

    EI_num_G = args.EI_num_G
    EI_batch_size = args.EI_batch_size

    model_id = "Qwen/Qwen2.5-Math-1.5B"
    device_vllm = "cuda"
    device_SFT = "cuda:2"
    train_file_path = "./data/gsm8k/train.jsonl"
    test_file_path = "./data/gsm8k/test.jsonl"
    TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt" # for train & test

    n_expert_iteration_steps = 5
    sampling_temperature = 1.0
    sampling_max_tokens = 1024
    sampling_min_tokens = 4

    # init policy model
    EI_vllm = init_vllm(model_id, device_vllm, seed=SEED, gpu_memory_utilization=0.9)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_SFT,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # init reward function: r1...
    # init task question D (load questions)
    train_data = load_jsonl(train_file_path)
    test_data = load_jsonl(test_file_path)

    formatted_train_prompts = [
        format_prompt_with_template(example["question"], TEMPLATE_PATH) for example in train_data
    ]
    formatted_test_prompts = [
        format_prompt_with_template(example["question"], TEMPLATE_PATH) for example in test_data
    ]

    # for step in 1 ... n_expert_iteration_steps
    # sample batch question Db
    
    for idx in range(n_expert_iteration_steps):
        ei_step = idx + 1
        print(f"Starting Expert Iteration {ei_step}")
        # old policy model <- policy model
        # sample G outputs with (old policy model, Db)
        formatted_train_prompts_batch, train_data_batch = get_batch(formatted_train_prompts, train_data, EI_batch_size)

        sampling_params = SamplingParams(
            temperature=sampling_temperature,
            top_p=1.0,
            max_tokens=sampling_max_tokens,
            min_tokens=sampling_min_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            n=EI_num_G,
            seed=SEED,
        )

        outputs = EI_vllm.generate(formatted_train_prompts_batch, sampling_params)
        all_generated_texts = [
            [o.text.strip() for o in output.outputs]
            for output in outputs
        ]
        # compute rewards for each output (reward function)
        results_per_rollout = []
        format_error_examples = []
        answer_error_examples = []
        
        for prompt_idx, (prompt, generated_answers, example) in enumerate(zip(formatted_train_prompts_batch, all_generated_texts, train_data_batch)):
            ground_truth = example["answer"]
            reference_answer = extract_reference_answer(ground_truth)

            for rollout_idx, generated_text in enumerate(generated_answers):
                metrics = r1_zero_reward_fn(generated_text, reference_answer)
                is_correct = metrics["reward"] == 1.0
                is_format_wrong = metrics["format_reward"] == 0.0
                is_answer_wrong = metrics["answer_reward"] == 0.0 and metrics["format_reward"] == 1.0

                results_per_rollout.append({
                    "prompt_idx": prompt_idx,
                    "rollout_idx": rollout_idx,
                    "metrics": metrics,
                    "prompt": prompt,
                    "response": generated_text,
                    "is_correct": is_correct
                })

                if is_format_wrong:
                    format_error_examples.append({
                        "prompt": prompt,
                        "response": generated_text,
                        "expected": reference_answer
                    })
                elif is_answer_wrong:
                    answer_error_examples.append({
                        "prompt": prompt,
                        "response": generated_text,
                        "expected": reference_answer
                    })

        # 打印一些範例
        print("\n=== Format Error Examples ===")
        for i, ex in enumerate(format_error_examples[:1]):
            print(f"{i+1}. Prompt: {ex['prompt']}")
            print(f"   Response: {ex['response']}")
            print(f"   Expected Answer: {ex['expected']}\n")

        print("\n=== Answer Error Examples ===")
        for i, ex in enumerate(answer_error_examples[:1]):
            print(f"{i+1}. Prompt: {ex['prompt']}")
            print(f"   Response: {ex['response']}")
            print(f"   Expected Answer: {ex['expected']}\n")
        # ✅ Sanity Check: Print how many total responses and how many are correct
        total_responses = len(results_per_rollout)
        correct_responses = sum(1 for item in results_per_rollout if item["is_correct"])
        print(f"\nSanity Check in the end of Expert Iteration Step {idx + 1}:")
        print(f"Total generated responses: {total_responses}")
        print(f"Correct responses: {correct_responses}")
        print(f"Accuracy so far: {correct_responses / total_responses * 100:.2f}%\n")
        # filter out wrong output -> Dsft
        sft_data = []
        for item in results_per_rollout:
            if item["is_correct"]:
                sft_data.append({
                    "prompt": item["prompt"],
                    "response": item["response"]
                })
        # print(sft_data)
        # policy model <- SFT(policy model, Dsft)
        model, global_step = SFT(
            args,
            sft_data,
            model,
            tokenizer,
            optimizer,
            EI_vllm,
            test_data,
            formatted_test_prompts,
            device_SFT=device_SFT,
            global_step=global_step,
        )
        load_policy_into_vllm_instance(model, EI_vllm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--EI_num_G", type=int, default=5)
    parser.add_argument("--SFT_num_epochs", type=int, default=1)
    # n_expert_iteration = 5
    parser.add_argument("--EI_batch_size", type=int, default=512)
    args = parser.parse_args()

    wandb.init(
        entity="koala34025-national-tsing-hua-university",
        project="expert_iteration",
        config=args,
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")

    wandb.define_metric("train/*", step_metric="train_step")

    wandb.define_metric("eval/*", step_metric="eval_step")

    main(args)