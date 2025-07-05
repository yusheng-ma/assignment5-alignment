import json
import re
import random


ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")
CALC_RE = re.compile(r"<<[^>]*>>")  # 匹配所有 <<...>> 的计算器注释


def extract_reference_answer(answer: str) -> tuple:
    match = ANS_RE.search(answer)
    if match:
        extracted = match.group(1).strip().replace(",", "")
        before = answer[:match.start()]
        after = answer[match.end():]
        remaining = before + after
        return extracted, remaining
    return "[invalid]", answer


def remove_calculator_annotations(text: str) -> str:
    return CALC_RE.sub("", text)


def format_prompt_with_template(question: str, template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(question=question)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def corrupt_answer(ref_answer, corruption_rate, all_answers):
    """以 corruption_rate 的機率用其他 sample 的答案替換"""
    if random.random() < corruption_rate:
        # Always return a different answer if possible
        candidates = [ans for ans in all_answers if ans != ref_answer]
        if candidates:
            return str(random.choice(candidates))
        else:
            return ref_answer
    return ref_answer


def main(corruption_rate=0.25):
    input_path = "data/gsm8k/train.jsonl"
    output_path = "data/gsm8k/processed_train.jsonl"
    output_corrupt_path = "data/gsm8k/processed_train_corrupted.jsonl"
    r1_zero_prompt_path = "cs336_alignment/prompts/r1_zero.prompt"

    data = load_jsonl(input_path)

    all_answers = []
    for item in data:
        raw_answer = item["answer"]
        ref_answer, _ = extract_reference_answer(raw_answer)
        all_answers.append(ref_answer)

    processed_clean = []
    processed_corrupted = []

    for item in data:
        question = item["question"]
        raw_answer = item["answer"]

        prompt = format_prompt_with_template(question, r1_zero_prompt_path)
        # Step 1: 提取参考答案和剩余文本
        ref_answer, modified_answer = extract_reference_answer(raw_answer)

        # Step 2: 移除计算器注释
        calc_removed_answer = remove_calculator_annotations(modified_answer)

        # Step 3: 构建 response 字段（干净版）
        response_clean = calc_removed_answer + "</think> <answer>" + ref_answer + "</answer>"
        processed_clean.append({
            "prompt": prompt,
            "response": response_clean
        })

        # Step 4: 构建 response 字段（有污染版）
        corrupted_ref_answer = corrupt_answer(ref_answer, corruption_rate, all_answers)
        response_corrupt = calc_removed_answer + "</think> <answer>" + corrupted_ref_answer + "</answer>"
        processed_corrupted.append({
            "prompt": prompt,
            "response": response_corrupt
        })

    # Step 5: 写入 JSONL 文件
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in processed_clean:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(output_corrupt_path, "w", encoding="utf-8") as f:
        for entry in processed_corrupted:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Processed {len(processed_clean)} items and saved to {output_path}")
    print(f"✅ Processed {len(processed_corrupted)} items (corrupted) and saved to {output_corrupt_path}")


if __name__ == "__main__":
    main()