import json
import re


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


def main():
    input_path = "data/gsm8k/train.jsonl"
    output_path = "data/gsm8k/processed_train.jsonl"
    r1_zero_prompt_path = "cs336_alignment/prompts/r1_zero.prompt"

    data = load_jsonl(input_path)

    processed_data = []
    for item in data:
        question = item["question"]
        raw_answer = item["answer"]

        prompt = format_prompt_with_template(question, r1_zero_prompt_path)
        # Step 1: 提取参考答案和剩余文本
        ref_answer, modified_answer = extract_reference_answer(raw_answer)

        # Step 2: 移除计算器注释
        calc_removed_answer = remove_calculator_annotations(modified_answer)

        # Step 3: 构建 response 字段
        # <think> is in previous prompt template
        response = calc_removed_answer + "</think> <answer>" + ref_answer + "</answer>"

        # Step 4: 添加到输出列表
        processed_data.append({
            "prompt": prompt,
            "response": response
        })

    # Step 5: 写入 JSONL 文件
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Processed {len(processed_data)} items and saved to {output_path}")


if __name__ == "__main__":
    main()