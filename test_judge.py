from src.llm_evaluator import RAGEvaluator

print("="*50)
print("🚀 正在单独唤醒 9B 裁判模型进行开卷测试...")
print("="*50)

# 1. 只加载模型，不跑检索
evaluator = RAGEvaluator(generator_id="Qwen/Qwen3.5-2B", judge_id=r"D:\models\Qwen3.5-9B")

# 2. 伪造一份完美的问答数据
query = "What is the capital of France?"
context = "[Source: doc1.pdf] Paris is the capital and most populous city of France."
ground_truth = "Paris"
generated_answer = "Based on the context, the capital of France is Paris."

# 3. 拦截并打印裁判的【原始输出】
messages = [
    {"role": "system", "content": (
        "You are an impartial and expert academic judge evaluating an AI-generated answer. Evaluate based on two criteria using a 0-10 scale.\n\n"
        "CRITERIA 1: Faithfulness (0-10)\n"
        "- 10: Perfectly aligned with context.\n"
        "- 0: Contradicts context.\n\n"
        "CRITERIA 2: Relevance (0-10)\n"
        "- 10: Directly answers the question.\n"
        "- 0: Off-topic.\n\n"
        "Provide your evaluation STRICTLY in this format:\nFaithfulness: [Score]\nRelevance: [Score]"
    )},
    {"role": "user", "content": f"Context: {context}\nQuestion: {query}\nHuman Ground Truth: {ground_truth}\nAI Answer: {generated_answer}"}
]

text = evaluator.judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = evaluator.judge_tokenizer([text], return_tensors="pt").to(evaluator.device)

print("\n⏳ 裁判正在思考...")
outputs = evaluator.judge_model.generate(
    model_inputs.input_ids, max_new_tokens=50, temperature=0.1, pad_token_id=evaluator.judge_tokenizer.eos_token_id
)

generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)]
judgment = evaluator.judge_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print("\n" + "="*50)
print("🎯 【裁判的原始输出】(请把下面这段发给我看)：")
print(f"[{judgment}]")
print("="*50)