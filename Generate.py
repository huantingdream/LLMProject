import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(prompt, model_path='./llama_final_model', max_length=50):
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors='pt')

    # 生成文本
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    prompt = "治疗慢性胰腺炎和治疗抑郁症有什么不同之处？"
    generated_text = generate_text(prompt)
    print("Generated Text: ", generated_text)