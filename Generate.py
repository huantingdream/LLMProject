import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(instruction, model_path='./llama_final_model', max_length=512, num_beams=5):
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # 简化的提示模板
    prompt = f"Instruction: {instruction}\nOutput:"

    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    print("Encoded input IDs:", inputs['input_ids'])

    # 生成文本
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    print("Generated output IDs:", outputs)

    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    instruction = "哮喘和铅中毒的辅助检查有什么不同？"
    generated_text = generate_text(instruction)
    print("Generated Text: ", generated_text)