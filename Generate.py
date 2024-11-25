import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", torch.float32
    else:
        return "cpu", torch.float32

def generate_text(instruction, model_path='./llama_final_model', max_length=512, num_beams=5):
    # 检测设备
    device, dtype = get_device()
    print(f"使用设备: {device}")
    
    if device == "cuda":
        print("使用 NVIDIA GPU 进行推理")
    elif device == "mps":
        print("使用 Apple Silicon GPU 进行推理")
    else:
        print("使用 CPU 进行推理")

    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,  # 根据设备选择精度
        device_map="auto" if device == "cuda" else None  # 只在CUDA时使用device_map
    )
    
    # 对于MPS和CPU，需要手动将模型移到设备上
    if device != "cuda":
        model = model.to(device)

    # 简化的提示模板
    prompt = f"Instruction: {instruction}\nOutput:"

    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    
    # 将输入移到相应设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 生成文本
    with torch.no_grad():  # 推理时不需要梯度
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    instruction = "哮喘和铅中毒的辅助检查有什么不同？"
    generated_text = generate_text(instruction)
    print("\n问题:", instruction)
    print("\n回答:", generated_text)