import json
import torch
import os
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer
)

# 禁用 huggingface_hub 符号链接警告
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 定义数据集类
class LlamaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建提示模板
        prompt = f"Instruction: {item['instruction']}\n"
        if item['input']:
            prompt += f"Input: {item['input']}\n"
        prompt += f"Output: {item['output']}"

        # 编码文本
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

def main():
    print("开始加载训练数据...")
    # 加载数据
    with open('llama_data.json', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    print(f"成功加载数据，共 {len(data)} 条记录")

    print("\n检测可用设备...")
    # 检测并设置设备
    if torch.cuda.is_available():
        device = "cuda"
        print("使用 NVIDIA GPU 进行训练")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("使用 Apple Silicon GPU 进行训练")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = "xpu"
        print("使用 Intel XPU 进行训练")
    elif hasattr(torch, 'hip') and torch.hip.is_available():
        device = "hip"
        print("使用 AMD GPU (ROCm) 进行训练")
    else:
        device = "cpu"
        print("未检测到支持的 GPU，使用 CPU 进行训练")

    print("\n正在初始化tokenizer和模型...")
    # 初始化tokenizer和模型
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"使用模型: {model_name}")
    
    # 先创建模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 使用模型原有的eos_token作为padding token
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"  # 确保在右侧进行padding
    
    print("模型加载完成")

    print("\n准备数据集...")
    # 准备数据集
    dataset = LlamaDataset(data, tokenizer)
    
    # 分割训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"数据集分割完成：训练集 {train_size} 条，验证集 {val_size} 条")

    print("\n配置训练参数...")
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./llama_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        disable_tqdm=False,
        report_to=["tensorboard"],
        optim="adamw_torch",
        # 添加以下设备相关参数
        no_cuda=device == "cpu",  # 当使用CPU时禁用CUDA
        use_mps_device=device == "mps",  # 使用Apple Silicon GPU
    )
    print("训练参数配置完成")

    print("\n初始化训练器...")
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    print("训练器初始化完成")

    print("\n开始训练模型...")
    # 开始训练
    trainer.train()
    print("\n训练完成！")

    print("\n保存模型...")
    # 保存模型
    trainer.save_model("./llama_final_model")
    tokenizer.save_pretrained("./llama_final_model")
    print("模型保存完成，保存路径：./llama_final_model")

if __name__ == "__main__":
    os.environ['HTTP_PROXY'] = 'http://localhost:7890'
    os.environ['HTTPS_PROXY'] = 'http://localhost:7890'
    main()
