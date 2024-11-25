import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer
)

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

    print("\n正在初始化tokenizer和模型...")
    # 初始化tokenizer和模型
    model_name = "Qwen/Qwen-7B"
    print(f"使用模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
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
        # 添加进度条显示
        disable_tqdm=False,
        # 添加日志输出
        report_to=["tensorboard"],
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
    main()
