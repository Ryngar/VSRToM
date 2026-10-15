import json
import random
from collections import defaultdict, Counter
from typing import Dict, List, Any
import copy
import os
from pathlib import Path

def shuffle_dataset_options(input_file: str, output_file: str, seed: int = 42):
    """
    对JSONL格式的benchmark数据集进行选项shuffle，保持答案分布均匀
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSONL文件路径  
        seed: 随机种子，确保结果可复现
    """
    random.seed(seed)
    
    # 读取所有数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    print(f"读取了 {len(data)} 条数据")
    
    # 分析数据结构
    analyze_data_structure(data)
    
    # 按任务类型分组处理
    task_groups = defaultdict(list)
    for item in data:
        task_type = item.get('task', 'unknown')
        task_groups[task_type].append(item)
    
    shuffled_data = []

    
    for task_type, task_data in task_groups.items():
        print(f"\n处理任务类型: {task_type} ({len(task_data)} 条数据)")
        
        # 获取选项键名（如 visibility, distribution, count）
        sample_item = task_data[0]
        option_key = list(sample_item['answer_options'].keys())[0]
        options = sample_item['answer_options'][option_key]
        option_labels = list(options.keys())
        
        print(f"选项标签: {option_labels}")
        
        # 统计原始答案分布
        original_distribution = Counter(item['answer'] for item in task_data)
        print(f"原始答案分布: {dict(original_distribution)}")
        
        # 执行shuffle
        shuffled_task_data = shuffle_task_data(task_data, option_key, option_labels)
        
        # 验证shuffle后的分布
        new_distribution = Counter(item['answer'] for item in shuffled_task_data)
        print(f"Shuffle后答案分布: {dict(new_distribution)}")
        
        shuffled_data.extend(shuffled_task_data)
    
    # 打乱整体顺序
    # random.shuffle(shuffled_data)
    shuffled_data.sort(key=lambda x: x.get('id', 0))
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in shuffled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nShuffle完成！结果保存到: {output_file}")
    
    # 最终验证
    final_verification(shuffled_data)

def analyze_data_structure(data: List[Dict]):
    """分析数据结构"""
    tasks = Counter(item.get('task') for item in data)
    answers = Counter(item.get('answer') for item in data)
    
    print("数据结构分析:")
    print(f"任务类型分布: {dict(tasks)}")
    print(f"总体答案分布: {dict(answers)}")

def shuffle_task_data(task_data: List[Dict], option_key: str, option_labels: List[str]) -> List[Dict]:
    """
    对特定任务类型的数据进行shuffle
    """
    n_options = len(option_labels)
    n_samples = len(task_data)
    target_count_per_option = n_samples // n_options
    
    print(f"每个选项目标数量: {target_count_per_option}")

    
    # 初始化计数器
    answer_counts = {label: 0 for label in option_labels}
    
    # 按原始答案分组
    answer_groups = defaultdict(list)
    for item in task_data:
        answer_groups[item['answer']].append(item)
    
    shuffled_items = []
    
    # 处理每个原始答案组
    for original_answer, items in answer_groups.items():
        print(f"处理原始答案 {original_answer}: {len(items)} 条")
        
        for item in items:
            # 创建item的深拷贝
            new_item = copy.deepcopy(item)
            
            # 生成选项的随机排列
            shuffled_labels = option_labels.copy()
            random.shuffle(shuffled_labels)
            
            # 找到当前计数最少的选项作为新答案
            min_count = min(answer_counts.values())
            available_positions = [label for label, count in answer_counts.items() 
                                 if count == min_count]
            new_answer = random.choice(available_positions)
            
            # 找到原始答案内容在新排列中的位置
            original_content = item['answer_options'][option_key][original_answer]
            
            # 重新排列选项内容
            new_options = {}
            old_options = item['answer_options'][option_key]
            
            # 确保原始正确答案的内容被分配到新答案位置
            new_options[new_answer] = original_content
            
            # 分配其他选项内容
            remaining_labels = [label for label in shuffled_labels if label != new_answer]
            remaining_old_labels = [label for label in option_labels if label != original_answer]
            
            for new_label, old_label in zip(remaining_labels, remaining_old_labels):
                new_options[new_label] = old_options[old_label]
            
            # 按照key排序
            new_options_sorted = dict(sorted(new_options.items(), key=lambda x: x[0]))
            # 更新数据
            new_item['answer'] = new_answer
            new_item['answer_options'][option_key] = new_options_sorted
            
            # 更新计数
            answer_counts[new_answer] += 1
            
            shuffled_items.append(new_item)
    
    return shuffled_items

def final_verification(data: List[Dict]):
    """最终验证数据的正确性"""
    print("\n=== 最终验证 ===")
    
    # 按任务分组验证
    task_groups = defaultdict(list)
    for item in data:
        task_groups[item['task']].append(item)
    
    for task_type, items in task_groups.items():
        answer_dist = Counter(item['answer'] for item in items)
        print(f"任务 {task_type}: {dict(answer_dist)}")
        
        # 检查是否均匀分布
        counts = list(answer_dist.values())
        if len(set(counts)) <= 1:
            print(f"  ✓ 分布均匀")
        else:
            print(f"  ⚠ 分布不均匀，差异: {max(counts) - min(counts)}")

def sample_comparison(input_file: str, output_file: str, n_samples: int = 3):
    """比较shuffle前后的样本"""
    print(f"\n=== 样本对比 (前{n_samples}条) ===")
    
    # 读取原始数据
    original_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            original_data.append(json.loads(line.strip()))
    
    # 读取shuffle后数据
    shuffled_data = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            shuffled_data.append(json.loads(line.strip()))
    
    for i in range(n_samples):
        orig = original_data[i]
        shuf = shuffled_data[i]
        
        print(f"\n--- 样本 {i+1} ---")
        print(f"原始答案: {orig['answer']}")
        print(f"新答案: {shuf['answer']}")
        
        # 显示选项变化
        task_key = list(orig['answer_options'].keys())[0]
        orig_options = orig['answer_options'][task_key]
        shuf_options = shuf['answer_options'][task_key]
        
        print("选项内容变化:")
        for label in orig_options:
            orig_text = orig_options[label]['text']
            shuf_text = shuf_options[label]['text']
            if orig_text != shuf_text:
                print(f"  {label}: '{orig_text}' → '{shuf_text}'")
        
        if orig_options[orig['answer']]['text'] != shuf_options[shuf['answer']]['text']:
            print("出错啦出错啦出错啦!!!")

import os
from pathlib import Path

def process_folder(input_folder: str, output_folder: str, seed: int = 42):
    """
    处理文件夹内的所有JSONL文件
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        seed: 随机种子
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 找到所有jsonl文件
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"在 {input_folder} 中没有找到JSONL文件")
        return
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件:")
    for file in jsonl_files:
        print(f"  - {file.name}")
    
    # 处理每个文件
    for i, input_file in enumerate(jsonl_files):
        output_file = output_path / input_file.name
        
        print(f"\n{'='*60}")
        print(f"处理文件 {i+1}/{len(jsonl_files)}: {input_file.name}")
        print(f"{'='*60}")
        
        try:
            # 为每个文件使用不同的种子，但保持可复现性
            file_seed = seed + i
            shuffle_dataset_options(str(input_file), str(output_file), seed=file_seed)
            print(f"✓ 成功处理: {input_file.name} → {output_file.name}")
            
        except Exception as e:
            print(f"✗ 处理失败: {input_file.name}")
            print(f"错误信息: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("批处理完成！")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"{'='*60}")

# 使用示例
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shuffle options and remap correct labels")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("input and output must differ")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    shuffle_dataset_options(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
