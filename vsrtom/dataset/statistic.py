import json
from collections import defaultdict, Counter

def analyze_jsonl_dataset(file_path):
    """
    分析JSONL数据集，统计每个level和task组合下answer选项的text分布
    """
    # 存储统计结果：{(level, task): {answer: [text1, text2, ...]}}
    stats = defaultdict(lambda: defaultdict(list))
    
    # 读取并解析JSONL文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # 提取关键字段
                level = data.get('level', 'unknown')
                task = data.get('task', 'unknown')
                answer = data.get('answer', 'unknown')
                
                # 获取answer对应的text
                answer_options = data.get('answer_options', {})
                task_options = answer_options.get(task, {})
                answer_text = task_options.get(answer, {}).get('text', 'unknown')
                
                # 统计
                stats[(level, task)][answer].append(answer_text)
                
            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析错误: {e}")
            except Exception as e:
                print(f"第{line_num}行处理错误: {e}")
    
    return stats

def print_statistics(stats):
    """
    打印统计结果
    """
    print("=" * 80)
    print("数据集统计结果")
    print("=" * 80)
    
    # 按level和task排序
    sorted_keys = sorted(stats.keys())
    
    for (level, task) in sorted_keys:
        print(f"\n【{level} {task}】:")
        print("-" * 50)
        
        answer_stats = stats[(level, task)]
        
        # 按answer选项排序（A, B, C, D）
        sorted_answers = sorted(answer_stats.keys())
        
        for answer in sorted_answers:
            text_list = answer_stats[answer]
            text_counter = Counter(text_list)
            
            # 格式化输出
            text_distributions = []
            for text, count in text_counter.most_common():
                text_distributions.append(f"{text}({count}个)")
            
            print(f"  {answer}: {' '.join(text_distributions)}")

def generate_detailed_report(stats):
    """
    生成详细报告
    """
    print("\n" + "=" * 80)
    print("详细统计报告")
    print("=" * 80)
    
    total_samples = 0
    for (level, task), answer_stats in stats.items():
        level_task_total = sum(len(texts) for texts in answer_stats.values())
        total_samples += level_task_total
        
        print(f"\n【{level} {task}】- 总计 {level_task_total} 个样本")
        
        for answer, text_list in sorted(answer_stats.items()):
            text_counter = Counter(text_list)
            print(f"  答案 {answer}: {len(text_list)} 个")
            for text, count in text_counter.most_common():
                percentage = (count / len(text_list)) * 100
                print(f"    └─ '{text}': {count} 个 ({percentage:.1f}%)")
    
    print(f"\n总样本数: {total_samples}")

def save_to_csv(stats, output_file='dataset_analysis.csv'):
    """
    将统计结果保存为CSV文件
    """
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Level', 'Task', 'Answer', 'Text', 'Count'])
        
        for (level, task), answer_stats in sorted(stats.items()):
            for answer, text_list in sorted(answer_stats.items()):
                text_counter = Counter(text_list)
                for text, count in text_counter.items():
                    writer.writerow([level, task, answer, text, count])
    
    print(f"\n统计结果已保存到: {output_file}")

# 主函数
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Report answer distributions")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    file_path = args.input

    print("开始分析JSONL数据集...")
    stats = analyze_jsonl_dataset(file_path)
    
    # 打印统计结果
    print_statistics(stats)
    
    # 生成详细报告
    generate_detailed_report(stats)
    
    # 保存为CSV文件
    # save_to_csv(stats)

if __name__ == "__main__":
    main()
