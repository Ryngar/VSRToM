import json
import random
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

# 常量定义
LEVELS = ['0-order', '1st-order', '2nd-order']  # camera, woman, man perspectives
TASKS = ['visibility', 'count', 'layout']

def load_data(file_path: str) -> List[Dict]:
    """Load data from jsonl file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_target_samples(target_matrix: List[List[int]]) -> Dict[Tuple[str, str], int]:
    """
    Create target sample counts for each level-task combination
    
    Args:
        target_matrix: 3x3 matrix where target_matrix[level_idx][task_idx] = target_count
        
    Returns:
        target_samples: Dictionary mapping (level, task) to target count
    """
    target_samples = {}
    for i, level in enumerate(LEVELS):
        for j, task in enumerate(TASKS):
            target_samples[(level, task)] = target_matrix[i][j]
    return target_samples

def extract_answer_patterns(data: List[Dict]) -> List[Tuple[int, str]]:
    """
    Extract answer patterns for all samples, excluding patterns with excluded/ambiguous answers
    
    Args:
        data: List of data samples
        
    Returns:
        patterns: List of (sample_id, pattern) tuples where pattern includes:
                 - 9 answer characters (3 levels × 3 tasks)
                 - 4 visibility analysis flags
    """
    excluded_count = 0
    patterns = []
    
    for sample in data:
        sample_id = sample['sample_id']
        answers = sample['answers']
        
        # Extract visibility analysis parameters
        man_cant_sees_balls_by_behind = sample['not_visible_analysis']['man_cant_sees_balls_by_behind']
        man_cant_sees_balls_by_obs = sample['not_visible_analysis']['man_cant_sees_balls_by_obs']
        woman_cant_sees_balls_by_behind = sample['not_visible_analysis']['woman_cant_sees_balls_by_behind']
        woman_cant_sees_balls_by_obs = sample['not_visible_analysis']['woman_cant_sees_balls_by_obs']
        
        if sample_id <= 5:
            print(f"Sample {sample_id} answers: {answers}")
            print(f"  Visibility analysis: man_behind={man_cant_sees_balls_by_behind}, "
                  f"man_obs={man_cant_sees_balls_by_obs}, "
                  f"woman_behind={woman_cant_sees_balls_by_behind}, "
                  f"woman_obs={woman_cant_sees_balls_by_obs}")
        
        # Skip samples that contain any excluded/ambiguous answers
        excluded_patterns = ['EXCLUDED_AMBIGUOUS', 'EXCLUDED_NO_SIGHT', 'EXCLUDED']
        if any(pattern in str(answers) for pattern in excluded_patterns):
            excluded_count += 1
            continue
        
        # Create extended pattern: 9 answers + 4 visibility flags
        pattern = ''.join(answers[0:9])
        pattern += str(int(man_cant_sees_balls_by_behind))
        pattern += str(int(man_cant_sees_balls_by_obs))
        pattern += str(int(woman_cant_sees_balls_by_behind))
        pattern += str(int(woman_cant_sees_balls_by_obs))
        
        patterns.append((sample_id, pattern))
    
    print(f"Debug: Total excluded samples: {excluded_count}")
    return patterns

def analyze_pattern_distribution(patterns: List[Tuple[int, str]]) -> Dict[str, int]:
    """
    Analyze the distribution of answer patterns
    
    Args:
        patterns: List of (sample_id, pattern) tuples
        
    Returns:
        distribution: Pattern count statistics
    """
    pattern_counts = Counter([pattern for _, pattern in patterns])
    return dict(pattern_counts)

def sample_with_pattern_balance(pattern_groups: Dict[str, List[int]], target_count: int, 
                              answer_position: int, task: str) -> Tuple[List[int], Dict[str, Any]]:
    """
    Sample with pattern balance using layered allocation strategy
    
    Returns:
        Tuple of (selected_sample_ids, sampling_stats)
    """
    if not pattern_groups or target_count <= 0:
        return [], {}
    
    # 获取每个模式的样本数量
    pattern_counts = {pattern: len(sample_ids) for pattern, sample_ids in pattern_groups.items()}
    num_patterns = len(pattern_counts)
    
    sampling_stats = {
        'total_available_patterns': num_patterns,
        'total_available_samples': sum(pattern_counts.values()),
        'target_count': target_count,
        'pattern_allocation': {},
        'sampling_strategy': ''
    }
    
    if num_patterns >= target_count:
        # 模式数量够多，每个模式选1个
        selected_patterns = random.sample(list(pattern_groups.keys()), target_count)
        selected_samples = []
        for pattern in selected_patterns:
            selected_samples.append(random.choice(pattern_groups[pattern]))
            sampling_stats['pattern_allocation'][pattern] = 1
        sampling_stats['sampling_strategy'] = f"Selected {target_count} patterns, 1 sample each"
        print(f"  Strategy: Selected {target_count} patterns, 1 sample each")
        return selected_samples, sampling_stats
    
    # 逐层分配算法
    allocation = {pattern: 0 for pattern in pattern_counts.keys()}
    remaining_counts = pattern_counts.copy()
    remaining_target = target_count
    
    while remaining_target > 0:
        # 找到还有样本的模式中的最小样本数
        active_patterns = {p: count for p, count in remaining_counts.items() if count > 0}
        if not active_patterns:
            break
        
        min_count = min(active_patterns.values())
        num_active = len(active_patterns)
        
        # 计算这轮每个模式分配多少
        per_pattern = min(min_count, remaining_target // num_active)
        if per_pattern == 0:
            # 剩余目标数不够平均分配，随机分配给前几个模式
            patterns_to_assign = random.sample(list(active_patterns.keys()), remaining_target)
            for pattern in patterns_to_assign:
                allocation[pattern] += 1
            break
        
        # 给每个活跃模式分配样本
        for pattern in active_patterns.keys():
            allocation[pattern] += per_pattern
            remaining_counts[pattern] -= per_pattern
            remaining_target -= per_pattern
    
    # 执行实际采样
    selected_samples = []
    for pattern, count in allocation.items():
        if count > 0:
            sample_ids = pattern_groups[pattern]
            if count >= len(sample_ids):
                selected_samples.extend(sample_ids)
                sampling_stats['pattern_allocation'][pattern] = len(sample_ids)
            else:
                selected_samples.extend(random.sample(sample_ids, count))
                sampling_stats['pattern_allocation'][pattern] = count
    
    sampling_stats['sampling_strategy'] = f"Layered allocation - {dict(allocation)}"
    print(f"  Strategy: Layered allocation - {dict(allocation)}")
    return selected_samples, sampling_stats

def sample_for_level_task(level: str, task: str, patterns: List[Tuple[int, str]], target_count: int) -> Tuple[List[int], Dict[str, Any]]:
    """
    Sample data for a specific level-task combination using answer-balanced sampling
    
    Returns:
        Tuple of (selected_sample_ids, sampling_stats)
    """
    if target_count <= 0:
        return [], {}
        
    task_idx = TASKS.index(task)
    level_idx = LEVELS.index(level)
    
    # Calculate the answer position for this level-task combination
    answer_position = level_idx * 3 + task_idx
    
    # Step 1: Group samples by their answer at the target position
    answer_groups = defaultdict(list)
    for sample_id, pattern in patterns:
        if answer_position < len(pattern):
            answer = pattern[answer_position]
            answer_groups[answer].append((sample_id, pattern))
    
    # Step 2: Calculate target samples per answer type
    answer_types = list(answer_groups.keys())
    num_answer_types = len(answer_types)
    
    if num_answer_types == 0:
        return [], {}
    
    samples_per_answer = target_count // num_answer_types
    remaining_samples = target_count % num_answer_types
    
    sampling_stats = {
        'total_available_answers': num_answer_types,
        'answer_types': answer_types,
        'samples_per_answer': samples_per_answer,
        'remaining_samples': remaining_samples,
        'answer_allocations': {},
        'pattern_distributions': {}
    }
    
    # Step 3: Sample from each answer group
    selected_samples = []
    
    for i, answer in enumerate(answer_types):
        # Calculate target for this answer (distribute remaining samples to first few answers)
        answer_target = samples_per_answer + (1 if i < remaining_samples else 0)
        
        # Group patterns within this answer group
        answer_pattern_groups = defaultdict(list)
        for sample_id, pattern in answer_groups[answer]:
            answer_pattern_groups[pattern].append(sample_id)
        
        # Sample with pattern balance within this answer group
        answer_samples, answer_sampling_stats = sample_with_pattern_balance(
            answer_pattern_groups, answer_target, answer_position, f"{task}_{answer}"
        )
        
        selected_samples.extend(answer_samples)
        
        # Record statistics for this answer
        sampling_stats['answer_allocations'][answer] = {
            'target': answer_target,
            'actual': len(answer_samples),
            'available_patterns': len(answer_pattern_groups),
            'available_samples': len(answer_groups[answer]),
            'sampling_strategy': answer_sampling_stats.get('sampling_strategy', ''),
            'pattern_allocation': answer_sampling_stats.get('pattern_allocation', {})
        }
        
        # Count pattern distribution for this answer
        pattern_counts = defaultdict(int)
        sample_id_set = set(answer_samples)
        for sample_id, pattern in answer_groups[answer]:
            if sample_id in sample_id_set:
                pattern_counts[pattern] += 1
        sampling_stats['pattern_distributions'][answer] = dict(pattern_counts)
    
    # Overall sampling strategy description
    sampling_stats['sampling_strategy'] = (
        f"Answer-balanced sampling: {num_answer_types} answer types, "
        f"{samples_per_answer}±1 samples per answer"
    )
    
    print(f"  Answer-balanced strategy: {num_answer_types} answer types ({answer_types})")
    print(f"  Target per answer: {samples_per_answer}±1")
    
    return selected_samples, sampling_stats

def get_sampled_pattern_distribution(sample_ids: List[int], level: str, task: str, patterns: List[Tuple[int, str]]) -> Dict[str, Any]:
    """Helper function to analyze pattern distribution in sampled data"""
    sample_id_set = set(sample_ids)
    
    # Count patterns in sampled data
    pattern_counts = defaultdict(int)
    answer_counts = defaultdict(int)
    visibility_counts = defaultdict(int)
    
    # Calculate answer position for this level-task combination
    level_idx = LEVELS.index(level)
    task_idx = TASKS.index(task)
    answer_position = level_idx * 3 + task_idx
    
    for sample_id, pattern in patterns:
        if sample_id in sample_id_set:
            pattern_counts[pattern] += 1
            
            # Count specific answer at target position
            if answer_position < len(pattern):
                answer = pattern[answer_position]
                answer_counts[answer] += 1
            
            # Count visibility flags (last 4 characters)
            if len(pattern) >= 13:  # 9 answers + 4 visibility flags
                vis_flags = pattern[-4:]
                visibility_counts[f'visibility_{vis_flags}'] += 1
    
    return {
        'patterns': dict(pattern_counts),
        'answer_distribution': dict(answer_counts),
        'visibility_distribution': dict(visibility_counts)
    }

def generate_balanced_dataset(data: List[Dict], target_samples: Dict[Tuple[str, str], int], 
                            seed: int = 42, verbose: bool = True) -> Tuple[Dict[Tuple[str, str], List[int]], Dict]:
    """
    Generate balanced dataset according to the sampling strategy
    
    Returns:
        Tuple of (sampled_data, sampling_statistics)
    """
    random.seed(seed)
    
    # Extract patterns
    patterns = extract_answer_patterns(data)
    
    # Generate samples for each level-task combination
    sampled_data = {}
    sampling_statistics = {
        'seed': seed,
        'total_patterns': len(patterns),
        'level_task_stats': {},
        'summary': {}
    }
    
    for level in LEVELS:
        for task in TASKS:
            target_count = target_samples.get((level, task), 0)
            if target_count > 0:
                sample_ids, sampling_stats = sample_for_level_task(level, task, patterns, target_count)
                sampled_data[(level, task)] = sample_ids
                
                # Get detailed pattern distribution for sampled data
                sampled_patterns = get_sampled_pattern_distribution(sample_ids, level, task, patterns)
                
                # Store comprehensive statistics
                sampling_statistics['level_task_stats'][(level, task)] = {
                    'target_count': target_count,
                    'actual_count': len(sample_ids),
                    'sampling_strategy': sampling_stats.get('sampling_strategy', ''),
                    'total_available_answers': sampling_stats.get('total_available_answers', 0),
                    'answer_types': sampling_stats.get('answer_types', []),
                    'samples_per_answer': sampling_stats.get('samples_per_answer', 0),
                    'answer_allocations': sampling_stats.get('answer_allocations', {}),
                    'pattern_distributions_by_answer': sampling_stats.get('pattern_distributions', {}),
                    'sampled_pattern_distribution': sampled_patterns['patterns'],
                    'answer_distribution': sampled_patterns['answer_distribution'],
                    'visibility_distribution': sampled_patterns['visibility_distribution'],
                    'pattern_diversity': len(sampled_patterns['patterns']),
                    'sample_ids': sample_ids
                }
                
                if verbose:
                    print(f"Sampled {len(sample_ids)} samples for {level}-{task} (target: {target_count})")
                    print(f"  Answer distribution: {sampled_patterns['answer_distribution']}")
                    print(f"  Pattern diversity: {len(sampled_patterns['patterns'])} unique patterns")
                    
                    # Show answer allocation details
                    answer_allocations = sampling_stats.get('answer_allocations', {})
                    for answer, alloc in answer_allocations.items():
                        print(f"    Answer '{answer}': {alloc['actual']}/{alloc['target']} samples, {alloc['available_patterns']} patterns")
                    
                    print(f"  Top 3 patterns: {dict(list(sorted(sampled_patterns['patterns'].items(), key=lambda x: x[1], reverse=True))[:3])}")
    
    # Generate summary statistics
    sampling_statistics['summary'] = {
        'total_sampled': sum(len(sample_ids) for sample_ids in sampled_data.values()),
        'total_target': sum(target_samples.values()),
        'level_task_combinations': len(sampled_data),
        'overall_pattern_diversity': len(set().union(*[stats['sampled_pattern_distribution'].keys() 
                                                       for stats in sampling_statistics['level_task_stats'].values()]))
    }
    
    return sampled_data, sampling_statistics

def get_question_answer_index(level: str, task: str) -> int:
    """
    Get the question/answer index for a specific level-task combination
    """
    level_idx = LEVELS.index(level)
    task_idx = TASKS.index(task)
    return level_idx * 3 + task_idx

def export_sampled_dataset(data: List[Dict], sampled_data: Dict[Tuple[str, str], List[int]], 
                          sampling_statistics: Dict, output_dir: str):
    """
    Export sampled datasets to separate files and save comprehensive statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create lookup for quick sample retrieval
    sample_lookup = {sample['sample_id']: sample for sample in data}
    
    # Collect all samples for the combined dataset
    all_combined_samples = []
    
    for (level, task), sample_ids in sampled_data.items():
        # Get the question/answer index for this level-task combination
        qa_index = get_question_answer_index(level, task)
        
        # Create filtered samples with only the relevant question and answer
        filtered_samples = []
        for sid in sample_ids:
            if sid in sample_lookup:
                original_sample = sample_lookup[sid]
                
                # Create new sample with only the relevant question and answer
                filtered_sample = {
                    'sample_id': original_sample['sample_id'],
                    'metadata': original_sample['metadata'],
                    'question': original_sample['questions'][qa_index],
                    'answer': original_sample['answers'][qa_index],
                    'answer_options': {task: original_sample['answer_options'][task]},
                    'level': level,
                    'task': task,
                }
                filtered_samples.append(filtered_sample)
                
                # Add to combined dataset
                all_combined_samples.append(filtered_sample)
        
        # Save individual level-task file
        output_file = os.path.join(output_dir, f"{level}_{task}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in filtered_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Exported {len(filtered_samples)} samples to {output_file}")
    
    # Save combined dataset with all samples
    combined_output_file = os.path.join(output_dir, "VSRToM_Bench.jsonl")
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        for sample in all_combined_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Exported {len(all_combined_samples)} total samples to {combined_output_file}")
    
    # Convert tuple keys to strings for JSON serialization
    json_compatible_stats = convert_tuples_to_strings(sampling_statistics)
    
    # Save comprehensive sampling statistics
    stats_file = os.path.join(output_dir, "sampling_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(json_compatible_stats, f, ensure_ascii=False, indent=2)
    print(f"Saved sampling statistics to {stats_file}")
    
    # Save a human-readable summary
    summary_file = os.path.join(output_dir, "sampling_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("SAMPLING SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall summary
        summary = sampling_statistics['summary']
        f.write(f"Overall Statistics:\n")
        f.write(f"  Total samples targeted: {summary['total_target']}\n")
        f.write(f"  Total samples obtained: {summary['total_sampled']}\n")
        f.write(f"  Level-task combinations: {summary['level_task_combinations']}\n")
        f.write(f"  Overall pattern diversity: {summary['overall_pattern_diversity']} unique patterns\n")
        f.write(f"  Random seed: {sampling_statistics['seed']}\n")
        f.write(f"  Combined dataset file: VSRToM_Bench.jsonl\n\n")
        
        # File summary
        f.write(f"Generated Files:\n")
        f.write(f"  VSRToM_Bench.jsonl - All {len(all_combined_samples)} samples combined\n")
        for (level, task) in sampled_data.keys():
            count = len(sampled_data[(level, task)])
            f.write(f"  {level}_{task}.jsonl - {count} samples\n")
        f.write("\n")
        
        # Detailed breakdown by level-task
        for (level, task), stats in sampling_statistics['level_task_stats'].items():
            f.write(f"{level.upper()} - {task.upper()}:\n")
            f.write(f"  Target: {stats['target_count']}, Actual: {stats['actual_count']}\n")
            f.write(f"  Strategy: {stats['sampling_strategy']}\n")
            f.write(f"  Available answer types: {stats['total_available_answers']}\n")
            f.write(f"  Answer types: {stats['answer_types']}\n")
            f.write(f"  Samples per answer: {stats['samples_per_answer']}\n")
            f.write(f"  Pattern diversity: {stats['pattern_diversity']}\n")
            
            # Answer allocation details
            f.write(f"  Answer allocations:\n")
            for answer, alloc in stats['answer_allocations'].items():
                f.write(f"    Answer '{answer}': {alloc['actual']}/{alloc['target']} samples\n")
                f.write(f"      Available patterns: {alloc['available_patterns']}, Available samples: {alloc['available_samples']}\n")
                f.write(f"      Strategy: {alloc['sampling_strategy']}\n")
                # Show top patterns for this answer
                if alloc['pattern_allocation']:
                    top_patterns = sorted(alloc['pattern_allocation'].items(), key=lambda x: x[1], reverse=True)[:3]
                    f.write(f"      Top patterns: {dict(top_patterns)}\n")
            
            # Overall answer distribution
            f.write(f"  Final answer distribution: {stats['answer_distribution']}\n")
            
            # Visibility distribution  
            f.write(f"  Visibility distribution: {stats['visibility_distribution']}\n")
            f.write("\n")
    
    print(f"Saved human-readable summary to {summary_file}")
    
    # Save a dataset overview file
    overview_file = os.path.join(output_dir, "dataset_overview.txt")
    with open(overview_file, 'w', encoding='utf-8') as f:
        f.write("DATASET OVERVIEW\n")
        f.write("=" * 30 + "\n\n")
        
        f.write(f"Total samples: {len(all_combined_samples)}\n")
        f.write(f"Random seed: {sampling_statistics['seed']}\n\n")
        
        # Count by level and task
        level_counts = defaultdict(int)
        task_counts = defaultdict(int)
        answer_counts = defaultdict(int)
        
        for sample in all_combined_samples:
            level_counts[sample['level']] += 1
            task_counts[sample['task']] += 1
            answer_counts[sample['answer']] += 1
        
        f.write("Distribution by Level:\n")
        for level, count in sorted(level_counts.items()):
            f.write(f"  {level}: {count} samples\n")
        
        f.write("\nDistribution by Task:\n")
        for task, count in sorted(task_counts.items()):
            f.write(f"  {task}: {count} samples\n")
        
        f.write("\nDistribution by Answer:\n")
        for answer, count in sorted(answer_counts.items()):
            f.write(f"  {answer}: {count} samples\n")
        
        f.write(f"\nLevel-Task Matrix:\n")
        f.write("Level\\Task".ljust(12))
        for task in TASKS:
            f.write(task.ljust(12))
        f.write("\n")
        
        for level in LEVELS:
            f.write(level.ljust(12))
            for task in TASKS:
                count = len(sampled_data.get((level, task), []))
                f.write(str(count).ljust(12))
            f.write("\n")
    
    print(f"Saved dataset overview to {overview_file}")

def convert_tuples_to_strings(obj):
    """
    Recursively convert tuple keys to strings for JSON serialization
    """
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            # Convert tuple keys to string representation
            if isinstance(key, tuple):
                new_key = f"{key[0]}_{key[1]}"
            else:
                new_key = key
            new_dict[new_key] = convert_tuples_to_strings(value)
        return new_dict
    elif isinstance(obj, list):
        return [convert_tuples_to_strings(item) for item in obj]
    else:
        return obj

def print_dataset_statistics(data: List[Dict]):
    """Print statistics about data filtering"""
    total_samples = len(data)
    # Check for various exclusion patterns
    excluded_patterns = ['EXCLUDED_AMBIGUOUS', 'EXCLUDED_NO_SIGHT', 'EXCLUDED']
    samples_with_excluded = sum(1 for sample in data 
                               if any(pattern in str(sample['answers']) for pattern in excluded_patterns))
    valid_samples = total_samples - samples_with_excluded
    
    print(f"Dataset statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Samples with excluded answers: {samples_with_excluded}")
    print(f"  Valid samples (no excluded/ambiguous): {valid_samples}")
    print(f"  Filtering ratio: {valid_samples/total_samples:.2%}")
    print()

def analyze_visibility_distribution(data: List[Dict]):
    """Analyze the distribution of visibility analysis parameters"""
    visibility_stats = {
        'man_cant_sees_balls_by_behind': defaultdict(int),
        'man_cant_sees_balls_by_obs': defaultdict(int),
        'woman_cant_sees_balls_by_behind': defaultdict(int),
        'woman_cant_sees_balls_by_obs': defaultdict(int)
    }
    
    for sample in data:
        # Skip excluded samples
        excluded_patterns = ['EXCLUDED_AMBIGUOUS', 'EXCLUDED_NO_SIGHT', 'EXCLUDED']
        if any(pattern in str(sample['answers']) for pattern in excluded_patterns):
            continue
            
        for key in visibility_stats.keys():
            value = sample['not_visible_analysis'][key]
            visibility_stats[key][value] += 1
    
    print("Visibility analysis distribution:")
    for param, counts in visibility_stats.items():
        print(f"  {param}: {dict(counts)}")
    print()

# Main execution
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Filter and sample nine VSRToM categories")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-category", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_category <= 0:
        parser.error("--per-category must be positive")
    data = load_data(args.input)
    if not data:
        parser.error("input contains no samples")
    targets = create_target_samples([[args.per_category] * 3 for _ in range(3)])
    selected, stats = generate_balanced_dataset(data, targets, seed=args.seed)
    export_sampled_dataset(data, selected, stats, args.output_dir)
    actual = stats["summary"]["total_sampled"]
    expected = args.per_category * 9
    if actual != expected:
        print(f"WARNING: insufficient eligible samples: {actual}/{expected}. "
              "Increase the candidate pool; inspect sampling_statistics.json.")


if __name__ == "__main__":
    main()
