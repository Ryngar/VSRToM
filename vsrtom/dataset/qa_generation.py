#!/usr/bin/env python3
"""
Stage 2: QA Generation (Modified Visibility Rules with Person Blocking)
集成了修改后的can_see函数，包含人员阻挡检测
"""

import json
import math
import argparse
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum

class VisibilityState(Enum):
    """Three-state visibility system"""
    AMBIGUOUS = "AMBIGUOUS"
    VISIBLE = "VISIBLE" 
    NOT_VISIBLE_BY_OBS = "NOT_VISIBLE_BY_OBS"
    NOT_VISIBLE_BY_BEHIND = "NOT_VISIBLE_BY_BEHIND"
    OVERLAP = "OVERLAP"

def normalize_coordinates_for_line_check(pos: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """
    将不完美的25宫格坐标转换为完美的25宫格坐标（仅用于直线检测）
    x: -5,-3,0,1.5,3 -> -2,-1,0,1,2
    y: -4,-2,0,2,4 -> -2,-1,0,1,2
    z: 保持不变
    """
    x, y, z = pos
    
    # x坐标映射
    x_mapping = {-5: -2, -3: -1, 0: 0, 1.5: 1, 3: 2}
    # y坐标映射  
    y_mapping = {-4: -2, -2: -1, 0: 0, 2: 1, 4: 2}
    
    # 找到最接近的坐标值
    normalized_x = x_mapping.get(x, round(x))
    normalized_y = y_mapping.get(y, round(y))
    
    return (normalized_x, normalized_y, int(z))

def load_metadata(filename: str) -> List[Dict]:
    """Load metadata from JSONL file."""
    metadata_list = []
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    metadata_list.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
        print(f"Successfully loaded {len(metadata_list)} metadata samples from {filename}")
    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found")
        raise
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        raise
    return metadata_list

def vector_subtract(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Calculate vector from pos2 to pos1."""
    return (pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2])

def vector_magnitude(vec: Tuple[float, float, float]) -> float:
    """Calculate magnitude of a vector."""
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def vector_normalize(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a vector."""
    mag = vector_magnitude(vec)
    if mag == 0:
        return (0, 0, 0)
    return (vec[0]/mag, vec[1]/mag, vec[2]/mag)

def dot_product(vec1: Tuple[float, float, float], vec2: Tuple[float, float, float]) -> float:
    """Calculate dot product of two vectors."""
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]

def cross_product(vec1: Tuple[float, float, float], vec2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Calculate cross product of two vectors."""
    return (
        vec1[1] * vec2[2] - vec1[2] * vec2[1],
        vec1[2] * vec2[0] - vec1[0] * vec2[2], 
        vec1[0] * vec2[1] - vec1[1] * vec2[0]
    )

def cross_product_z(vec1: Tuple[float, float, float], vec2: Tuple[float, float, float]) -> float:
    """Calculate z component of cross product (for 2D plane)."""
    return vec1[0] * vec2[1] - vec1[1] * vec2[0]

def angle_between_vectors(vec1: Tuple[float, float, float], vec2: Tuple[float, float, float]) -> float:
    """Calculate angle between two vectors in degrees."""
    mag1 = vector_magnitude(vec1)
    mag2 = vector_magnitude(vec2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    cos_angle = dot_product(vec1, vec2) / (mag1 * mag2)
    # Clamp to avoid numerical errors
    cos_angle = max(-1, min(1, cos_angle))
    
    return math.degrees(math.acos(cos_angle))

def are_vectors_collinear(vec1: Tuple[float, float, float], vec2: Tuple[float, float, float], tolerance: float = 1e-6) -> bool:
    """检查两个向量是否共线"""
    cross = cross_product(vec1, vec2)
    cross_magnitude = vector_magnitude(cross)
    return cross_magnitude < tolerance

def is_obstacle_on_line(observer_pos: Tuple[float, float, float], 
                     target_pos: Tuple[float, float, float], 
                     obstacle_pos: Tuple[float, float, float],
                     tolerance: float = 1e-6) -> bool:
    """
    检测人是否在observer和target之间的直线上
    使用向量共线方法，并转换为完美网格坐标
    """
    # 转换为完美网格坐标
    norm_observer = normalize_coordinates_for_line_check(observer_pos)
    norm_target = normalize_coordinates_for_line_check(target_pos)
    norm_obstacle = normalize_coordinates_for_line_check(obstacle_pos)
    
    # 计算向量（使用转换后的坐标）
    observer_to_obstacle = vector_subtract(norm_obstacle, norm_observer)
    observer_to_target = vector_subtract(norm_target, norm_observer)
    
    # 检查两向量是否共线
    if not are_vectors_collinear(observer_to_obstacle, observer_to_target, tolerance):
        return False
    
    # 检查person是否在observer和target之间
    dist_observer_to_obstacle = vector_magnitude(observer_to_obstacle)
    dist_observer_to_target = vector_magnitude(observer_to_target)
    
    # person必须在observer和target之间，且不能与observer重合
    return 0 < dist_observer_to_obstacle < dist_observer_to_target

def is_on_line_simple(observer_pos: Tuple[float, float, float], 
                     target_pos: Tuple[float, float, float], 
                     obstacle_pos: Tuple[float, float, float],
                     tolerance: float = 1e-6) -> bool:
    """
    简单的直线检测（用于障碍物检测）
    转换为完美网格坐标后检查是否在同一条线上
    """
    # 转换为完美网格坐标
    norm_obs = normalize_coordinates_for_line_check(observer_pos)
    norm_tar = normalize_coordinates_for_line_check(target_pos)
    norm_obst = normalize_coordinates_for_line_check(obstacle_pos)
    
    obs_x, obs_y, obs_z = norm_obs
    tar_x, tar_y, tar_z = norm_tar
    obst_x, obst_y, obst_z = norm_obst
    
    # 检查是否在x轴上共线
    if (abs(obs_x - tar_x) < tolerance and abs(obs_x - obst_x) < tolerance):
        min_y = min(obs_y, tar_y)
        max_y = max(obs_y, tar_y)
        return min_y <= obst_y <= max_y
    
    # 检查是否在y轴上共线
    if (abs(obs_y - tar_y) < tolerance and abs(obs_y - obst_y) < tolerance):
        min_x = min(obs_x, tar_x)
        max_x = max(obs_x, tar_x)
        return min_x <= obst_x <= max_x
    
    return False

def can_see(observer_pos: Tuple[float, float, float], 
           observer_orientation: Tuple[float, float, float], 
           target_pos: Tuple[float, float, float],
           obstacles: List[Dict] = [],
           other_person_pos: Optional[Tuple[float, float, float]] = None,
           angle_ambiguity_threshold: float = 1e-6,
           obstacle_angle_threshold: float = 135.0) -> VisibilityState:
    """
    修改后的可见性检查函数，包含四个规则：
    1. 如果点积 < 0，一定看不见 (NOT_VISIBLE)
    2. 如果点积 = 0，且observer和target中间存在障碍物，一定看不见 (NOT_VISIBLE)
    3. 如果点积 > 0，且所有障碍物角度 <= 135度，且连线直线不存在另一个人，看得见 (VISIBLE)
    4. 否则有歧义 (AMBIGUOUS)
    """
    if observer_pos == target_pos:
        return VisibilityState.OVERLAP
    
    # 计算从observer到target的向量
    to_target = vector_subtract(target_pos, observer_pos)
    
    # 计算点积
    dot_prod = dot_product(observer_orientation, to_target)
    
    # 规则1: 如果点积 < 0，一定看不见
    if dot_prod < 0:
        return VisibilityState.NOT_VISIBLE_BY_BEHIND
    
    # 规则2: 如果点积 = 0，且障碍物存在于observer-target线上，一定看不见
    if abs(dot_prod) < angle_ambiguity_threshold:
        for obstacle in obstacles:
            obstacle_pos = tuple(obstacle["position"])
            if is_on_line_simple(observer_pos, target_pos, obstacle_pos):
                return VisibilityState.NOT_VISIBLE_BY_OBS
        return VisibilityState.AMBIGUOUS
    
    # 规则3: 如果点积 > 0，检查角度条件和人员阻挡
    if dot_prod > 0:
        # 检查所有障碍物的角度条件
        for obstacle in obstacles:
            obstacle_pos = tuple(obstacle["position"])
            obstacle_to_observer = vector_subtract(observer_pos, obstacle_pos)
            obstacle_to_target = vector_subtract(target_pos, obstacle_pos)
            angle_between = angle_between_vectors(obstacle_to_observer, obstacle_to_target)
            
            if is_obstacle_on_line(observer_pos, target_pos, obstacle_pos):
                return VisibilityState.NOT_VISIBLE_BY_OBS
            if angle_between > obstacle_angle_threshold:
                return VisibilityState.AMBIGUOUS
        
        # 检查是否有另一个人阻挡视线
        if other_person_pos is not None:
            if is_obstacle_on_line(observer_pos, target_pos, other_person_pos):
                return VisibilityState.AMBIGUOUS
        
        # 所有条件满足，可见
        return VisibilityState.VISIBLE
    
    # 默认情况：返回有歧义
    return VisibilityState.AMBIGUOUS

def has_any_ambiguity(observer_pos: Tuple[float, float, float], 
                     observer_orientation: Tuple[float, float, float], 
                     target_positions: List[Tuple[float, float, float]],
                     obstacles: List[Dict],
                     other_person_pos: Optional[Tuple[float, float, float]] = None,
                     angle_ambiguity_threshold: float = 1e-6,
                     obstacle_angle_threshold: float = 135.0) -> bool:
    """检查是否有任何目标存在歧义"""
    for target_pos in target_positions:
        visibility_state = can_see(
            observer_pos, observer_orientation, target_pos, obstacles,
            other_person_pos, angle_ambiguity_threshold, obstacle_angle_threshold
        )
        if visibility_state == VisibilityState.AMBIGUOUS:
            return True
    return False

def get_visible_targets(observer_pos: Tuple[float, float, float], 
                       observer_orientation: Tuple[float, float, float], 
                       target_positions: List[Tuple[float, float, float]],
                       obstacles: List[Dict],
                       other_person_pos: Optional[Tuple[float, float, float]] = None,
                       angle_ambiguity_threshold: float = 1e-6,
                       obstacle_angle_threshold: float = 135.0) -> List[Tuple[float, float, float]]:
    """获取可见目标列表（排除有歧义的）"""
    visible_targets = []
    for target_pos in target_positions:
        visibility_state = can_see(
            observer_pos, observer_orientation, target_pos, obstacles,
            other_person_pos, angle_ambiguity_threshold, obstacle_angle_threshold
        )
        if visibility_state == VisibilityState.VISIBLE:
            visible_targets.append(target_pos)
    return visible_targets

def get_spatial_distribution(observer_pos: Tuple[float, float, float], 
                            observer_orientation: Tuple[float, float, float], 
                            target_pos: Tuple[float, float, float]) -> Optional[str]:
    """Get spatial distribution of target relative to observer."""
    to_target = vector_subtract(target_pos, observer_pos)
    
    # Use cross product to determine left/right
    cross_z = cross_product_z(observer_orientation, to_target)
    
    if abs(cross_z) < 1e-6:  # Nearly zero, in front
        return "center"
    elif cross_z > 0:
        return "left"
    else:
        return "right"

def count_not_visible_balls(observer_pos: Tuple[float, float, float], 
                       observer_orientation: Tuple[float, float, float], 
                       target_positions: List[Tuple[float, float, float]],
                       obstacles: List[Dict],
                       other_person_pos: Optional[Tuple[float, float, float]] = None,
                       course: VisibilityState = VisibilityState.NOT_VISIBLE_BY_BEHIND,
                       angle_ambiguity_threshold: float = 1e-6,
                       obstacle_angle_threshold: float = 135.0) -> int:
    not_visible_balls = 0
    for target_pos in target_positions:
        visibility_state = can_see(
            observer_pos, observer_orientation, target_pos, obstacles,
            other_person_pos, angle_ambiguity_threshold, obstacle_angle_threshold
        )
        if visibility_state == course:
            not_visible_balls += 1
    return not_visible_balls



def format_visibility_answer(is_visible: bool) -> str:
    """Format visibility answer as multiple choice."""
    return "A" if is_visible else "B"

def format_count_answer(count: int) -> str:
    """Format count answer as multiple choice."""
    if count == 0:
        return "A"
    elif count == 1:
        return "B"
    elif count == 2:
        return "C"
    elif count == 3:
        return "D"
    else:
        return "D"

def format_distribution_answer(distributions: List[str]) -> str:
    """Format spatial distribution answer as multiple choice."""
    if not distributions or len(distributions) == 0:
        return "A"  # None - No ball is visible
    elif len(distributions) == 1:
        return "B"  # All on the same side
    elif len(distributions) == 2:
        return "C"  # Two sides
    elif len(distributions) == 3:
        return "D"  # Three sides
    else:
        return "A"  # Fallback

def generate_qa_for_sample(metadata: Dict, 
                          angle_ambiguity_threshold: float = 1e-6,
                          obstacle_angle_threshold: float = 135.0) -> Dict:
    """Generate questions and answers for a single metadata sample."""
    
    man_pos = tuple(metadata["man"]["position"])
    man_orientation = tuple(metadata["man"]["orientation"])
    woman_pos = tuple(metadata["woman"]["position"])
    woman_orientation = tuple(metadata["woman"]["orientation"])
    basketball_positions = [tuple(ball["position"]) for ball in metadata["basketballs"]]
    obstacles = metadata.get("obstacles", [])
    
    qa_data = {
        "sample_id": metadata["sample_id"],
        "metadata": metadata,
        "questions": [],
        "answers": [],
        "answer_options": get_answer_options(),
        "ambiguity_analysis": {}
    }
    
    # Question templates
    questions = [
        # Zero-order ToM (Camera perspective)
        "From the camera's perspective, is any basketball visible?",
        "From the camera's perspective, how many basketballs are there?", 
        "From the camera's perspective, what is the spatial layout of basketball(s)?",
        
        # First-order ToM (Woman's perspective)
        "From the woman's perspective, is any basketball visible?",
        "From the woman's perspective, how many basketballs can she see?",
        "From the woman's perspective, what is the spatial layout of basketball(s)?",
        
        # Second-order ToM (Man's perspective about woman)
        "From the man's perspective, does he think the woman can see any basketball?",
        "From the man's perspective, how many basketballs does he think the woman can see?",
        "From the man's perspective, what does he think is the spatial layout of basketball(s) from the woman's view?"
    ]
    
    qa_data["questions"] = questions
    
    # Ambiguity analysis (现在包含人员阻挡检测)
    woman_sees_balls_ambiguous = has_any_ambiguity(
        woman_pos, woman_orientation, basketball_positions, obstacles,
        man_pos,  # 女人看球时，男人可能阻挡
        angle_ambiguity_threshold, obstacle_angle_threshold
    )
    
    man_sees_balls_ambiguous = has_any_ambiguity(
        man_pos, man_orientation, basketball_positions, obstacles,
        woman_pos,  # 男人看球时，女人可能阻挡
        angle_ambiguity_threshold, obstacle_angle_threshold
    )
    
    man_sees_woman_ambiguous = has_any_ambiguity(
        man_pos, man_orientation, [woman_pos], obstacles,
        None,  # 男人看女人时，没有其他人阻挡
        angle_ambiguity_threshold, obstacle_angle_threshold
    )
    
    qa_data["ambiguity_analysis"] = {
        "woman_sees_balls": woman_sees_balls_ambiguous,
        "man_sees_balls": man_sees_balls_ambiguous,
        "man_sees_woman": man_sees_woman_ambiguous
    }

    man_cant_sees_balls_by_behind = count_not_visible_balls(man_pos, man_orientation, basketball_positions, obstacles, woman_pos, VisibilityState.NOT_VISIBLE_BY_BEHIND)
    man_cant_sees_balls_by_obs = count_not_visible_balls(man_pos, man_orientation, basketball_positions, obstacles, woman_pos, VisibilityState.NOT_VISIBLE_BY_OBS)
    
    woman_cant_sees_balls_by_behind = count_not_visible_balls(woman_pos, woman_orientation, basketball_positions, obstacles, man_pos, VisibilityState.NOT_VISIBLE_BY_BEHIND)
    woman_cant_sees_balls_by_obs = count_not_visible_balls(woman_pos, woman_orientation, basketball_positions, obstacles, man_pos, VisibilityState.NOT_VISIBLE_BY_OBS)
    qa_data["not_visible_analysis"] = {
        "man_cant_sees_balls_by_behind":man_cant_sees_balls_by_behind,
        "man_cant_sees_balls_by_obs":man_cant_sees_balls_by_obs,
        "woman_cant_sees_balls_by_behind":woman_cant_sees_balls_by_behind,
        "woman_cant_sees_balls_by_obs":woman_cant_sees_balls_by_obs,
    }


    # Calculate answers
    answers = []
    
    # 1-3. Camera perspective (always unambiguous)
    answers.append(format_visibility_answer(len(basketball_positions) > 0))
    answers.append(format_count_answer(len(basketball_positions)))
    
    camera_distributions = []
    for ball_pos in basketball_positions:
        if ball_pos[1] < 0:
            camera_distributions.append("left")
        elif ball_pos[1] == 0:
            camera_distributions.append("center")
        else:
            camera_distributions.append("right")
    answers.append(format_distribution_answer(list(set(camera_distributions))))
    
    # 4-6. Woman's perspective
    if woman_sees_balls_ambiguous:
        answers.extend(["EXCLUDED_AMBIGUOUS", "EXCLUDED_AMBIGUOUS", "EXCLUDED_AMBIGUOUS"])
        qa_data["exclusion_reason_4_6"] = "Woman sees balls: ambiguous visibility (including person blocking)"
    else:
        woman_visible_balls = get_visible_targets(
            woman_pos, woman_orientation, basketball_positions, obstacles,
            man_pos,  # 检查男人是否阻挡
            angle_ambiguity_threshold, obstacle_angle_threshold
        )
        
        answers.append(format_visibility_answer(len(woman_visible_balls) > 0))
        answers.append(format_count_answer(len(woman_visible_balls)))
        
        woman_distributions = []
        for ball_pos in woman_visible_balls:
            distribution = get_spatial_distribution(woman_pos, woman_orientation, ball_pos)
            if distribution:
                woman_distributions.append(distribution)
        answers.append(format_distribution_answer(list(set(woman_distributions))))
    
    # 7-9. Second-order ToM (Man's perspective about woman)
    if (man_sees_balls_ambiguous or 
        man_sees_woman_ambiguous or 
        woman_sees_balls_ambiguous):
        answers.extend(["EXCLUDED_AMBIGUOUS", "EXCLUDED_AMBIGUOUS", "EXCLUDED_AMBIGUOUS"])
        qa_data["exclusion_reason_7_9"] = {
            "man_sees_balls": man_sees_balls_ambiguous,
            "man_sees_woman": man_sees_woman_ambiguous,
            "woman_sees_balls": woman_sees_balls_ambiguous
        }
    else:
        man_sees_woman_state = can_see(
            man_pos, man_orientation, woman_pos, obstacles,
            None,  # 男人看女人时没有其他人阻挡
            angle_ambiguity_threshold, obstacle_angle_threshold
        )
        
        if man_sees_woman_state != VisibilityState.VISIBLE:
            answers.extend(["EXCLUDED_NO_SIGHT", "EXCLUDED_NO_SIGHT", "EXCLUDED_NO_SIGHT"])
            qa_data["exclusion_reason_7_9"] = "Man cannot see woman"
        else:
            man_visible_balls = get_visible_targets(
                man_pos, man_orientation, basketball_positions, obstacles,
                woman_pos,  # 检查女人是否阻挡男人看球
                angle_ambiguity_threshold, obstacle_angle_threshold
            )
            
            man_thinks_woman_sees = []
            for ball_pos in man_visible_balls:
                woman_sees_ball_state = can_see(
                    woman_pos, woman_orientation, ball_pos, obstacles,
                    man_pos,  # 男人考虑自己是否阻挡女人看球
                    angle_ambiguity_threshold, obstacle_angle_threshold
                )
                if woman_sees_ball_state == VisibilityState.VISIBLE:
                    man_thinks_woman_sees.append(ball_pos)
            
            answers.append(format_visibility_answer(len(man_thinks_woman_sees) > 0))
            answers.append(format_count_answer(len(man_thinks_woman_sees)))
            
            man_thinks_distributions = []
            for ball_pos in man_thinks_woman_sees:
                distribution = get_spatial_distribution(woman_pos, woman_orientation, ball_pos)
                if distribution:
                    man_thinks_distributions.append(distribution)
            answers.append(format_distribution_answer(list(set(man_thinks_distributions))))
    
    qa_data["answers"] = answers
    return qa_data

def generate_all_qa(metadata_list: List[Dict], 
                   angle_ambiguity_threshold: float = 1e-6,
                   obstacle_angle_threshold: float = 135.0) -> List[Dict]:
    """Generate QA for all metadata samples."""
    qa_list = []
    excluded_count = 0
    
    for i, metadata in enumerate(metadata_list):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(metadata_list)}")
        
        qa_data = generate_qa_for_sample(
            metadata, angle_ambiguity_threshold, obstacle_angle_threshold
        )
        
        if any("EXCLUDED" in str(answer) for answer in qa_data["answers"]):
            excluded_count += 1
        
        qa_list.append(qa_data)
    
    print(f"Generated QA for {len(qa_list)} samples")
    print(f"Excluded {excluded_count} samples due to ambiguity or visibility issues")
    print(f"Usable samples: {len(qa_list) - excluded_count}")
    
    return qa_list

def save_qa_data(qa_list: List[Dict], filename: str):
    """Save QA data to JSONL file."""
    try:
        with open(filename, 'w') as f:
            for qa_data in qa_list:
                f.write(json.dumps(qa_data) + '\n')
        print(f"Successfully saved {len(qa_list)} QA samples to {filename}")
    except Exception as e:
        print(f"Error saving to file '{filename}': {e}")
        raise

def get_answer_options():
    """Get detailed answer options for all question types."""
    return {
        "visibility": {
            "A": {
                "text": "Yes",
                "description": "At least one basketball is visible to the observer"
            },
            "B": {
                "text": "No", 
                "description": "No basketball is visible to the observer"
            }
        },
        "count": {
            "A": {
                "text": "0",
                "description": "No basketballs are visible to the observer"
            },
            "B": {
                "text": "1",
                "description": "Exactly one basketball is visible to the observer"
            },
            "C": {
                "text": "2", 
                "description": "Exactly two basketballs are visible to the observer"
            },
            "D": {
                "text": "3",
                "description": "Exactly three basketballs are visible to the observer"
            }
        },
        "layout": {
            "A": {
                "text": "None",
                "description": "No basketball is visible to the observer in any of the three directions relative to the observer: left-front, center-front, or right-front.",
                "use_case": "Use when the observer does not see any basketballs in any of the three directions relative to the observer."
            },
            "B": {
                "text": "Laid out in one direction",
                "description": "At least one basketball is visible, and all visible balls are laid out in the same direction relative to the observer: left-front, center-front, or right-front.",
                "use_case": "Use when the observer sees one or more basketballs, all laid out in the same direction relative to the observer (e.g., all on the left-front)."
            },
            "C": {
                "text": "Laid out in two directions",
                "description": "Visible basketballs are laid out across exactly two of the three directions relative to the observer: left-front, center-front, or right-front.",
                "use_case": "Use when the observer sees basketballs laid out in two distinct directions relative to the observer (e.g., left-front and right-front, or center-front and right-front)."
            },
            "D": {
                "text": "Laid out in three directions",
                "description": "At least one basketball is visible in each of the three directions relative to the observer: left-front, center-front, and right-front.",
                "use_case": "Use when the observer sees basketballs laid out across all three directions relative to the observer."
            }
        }
    }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Q&A data with modified visibility system including person blocking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m vsrtom.dataset.qa_generation
  python -m vsrtom.dataset.qa_generation --input metadata.jsonl --output qa_data.jsonl
  python -m vsrtom.dataset.qa_generation --obstacle-angle-threshold 120.0
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='metadata.jsonl',
        help='Input metadata JSONL file (default: metadata.jsonl)'
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str,
        default='qa_data.jsonl',
        help='Output QA JSONL file (default: qa_data.jsonl)'
    )
    
    parser.add_argument(
        '--angle-ambiguity-threshold',
        type=float,
        default=1e-6,
        help='Threshold for viewing angle ambiguity detection (default: 1e-6)'
    )
    
    parser.add_argument(
        '--obstacle-angle-threshold',
        type=float,
        default=135.0,
        help='Angle threshold for obstacle blocking determination (degrees, default: 135.0)'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    print(f"Loading metadata from: {args.input}")
    metadata_list = load_metadata(args.input)
    
    print(f"Generating QA data with modified parameters (including person blocking):")
    print(f"  - Angle ambiguity threshold: {args.angle_ambiguity_threshold}")
    print(f"  - Obstacle angle threshold: {args.obstacle_angle_threshold}°")
    print(f"  - Person blocking detection: Enabled")
    
    qa_list = generate_all_qa(
        metadata_list, 
        args.angle_ambiguity_threshold,
        args.obstacle_angle_threshold
    )
    
    print(f"Saving QA data to: {args.output}")
    save_qa_data(qa_list, args.output)
    
    print(f"✓ Successfully generated QA for {len(qa_list)} samples")
    print(f"✓ Input: {args.input}")
    print(f"✓ Output: {args.output}")
    print(f"✓ New features: Person blocking detection with perfect grid coordinate conversion")
