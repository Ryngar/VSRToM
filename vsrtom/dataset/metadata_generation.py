#!/usr/bin/env python3
"""
Stage 1: Enhanced Metadata Generation with Sampling
Generates specified numbers of random configurations of man, woman, basketballs, and obstacles
with their positions and orientations.

Ensures no duplicate samples within each configuration type.
Extensible design for future support of multiple obstacles and more positions.
"""

import json
import random
from typing import List, Tuple, Dict, Set
import hashlib

def generate_metadata(args):
    """Generate specified numbers of unique random metadata configurations."""
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    
    # 25 specific positions - easily extensible for more positions
    positions = [
        (-5, -4, 0), (-5, -2, 0), (-5, 0, 0), (-5, 2, 0), (-5, 4, 0),
        (-3, -4, 0), (-3, -2, 0), (-3, 0, 0), (-3, 2, 0), (-3, 4, 0),
        (0, -4, 0), (0, -2, 0), (0, 0, 0), (0, 2, 0), (0, 4, 0),
        (1.5, -4, 0), (1.5, -2, 0), (1.5, 0, 0), (1.5, 2, 0), (1.5, 4, 0),
        (3, -4, 0), (3, -2, 0), (3, 0, 0), (3, 2, 0), (3, 4, 0)
    ]
    
    # 4 orientations (facing directions) - easily extensible
    orientations = [
        [1, 0, 0],   # facing positive x
        [-1, 0, 0],  # facing negative x
        [0, 1, 0],   # facing positive y
        [0, -1, 0]   # facing negative y
    ]
    
    metadata_list = []
    sample_id = 0
    
    # Configuration mapping - structured for easy extension
    configurations = [
        (0, 1, args.num_samples0_with_obstacles, "0 basketballs, 1 obstacle"),
        (1, 1, args.num_samples1_with_obstacles, "1 basketball, 1 obstacle"),
        (2, 1, args.num_samples2_with_obstacles, "2 basketballs, 1 obstacle"),
        (3, 1, args.num_samples3_with_obstacles, "3 basketballs, 1 obstacle"),
        (0, 0, args.num_samples0_without_obstacles, "0 basketballs, 0 obstacles"),
        (1, 0, args.num_samples1_without_obstacles, "1 basketball, 0 obstacles"),
        (2, 0, args.num_samples2_without_obstacles, "2 basketballs, 0 obstacles"),
        (3, 0, args.num_samples3_without_obstacles, "3 basketballs, 0 obstacles"),
    ]
    
    for num_basketballs, num_obstacles, num_samples, description in configurations:
        if num_samples <= 0:
            continue
            
        print(f"Generating {num_samples:,} unique samples for: {description}")
        
        # Track unique configurations for this specific scenario
        seen_configurations: Set[str] = set()
        generated_count = 0
        total_attempts = 0
        
        # Calculate maximum possible unique configurations for early termination
        max_possible = calculate_max_possible_configurations(
            positions, orientations, num_basketballs, num_obstacles
        )
        
        if num_samples > max_possible:
            print(f"Warning: Requested {num_samples:,} samples but only {max_possible:,} unique configurations possible.")
            print(f"Will generate maximum possible: {max_possible:,}")
            num_samples = max_possible
        
        while generated_count < num_samples and total_attempts < num_samples * args.max_attempts_per_sample:
            total_attempts += 1
            
            # Generate obstacle configuration
            if num_obstacles > 0:
                # Currently: 1 obstacle at (0,0,0) - easily extensible for more obstacles
                obstacle_positions = [(0, 0, 0)]
                obstacles_list = [{"position": [0, 0, 0]}]
            else:
                obstacle_positions = []
                obstacles_list = []
            
            # Calculate available positions (exclude obstacle positions)
            available_positions = [pos for pos in positions if pos not in obstacle_positions]
            
            # Validate we have enough positions
            min_required_positions = 2 + num_basketballs  # man + woman + basketballs
            if len(available_positions) < min_required_positions:
                print(f"Error: Not enough available positions for configuration. Breaking.")
                break
            
            # Randomly select positions for man and woman
            try:
                selected_positions = random.sample(available_positions, 2)
                man_pos, woman_pos = selected_positions
            except ValueError:
                continue
            
            # Calculate remaining positions for basketballs
            remaining_positions = [pos for pos in available_positions 
                                 if pos not in [man_pos, woman_pos]]
            
            # Randomly select positions for basketballs
            basketball_positions = []
            if num_basketballs > 0:
                if len(remaining_positions) >= num_basketballs:
                    basketball_positions = random.sample(remaining_positions, num_basketballs)
                else:
                    continue  # Not enough positions, try again
            
            # Randomly select orientations
            man_orientation = random.choice(orientations)
            woman_orientation = random.choice(orientations)
            
            # Generate unique configuration identifier
            config_hash = generate_configuration_hash(
                man_pos, man_orientation,
                woman_pos, woman_orientation,
                basketball_positions,
                obstacles_list
            )
            
            # Check if this configuration is unique
            if config_hash in seen_configurations:
                continue  # Duplicate, try again
            
            # Add to seen configurations
            seen_configurations.add(config_hash)
            
            # Create metadata with extensible structure
            metadata = {
                "sample_id": sample_id,
                "man": {
                    "position": list(man_pos),
                    "orientation": man_orientation
                },
                "woman": {
                    "position": list(woman_pos),
                    "orientation": woman_orientation
                },
                "basketballs": [
                    {"position": list(pos)} for pos in basketball_positions
                ],
                "obstacles": obstacles_list,
                "config": {
                    "num_basketballs": num_basketballs,
                    "num_obstacles": num_obstacles
                }
            }
            metadata_list.append(metadata)
            sample_id += 1
            generated_count += 1
            
            # Progress update for large datasets
            if generated_count % max(1, num_samples // 10) == 0:
                print(f"  Progress: {generated_count:,}/{num_samples:,} ({generated_count/num_samples*100:.1f}%) - Attempts: {total_attempts:,}")
        
        if generated_count < num_samples:
            print(f"Warning: Only generated {generated_count:,} unique samples out of {num_samples:,} requested after {total_attempts:,} attempts")
        else:
            print(f"Successfully generated {generated_count:,} unique samples in {total_attempts:,} attempts")
    
    return metadata_list

def generate_configuration_hash(man_pos, man_orientation, woman_pos, woman_orientation, basketball_positions, obstacles_list) -> str:
    """Generate a unique hash for a configuration to detect duplicates."""
    
    # Create a tuple representing the entire configuration
    config_tuple = (
        # Man configuration
        tuple(man_pos),
        tuple(man_orientation),
        # Woman configuration  
        tuple(woman_pos),
        tuple(woman_orientation),
        # Basketball positions (sorted for consistency)
        tuple(sorted([tuple(pos) for pos in basketball_positions])),
        # Obstacle positions (sorted for consistency)
        tuple(sorted([tuple(obs["position"]) for obs in obstacles_list]))
    )
    
    # Generate SHA256 hash
    config_str = str(config_tuple)
    return hashlib.sha256(config_str.encode()).hexdigest()

def calculate_max_possible_configurations(positions, orientations, num_basketballs, num_obstacles):
    """Calculate theoretical maximum number of unique configurations."""
    
    if num_obstacles > 0:
        # Currently: 1 obstacle at (0,0,0)
        obstacle_positions = [(0, 0, 0)]
        available_positions = [pos for pos in positions if pos not in obstacle_positions]
    else:
        available_positions = positions
    
    n_positions = len(available_positions)
    n_orientations = len(orientations)
    
    # Check if we have enough positions
    min_required = 2 + num_basketballs
    if n_positions < min_required:
        return 0
    
    # Calculate combinations
    # Man: n_positions choices
    # Woman: (n_positions - 1) choices  
    # Basketballs: C(n_positions - 2, num_basketballs) choices
    # Orientations: n_orientations^2 for man and woman
    
    import math
    
    man_woman_combinations = n_positions * (n_positions - 1)
    
    remaining_positions = n_positions - 2
    if num_basketballs == 0:
        basketball_combinations = 1
    else:
        if remaining_positions >= num_basketballs:
            basketball_combinations = math.comb(remaining_positions, num_basketballs)
        else:
            basketball_combinations = 0
    
    orientation_combinations = n_orientations ** 2
    
    max_configs = man_woman_combinations * basketball_combinations * orientation_combinations
    return max_configs

def save_metadata(metadata_list: List[Dict], filename: str = "metadata.jsonl"):
    """Save metadata to JSONL file."""
    with open(filename, 'w') as f:
        for metadata in metadata_list:
            f.write(json.dumps(metadata) + '\n')
    print(f"Saved {len(metadata_list)} metadata samples to {filename}")

def print_summary(args):
    """Print summary of configurations that will be generated."""
    
    total_with_obstacles = (args.num_samples0_with_obstacles + args.num_samples1_with_obstacles + 
                           args.num_samples2_with_obstacles + args.num_samples3_with_obstacles)
    total_without_obstacles = (args.num_samples0_without_obstacles + args.num_samples1_without_obstacles + 
                              args.num_samples2_without_obstacles + args.num_samples3_without_obstacles)
    total_samples = total_with_obstacles + total_without_obstacles
    
    print("Configuration Summary:")
    print("=" * 60)
    print("WITH OBSTACLES (1 obstacle):")
    print(f"  0 basketballs: {args.num_samples0_with_obstacles:,}")
    print(f"  1 basketball:  {args.num_samples1_with_obstacles:,}")
    print(f"  2 basketballs: {args.num_samples2_with_obstacles:,}")
    print(f"  3 basketballs: {args.num_samples3_with_obstacles:,}")
    print(f"  Subtotal:      {total_with_obstacles:,}")
    print()
    print("WITHOUT OBSTACLES (0 obstacles):")
    print(f"  0 basketballs: {args.num_samples0_without_obstacles:,}")
    print(f"  1 basketball:  {args.num_samples1_without_obstacles:,}")
    print(f"  2 basketballs: {args.num_samples2_without_obstacles:,}")
    print(f"  3 basketballs: {args.num_samples3_without_obstacles:,}")
    print(f"  Subtotal:      {total_without_obstacles:,}")
    print()
    print(f"TOTAL SAMPLES: {total_samples:,}")
    print("=" * 60)
    print("✅ Guarantees: Each sample is unique within its configuration type")
    print("⚠️  Note: If requested samples exceed possible unique combinations,")
    print("   the system will generate maximum possible unique samples")

def parse_arguments():
    """Parse command line arguments with default values."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate metadata configurations for man, woman, basketballs, and obstacles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Sample count parameters for each configuration type
    parser.add_argument("--num_samples0_with_obstacles", type=int, default=500,
                       help="Number of samples: 0 basketballs WITH obstacles")
    parser.add_argument("--num_samples1_with_obstacles", type=int, default=1000,
                       help="Number of samples: 1 basketball WITH obstacles")
    parser.add_argument("--num_samples2_with_obstacles", type=int, default=1500,
                       help="Number of samples: 2 basketballs WITH obstacles")
    parser.add_argument("--num_samples3_with_obstacles", type=int, default=1000,
                       help="Number of samples: 3 basketballs WITH obstacles")
    
    parser.add_argument("--num_samples0_without_obstacles", type=int, default=500,
                       help="Number of samples: 0 basketballs WITHOUT obstacles")
    parser.add_argument("--num_samples1_without_obstacles", type=int, default=1000,
                       help="Number of samples: 1 basketball WITHOUT obstacles")
    parser.add_argument("--num_samples2_without_obstacles", type=int, default=1500,
                       help="Number of samples: 2 basketballs WITHOUT obstacles")
    parser.add_argument("--num_samples3_without_obstacles", type=int, default=1000,
                       help="Number of samples: 3 basketballs WITHOUT obstacles")
    
    # Generation control parameters
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducible generation")
    parser.add_argument("--max_attempts_per_sample", type=int, default=1000,
                       help="Maximum attempts to generate each unique sample")
    
    # Output file parameter
    parser.add_argument("--output_file", type=str, default="metadata.jsonl",
                       help="Output filename for generated metadata")
    
    # Quick preset options
    parser.add_argument("--preset", type=str, choices=["small", "medium", "large", "test"], default=None,
                       help="Use predefined sample size presets")
    
    return parser.parse_args()

def apply_preset(args):
    """Apply predefined sample size presets."""
    presets = {
        "test": {
            "num_samples0_with_obstacles": 10,
            "num_samples1_with_obstacles": 20,
            "num_samples2_with_obstacles": 30,
            "num_samples3_with_obstacles": 20,
            "num_samples0_without_obstacles": 10,
            "num_samples1_without_obstacles": 20,
            "num_samples2_without_obstacles": 30,
            "num_samples3_without_obstacles": 20,
        },
        "small": {
            "num_samples0_with_obstacles": 100,
            "num_samples1_with_obstacles": 200,
            "num_samples2_with_obstacles": 300,
            "num_samples3_with_obstacles": 200,
            "num_samples0_without_obstacles": 100,
            "num_samples1_without_obstacles": 200,
            "num_samples2_without_obstacles": 300,
            "num_samples3_without_obstacles": 200,
        },
        "medium": {
            "num_samples0_with_obstacles": 500,
            "num_samples1_with_obstacles": 1000,
            "num_samples2_with_obstacles": 1500,
            "num_samples3_with_obstacles": 1000,
            "num_samples0_without_obstacles": 500,
            "num_samples1_without_obstacles": 1000,
            "num_samples2_without_obstacles": 1500,
            "num_samples3_without_obstacles": 1000,
        },
        "large": {
            "num_samples0_with_obstacles": 5000,
            "num_samples1_with_obstacles": 40000,
            "num_samples2_with_obstacles": 60000,
            "num_samples3_with_obstacles": 45000,
            "num_samples0_without_obstacles": 5000,
            "num_samples1_without_obstacles": 40000,
            "num_samples2_without_obstacles": 60000,
            "num_samples3_without_obstacles": 45000,
        }
    }
    
    if args.preset and args.preset in presets:
        print(f"Applying preset: {args.preset}")
        preset_config = presets[args.preset]
        for key, value in preset_config.items():
            setattr(args, key, value)
    
    return args

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Apply preset if specified
    args = apply_preset(args)
    
    print("Metadata Generation Settings:")
    print("=" * 40)
    print(f"  Random seed: {args.random_seed}")
    print(f"  Max attempts per sample: {args.max_attempts_per_sample:,}")
    print(f"  Output file: {args.output_file}")
    print("=" * 40)
    
    print_summary(args)
    
    print("\nGenerating metadata with uniqueness guarantee...")
    metadata = generate_metadata(args)
    save_metadata(metadata, args.output_file)
    print(f"\nSuccessfully generated {len(metadata):,} unique samples")
    print(f"Output saved to: {args.output_file}")
