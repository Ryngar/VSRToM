import bpy
import os
import math
import json
import glob
import sys
from pathlib import Path

ASSETS_DIR = None
RENDER_DEVICE = "CPU"

def asset_path(relative):
    return str(ASSETS_DIR / relative)

def load_jsonl_data(filepath, max_lines=None):
    """读取JSONL文件数据"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"第{i+1}行JSON解析错误: {e}")
                continue
    return data

def get_sample_ids_from_folder(folder_path):
    """遍历文件夹内的jsonl文件，获取所有sample_id"""
    print(f"📁 遍历文件夹: {folder_path}")
    
    # 查找文件夹内所有的jsonl文件
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    
    if not jsonl_files:
        print(f"⚠️  文件夹 {folder_path} 中未找到任何 .jsonl 文件")
        return set()
    
    print(f"📄 找到 {len(jsonl_files)} 个jsonl文件:")
    for file in jsonl_files:
        print(f"  - {os.path.basename(file)}")
    
    all_sample_ids = set()
    
    # 遍历每个jsonl文件
    for jsonl_file in jsonl_files:
        try:
            print(f"\n🔍 处理文件: {os.path.basename(jsonl_file)}")
            data = load_jsonl_data(jsonl_file)
            
            file_sample_ids = set()
            for item in data:
                sample_id = item.get('sample_id')
                if sample_id is not None:
                    file_sample_ids.add(sample_id)
                    all_sample_ids.add(sample_id)
            
            print(f"  ✅ 提取到 {len(file_sample_ids)} 个sample_id")
            
        except Exception as e:
            print(f"  ❌ 处理文件 {jsonl_file} 时出错: {e}")
            continue
    
    print(f"\n📊 总计提取到 {len(all_sample_ids)} 个唯一的sample_id")
    return all_sample_ids

def load_metadata_by_sample_ids(metadata_file, target_sample_ids):
    """根据sample_id列表从metadata文件中加载对应的数据"""
    print(f"📖 从 {metadata_file} 加载metadata...")
    
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata文件不存在: {metadata_file}")
        return []
    
    all_metadata = load_jsonl_data(metadata_file)
    print(f"📋 Metadata文件包含 {len(all_metadata)} 条记录")
    
    # 筛选出目标sample_id对应的metadata
    filtered_metadata = []
    found_sample_ids = set()
    
    for metadata in all_metadata:
        sample_id = metadata.get('sample_id')
        if sample_id in target_sample_ids:
            filtered_metadata.append(metadata)
            found_sample_ids.add(sample_id)
    
    # 统计信息
    missing_sample_ids = target_sample_ids - found_sample_ids
    
    print(f"✅ 成功匹配 {len(filtered_metadata)} 条metadata")
    print(f"📊 匹配的sample_id: {len(found_sample_ids)}")
    
    if missing_sample_ids:
        print(f"⚠️  未找到metadata的sample_id数量: {len(missing_sample_ids)}")
        if len(missing_sample_ids) <= 10:  # 如果数量不多，显示具体的ID
            print(f"   缺失的sample_id: {sorted(list(missing_sample_ids))}")
        else:
            print(f"   缺失的sample_id示例: {sorted(list(missing_sample_ids))[:10]}...")
    
    return filtered_metadata

def orientation_to_rotation(orientation):
    """将orientation向量转换为Euler旋转角度"""
    x, y, z = orientation
    angle = math.atan2(y, x)
    return (0, 0, angle)

def setup_fresh_scene():
    """为每个sample设置全新的基础场景（不包含篮球和障碍物）"""
    print("🏗️  重新加载基础场景...")
    
    # 重新加载基础场景文件，自动清空所有之前的数据
    bpy.ops.wm.open_mainfile(filepath=asset_path('assets/base_scene_centered.blend'))

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.cycles.device = RENDER_DEVICE
    if RENDER_DEVICE == 'GPU':
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == 'CUDA'
        if not any(device.type == 'CUDA' for device in prefs.devices):
            raise RuntimeError("CUDA device unavailable; use --device CPU")

    # 加载女性角色
    print("👩 加载女性角色...")
    blend_path = asset_path('Sophia.blend')
    bpy.ops.wm.append(directory=os.path.join(blend_path, "Collection") + os.sep, filename="Collection")
    obj = bpy.data.objects["Sophia"]
    obj.location = (-3, 3, 0)

    # 加载男性角色
    print("👨 加载男性角色...")
    blend_path = asset_path('Man.blend')
    bpy.ops.wm.append(directory=os.path.join(blend_path, "Collection") + os.sep, filename="Collection")
    obj = bpy.data.objects["Man"]
    obj.location = (-3, -3, 0)
    obj.rotation_euler = (0, 0, math.radians(90))
    
    print("✅ 基础场景设置完成")

def append_basketballs(num_basketballs):
    """根据需要的数量append篮球Collection"""
    if num_basketballs <= 0:
        print("🏀 无需添加篮球")
        raise SystemExit(1)
    
    print(f"🏀 添加 {num_basketballs} 个篮球...")
    
    blend_path = asset_path('Basketball.blend')
    
    for i in range(num_basketballs):
        # 每次都重新append Collection，确保对象独立
        bpy.ops.wm.append(directory=os.path.join(blend_path, "Collection") + os.sep, filename="Collection")
        
        # 重命名篮球对象
        basketball_obj = bpy.data.objects["Basketball"]
        basketball_obj.name = f"Basketball{i+1}"
        basketball_obj.location = (0, 0, 0)  # 默认位置，后续会设置具体位置
        
        print(f"  ✅ 添加篮球 {i+1}/{num_basketballs}")
    
    print(f"🏀 {num_basketballs} 个篮球添加完成")

def append_obstacles(num_obstacles):
    """根据需要的数量append障碍物Collection"""
    if num_obstacles <= 0:
        print("🚧 无需添加障碍物")
        raise SystemExit(1)
    
    print(f"🚧 添加 {num_obstacles} 个障碍物...")
    
    # 如果有障碍物模型，在这里添加相应的逻辑
    # 示例（需要有实际的障碍物模型文件）:
    blend_path = asset_path('Obstacle.blend')
    for i in range(num_obstacles):
        bpy.ops.wm.append(directory=os.path.join(blend_path, "Collection") + os.sep, filename="Collection")
        obstacle_obj = bpy.data.objects["Obstacle"]
        obstacle_obj.name = f"Obstacle{i+1}"
        obstacle_obj.location = (0, 0, 0)
        print(f"  ✅ 添加障碍物 {i+1}/{num_obstacles}")
    
    # print(f"⚠️  障碍物添加逻辑尚未实现（需要障碍物模型文件）")

def setup_scene_objects(metadata):
    """根据metadata设置场景中所有对象的位置和朝向"""
    print("📍 设置对象位置和朝向...")
    
    # 设置女性角色
    woman_data = metadata['woman']
    woman_obj = bpy.data.objects["Sophia"]
    woman_obj.location = woman_data['position']
    woman_obj.rotation_euler = orientation_to_rotation(woman_data['orientation'])
    print(f"  👩 女性角色: 位置{woman_data['position']}")
    
    # 设置男性角色
    man_data = metadata['man']
    man_obj = bpy.data.objects["Man"]
    man_obj.location = man_data['position']
    man_obj.rotation_euler = orientation_to_rotation(man_data['orientation'])
    print(f"  👨 男性角色: 位置{man_data['position']}")

    # 设置篮球位置
    basketballs_data = metadata.get('basketballs', [])
    for i, basketball_data in enumerate(basketballs_data):
        basketball_name = f"Basketball{i+1}"
        if basketball_name in bpy.data.objects:
            obj = bpy.data.objects[basketball_name]
            obj.location = basketball_data['position']
            print(f"  🏀 {basketball_name}: 位置{basketball_data['position']}")
        else:
            print(f"  ⚠️  未找到篮球对象: {basketball_name}")
    
    # 设置障碍物位置（如果有的话）
    obstacles_data = metadata.get('obstacles', [])
    for i, obstacle_data in enumerate(obstacles_data):
        obstacle_name = f"Obstacle{i+1}"
        if obstacle_name in bpy.data.objects:
            obj = bpy.data.objects[obstacle_name]
            obj.location = obstacle_data['position']
            print(f"  🚧 {obstacle_name}: 位置{obstacle_data['position']}")
        # 如果没有障碍物对象，这里不打印警告，因为可能没有实现障碍物
    
    print("✅ 对象位置设置完成")

def render_scene_from_metadata(metadata, output_dir="output"):
    """根据metadata渲染场景"""
    sample_id = metadata['sample_id']
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 获取配置信息
        config = metadata.get('config', {})
        num_basketballs = config.get('num_basketballs', len(metadata.get('basketballs', [])))
        num_obstacles = config.get('num_obstacles', len(metadata.get('obstacles', [])))
        
        print(f"🎬 配置: {num_basketballs} 个篮球, {num_obstacles} 个障碍物")
        
        # 1. 设置全新的基础场景（只包含角色）
        setup_fresh_scene()
        
        # 2. 根据需要append篮球Collection
        append_basketballs(num_basketballs)
        
        # 3. 根据需要append障碍物Collection
        append_obstacles(num_obstacles)
        
        # 4. 设置所有对象的位置和朝向
        setup_scene_objects(metadata)
        
        # 5. 设置输出文件路径
        output_path = os.path.join(output_dir, f"{sample_id}.png")
        bpy.context.scene.render.filepath = output_path
        
        # 6. 渲染
        print(f"🎨 开始渲染 Sample ID: {sample_id}...")
        bpy.ops.render.render(write_still=True)
        print(f"✅ 渲染完成 - Sample ID: {sample_id}, 保存至: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 渲染失败 - Sample ID: {sample_id}, 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    import argparse
    global ASSETS_DIR, RENDER_DEVICE
    parser = argparse.ArgumentParser(description="Render selected VSRToM scenes in Blender")
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--selected-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    ASSETS_DIR = args.assets_dir.resolve()
    RENDER_DEVICE = args.device
    for filename in ["assets/base_scene_centered.blend", "Sophia.blend", "Man.blend",
                     "Basketball.blend", "Obstacle.blend"]:
        if not (ASSETS_DIR / filename).is_file():
            parser.error(f"Missing required asset: {filename}")
    jsonl_folder = args.selected_dir
    metadata_jsonl_file = args.metadata
    output_directory = str(Path(args.output_dir).resolve())

    print("🚀 开始批量渲染流程...")
    print("="*60)
    
    # 第一步：遍历文件夹获取所有sample_id
    print("第一步：提取sample_id")
    target_sample_ids = get_sample_ids_from_folder(jsonl_folder)
    
    if not target_sample_ids:
        print("❌ 未找到任何sample_id，程序退出")
        raise SystemExit(1)
    
    print("="*60)
    
    # 第二步：根据sample_id从metadata中加载对应数据
    print("第二步：加载对应的metadata")
    metadata_list = load_metadata_by_sample_ids(metadata_jsonl_file, target_sample_ids)
    
    if not metadata_list:
        print("❌ 未找到任何匹配的metadata，程序退出")
        raise SystemExit(1)
    
    print("="*60)
    
    # 第三步：批量渲染
    print("第三步：开始批量渲染")
    success_count = 0
    total_count = len(metadata_list)
    
    for i, metadata in enumerate(metadata_list):
        sample_id = metadata.get('sample_id')
        if sample_id is None:
            print(f"⚠️  第{i+1}条数据缺少sample_id，跳过")
            continue
            
        print(f"\n{'='*60}")
        print(f"🎬 处理 {i+1}/{total_count} - Sample ID: {sample_id}")
        print(f"{'='*60}")
        
        # 每个sample都从头开始渲染（会自动清理之前的数据）
        if render_scene_from_metadata(metadata, output_directory):
            success_count += 1
            print(f"✅ Sample {sample_id} 处理成功")
        else:
            print(f"❌ Sample {sample_id} 渲染失败")
    
    print(f"\n{'='*60}")
    print(f"🎉 批量渲染完成！")
    print(f"📊 总体统计:")
    print(f"   - 目标sample_id数量: {len(target_sample_ids)}")
    print(f"   - 匹配到metadata数量: {len(metadata_list)}")
    print(f"   - 渲染成功数量: {success_count}")
    print(f"   - 渲染失败数量: {total_count - success_count}")
    if success_count != total_count:
        raise SystemExit("Some renders failed")
    print(f"   - 成功率: {success_count/total_count*100:.1f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()