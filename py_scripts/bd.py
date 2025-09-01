import os
import json

def build_tree(path):
    """递归构建目录树，最底层json直接解析为字典"""
    tree = {}
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            tree[entry] = build_tree(full_path)
        elif entry.endswith(".json"):
            try:
                if "result" in entry:
                    continue  # 跳过包含 "result" 的文件
                with open(full_path, "r", encoding="utf-8") as f:
                    tree[entry] = json.load(f)
                    if "vector" in tree[entry]:
                        tree[entry]['vector']= "omitted for brevity"
                    
            except Exception as e:
                print(f"⚠️ 无法读取 {full_path}: {e}")
                tree[entry] = None
    return tree


def save_tree(root_path, output_file):
    tree = build_tree(root_path)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=4)
    print(f"✅ 已保存到 {output_file}")


if __name__ == "__main__":
    # 修改为你的目录路径和输出文件路径
    root_dir = "./evaluation"
    output_file = "output.json"
    save_tree(root_dir, output_file)
