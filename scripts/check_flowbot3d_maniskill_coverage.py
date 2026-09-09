import json
import xml.etree.ElementTree as ET
from pathlib import Path

from mani_skill.utils.misc import get_raw_yaml


HOME = Path.home()

FLOWBOT_ROOT = HOME / "robot_baselines/repos/flowbot3d"
PM_ROOT = HOME / "robot_baselines/data/partnet-mobility"
VAL_SPLIT = HOME / "robot_baselines/splits/val_split.json"

DOOR_YAML = (
    FLOWBOT_ROOT
    / "third_party/ManiSkill/mani_skill/assets/config_files/cabinet_models_door.yml"
)

DRAWER_YAML = (
    FLOWBOT_ROOT
    / "third_party/ManiSkill/mani_skill/assets/config_files/cabinet_models_drawer.yml"
)

door_data = get_raw_yaml(DOOR_YAML)
drawer_data = get_raw_yaml(DRAWER_YAML)

door_ids = {str(x) for x in door_data.keys()}
drawer_ids = {str(x) for x in drawer_data.keys()}

with open(VAL_SPLIT) as f:
    split = json.load(f)

results = []

for shape_id in split["ids"]:

    obj_info = split["objects"][shape_id]

    urdf_path = PM_ROOT / shape_id / "mobility.urdf"

    root = ET.parse(urdf_path).getroot()

    # 建立：
    # 子部件名称 -> 控制它的关节
    link_to_joint = {}

    for joint in root.findall("joint"):

        child = joint.find("child")

        if child is None:
            continue

        child_link = child.attrib["link"]

        link_to_joint[child_link] = {
            "joint_name": joint.attrib["name"],
            "joint_type": joint.attrib["type"],
        }

    for link_info in obj_info["selected_links"]:

        link_name = link_info["link_name"]

        joint = link_to_joint.get(link_name)

        if joint is None:
            joint_type = "UNKNOWN"
            joint_name = "UNKNOWN"

        else:
            joint_type = joint["joint_type"]
            joint_name = joint["joint_name"]

        # 旋转关节应该进入“门”环境
        if joint_type in {"revolute", "continuous"}:

            expected_env = "旋转门"

            native = (
                shape_id in door_ids
            )

        # 平移关节应该进入“抽屉”环境
        elif joint_type == "prismatic":

            expected_env = "抽屉"

            native = (
                shape_id in drawer_ids
            )

        else:

            expected_env = "未知"

            native = False

        results.append(
            {
                "shape_id": shape_id,
                "category": obj_info["category"],
                "link_name": link_name,
                "joint_name": joint_name,
                "joint_type": joint_type,
                "expected_env": expected_env,
                "native": native,
            }
        )


print("=" * 105)
print("FlowBot3D 原始机器人环境对验证集的覆盖情况")
print("=" * 105)

native_count = 0
adapt_count = 0

for r in results:

    if r["native"]:
        status = "原生支持"
        native_count += 1
    else:
        status = "需要适配"
        adapt_count += 1

    print(
        f"{r['shape_id']:>8s}  "
        f"{r['link_name']:<8s}  "
        f"{r['category']:<20s}  "
        f"{r['joint_type']:<12s}  "
        f"{r['expected_env']:<8s}  "
        f"{status}"
    )


print()
print("=" * 105)
print("统计结果")
print("=" * 105)

print("验证目标总数:", len(results))
print("原生环境可覆盖:", native_count)
print("需要额外适配:", adapt_count)

print("=" * 105)


output = (
    HOME
    / "robot_baselines/results/flowbot3d/maniskill_coverage.json"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(output, "w") as f:

    json.dump(
        results,
        f,
        indent=2,
    )

print()
print("报告已保存:")
print(output)
