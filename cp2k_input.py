from get_defect_data import create_sampled_datasets
from cp2k_input_writer import write_inputs_from_json

# 1. 先对已生成的全量 JSON 做分层采样（约 1000 条 charge-state）
defect_json, charge_json = create_sampled_datasets(
    defect_json_path="data/defect_structures.json",
    charge_json_path="data/charge_states.json",
    target_charge_count=1000,          # 期望最终 .inp 数量
    strategy="stratified",             # 最保 (material, defect_type) 分布
    # random_seed=42,                  # 可复现
)

# 2. 再用原有入口生成输入文件
input_files = write_inputs_from_json(
    defect_json_path=defect_json,
    charge_json_path=charge_json,
    output_dir="cp2k_inputs",
    max_files=None,                    # 此时已是采样后的规模，无需再截断
)