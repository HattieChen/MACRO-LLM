import os
from datetime import datetime
import json
import numpy as np
import re
import ast
from pathlib import Path
from LLMmodules.TaskConfiguration import pandemic
from LLMmodules.model_config import apply_run_name_override

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_LLM_RESPONSE_ATTEMPTS = 10


def raise_if_llm_retry_exhausted(attempt, operation, detail):
    if attempt >= MAX_LLM_RESPONSE_ATTEMPTS:
        raise RuntimeError(
            f"{operation} failed after {MAX_LLM_RESPONSE_ATTEMPTS} attempts: {detail}"
        )

def sigmoid(x, alpha=1.0):
    return 1 / (1 + np.exp(-alpha * x))

def initialize_conversation_context(exp, epi, n_agent = 1, config = None, all_agents = None):
    """
    Initialize conversation file paths
    Args:
        exp (str): catchup

    Returns:
        None
    """
    model = config.get("LLM_CONFIG", "MODEL")
    run_name = apply_run_name_override(config)
    file_path_list = list()
    SF_path_list = list()
    base_path = os.path.join(os.getcwd(), f"LLMConversation/", exp)
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%H")
    # file_name = f"{timestamp}_Con.json"
    folder_name = f"{timestamp}_{model}/{run_name}/E_{epi}"
    folder_name2 = f"{timestamp}_{model}/{run_name}/E_{epi}/SF"
    conv_path = os.path.join(base_path, folder_name)
    conv_path2 = os.path.join(base_path, folder_name2)
    os.makedirs(conv_path, exist_ok=True)
    os.makedirs(conv_path2, exist_ok=True)
    conversation_context = {
    "experiment": exp,
    "messages": []
    }
    convtime = datetime.now().strftime("%m%d%H%M")
    if all_agents is not None:
        for agent in all_agents:
            conv_name = f"{convtime}_{agent.agent_name}{agent.agent_id}.json"
            conv_name2 = f"{convtime}_{agent.agent_name}{agent.agent_id}.json"
            file_path = os.path.join(conv_path, conv_name)
            SF_file_path2 = os.path.join(conv_path2, conv_name2)

            file_path_list.append(file_path)
            SF_path_list.append(SF_file_path2)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(conversation_context, f, indent=4, ensure_ascii=False)
            with open(SF_file_path2, "w", encoding="utf-8") as f:
                json.dump(conversation_context, f, indent=4, ensure_ascii=False)
            print(f"---------- Step1:'{exp}' saved to {file_path}. ----------.")
    else:
        for agent_ind in range(n_agent):
            convtime = datetime.now().strftime("%m%d%H%M")
            conv_name = f"{convtime}_agent{agent_ind+1}.json"
            conv_name2 = f"{convtime}_agent{agent_ind+1}.json"
            file_path = os.path.join(conv_path, conv_name)
            SF_file_path2 = os.path.join(conv_path2, conv_name2)

            file_path_list.append(file_path)
            SF_path_list.append(SF_file_path2)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(conversation_context, f, indent=4, ensure_ascii=False)
            with open(SF_file_path2, "w", encoding="utf-8") as f:
                json.dump(conversation_context, f, indent=4, ensure_ascii=False)
            print(f"---------- Step1:'{exp}' saved to {file_path}. ----------")
    return file_path_list, conversation_context, SF_path_list

def read_txt_file(file_path):
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file() and path.suffix == '.txt':
        with path.open('r', encoding='utf-8') as file:
            content = file.read()
        return content
    else:
        return f"No such .txt file: {path}"

def extract_from_label(message, label):
    assert f'<{label}>' in message and f'</{label}>' in message
    start = message.index(f'<{label}>') + len(f'<{label}>')
    end = message.index(f'</{label}>')
    return message[start:end]

def initialize_comm_matrix(num_agents, observable_range):
    comm_matrix = np.zeros((num_agents, num_agents), dtype=int)
    for i in range(num_agents):
        for j in range(max(0, i - observable_range), min(num_agents, i + observable_range + 1)):  # Ensure within bounds
            comm_matrix[i, j] = 1
    return comm_matrix

def record_proposal_to_agents_comm_range(agent_list, comm_range, proposal_dict, round, episode):
    received_proposal_dict = dict()
    for agent in agent_list:
        received_proposal_dict[f"{agent.agent_name}{agent.agent_id}"] = [{"role": "assistant", "name": f"{agent.agent_name}{agent.agent_id}","content": proposal_dict[f"{agent.agent_name}{agent.agent_id}"]}]
    for proposal_dict_key in proposal_dict:
        proposal = proposal_dict[proposal_dict_key]
        proposal_message = {
            "role": "assistant",
            "name": f"{proposal_dict_key}",
            "content": proposal
        }
        # sender_id = int(proposal_dict_key.split('_')[-1])
        match = re.search(r'\d+', proposal_dict_key)
        if match:
            sender_id = int(match.group())
        for agent in agent_list:
            if sender_id!=agent.agent_id and abs(sender_id - agent.agent_id) <= comm_range:
                agent.append_other_proposal_to_file(
                        proposal_content=proposal,
                        proposal_name=proposal_dict_key
                    )
                if sender_id<agent.agent_id:
                    proposal_SF = f"[{proposal_dict_key}'s (the Vehicle Ahead of Me) Round{round} Proposal in Episode_{episode}]: \n {proposal_dict_key} proposes {proposal}"
                    user_message = {
                        "role": "assistant",
                        "name": f"{proposal_dict_key}",
                        "content": f"{proposal_SF}"
                    }
                    agent.append_other_proposal_to_SF(proposal_content = user_message)
                else:
                    proposal_SF = f"[{proposal_dict_key}'s (the Vehicle Behind of Me) Round{round} Proposal in Episode_{episode}]: \n {proposal_dict_key} proposes {proposal}"
                    user_message = {
                        "role": "assistant",
                        "name": f"{proposal_dict_key}",
                        "content": f"{proposal_SF}"
                    }
                    agent.append_other_proposal_to_SF(proposal_content = user_message)
                received_proposal_dict[f"{agent.agent_name}{agent.agent_id}"].append(proposal_message)

                # print(f"{proposal_dict_key} received by agent {agent.agent_id}")

    return received_proposal_dict

def parse_vehicle_output_list(proposals_by_vehicle):
    parsed_dict = {}
    for vehicle, proposal_str_list in proposals_by_vehicle.items():
        vehicle_dict = {}
        for proposal_str in proposal_str_list:
            name_match = re.search(r'E\d+_R\d+_(vehicle\d+)_proposal', proposal_str)
            if not name_match:
                continue
            prop_vehicle = name_match.group(1)
            dict_match = re.search(r'=\s*(\{[\s\S]*\})', proposal_str)
            if not dict_match:
                continue
            dict_str = dict_match.group(1)
            try:
                proposal_dict = ast.literal_eval(dict_str)
            except Exception as e:
                continue
            vehicle_dict[prop_vehicle] = proposal_dict
        parsed_dict[vehicle] = vehicle_dict
    return parsed_dict

def collect_output_for_each_vehicle(proposal_round_dict):
    vehicle_to_proposals = {}
    for vehicle, proposal_str in proposal_round_dict.items():
        related_vehicles = [v for v in proposal_round_dict if f'"{v}"' in proposal_str]
        proposals = [proposal_round_dict[v] for v in related_vehicles]
        if vehicle not in related_vehicles:
            proposals.append(proposal_round_dict[vehicle])
        proposals = list(dict.fromkeys(proposals))
        vehicle_to_proposals[vehicle] = proposals
    parsed_dict = parse_vehicle_output_list(vehicle_to_proposals)
    return parsed_dict

def get_comm_agent(agent_id, comm_matrix, agent_list):
    comm_agents = []
    for agent_ind in range(len(comm_matrix[agent_id-1])):
        if comm_matrix[agent_id-1][agent_ind] == 1 and agent_ind != agent_id-1:
            comm_agents.append(agent_list[agent_ind])
    return comm_agents

# def string_extract_number(string):
#     matches = re.findall(r'[+-]?\d+(?:\.\d+)?', string)
#     numbers = [float(num) for num in matches]
#     # print(numbers)
#     return numbers

def string_extract_number(string):
    matches = re.findall(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', string)
    numbers = [float(num) for num in matches]
    return numbers

def check_format_label(query_content, check_label):
    if query_content is None:
        return False
    if f'<{check_label}>' in query_content and f'</{check_label}>' in query_content:
        return True
    else:
        return False


def build_json_schema_from_format_list(format_list):
    """
    Convert a format label list to JSON schema.
    Supports special compatibility modes:
    - ['action', 'final_decision']: require either one
    - ['deal', 'action', 'reasons']: deal=True requires action, deal=False requires reasons
    """
    if not isinstance(format_list, list) or len(format_list) == 0:
        return None

    normalized = []
    for item in format_list:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped and stripped not in normalized:
                normalized.append(stripped)

    label_set = set(normalized)

    if label_set == {"action", "final_decision"}:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "final_decision": {"type": "string"},
            },
            "additionalProperties": False,
            "anyOf": [
                {"required": ["action"]},
                {"required": ["final_decision"]},
            ],
        }

    if {"deal", "action", "reasons"}.issubset(label_set):
        return {
            "type": "object",
            "properties": {
                "deal": {"type": "string", "enum": ["True", "False"]},
                "action": {"type": "string"},
                "reasons": {"type": "string"},
            },
            "required": ["deal"],
            "additionalProperties": False,
            "anyOf": [
                {
                    "properties": {"deal": {"const": "True"}},
                    "required": ["deal", "action"],
                },
                {
                    "properties": {"deal": {"const": "False"}},
                    "required": ["deal", "reasons"],
                },
            ],
        }

    properties = {}
    for label in normalized:
        properties[label] = {"type": "string"}

    return {
        "type": "object",
        "properties": properties,
        "required": normalized,
        "additionalProperties": False,
    }


def render_tagged_response_from_json(parsed_json, format_list):
    """
    Render parsed json fields back to <tag>...</tag> format for legacy parser.
    Supports special compatibility modes used by legacy parsing.
    """
    if not isinstance(parsed_json, dict):
        raise ValueError("parsed_json must be a dict.")

    if not isinstance(format_list, list):
        raise ValueError("format_list must be a list.")

    normalized = []
    for label in format_list:
        if not isinstance(label, str):
            continue
        key = label.strip()
        if not key:
            continue
        if key not in normalized:
            normalized.append(key)

    label_set = set(normalized)
    output_chunks = []

    if label_set == {"action", "final_decision"}:
        if "action" in parsed_json:
            output_chunks.append(f"<action>{parsed_json.get('action', '')}</action>")
        elif "final_decision" in parsed_json:
            output_chunks.append(f"<final_decision>{parsed_json.get('final_decision', '')}</final_decision>")
        return "\n".join(output_chunks)

    if {"deal", "action", "reasons"}.issubset(label_set):
        deal_value = str(parsed_json.get("deal", "")).strip()
        output_chunks.append(f"<deal>{deal_value}</deal>")
        if deal_value == "True":
            output_chunks.append(f"<action>{parsed_json.get('action', '')}</action>")
        else:
            output_chunks.append(f"<reasons>{parsed_json.get('reasons', '')}</reasons>")
        return "\n".join(output_chunks)

    for key in normalized:
        value = parsed_json.get(key, "")
        output_chunks.append(f"<{key}>{value}</{key}>")

    return "\n".join(output_chunks)


def generate_structured_json_openai_response(
    default_client,
    default_model,
    messages,
    json_schema,
    temperature,
    # max_tokens=1024,
    max_tokens=None,
):
    """
    Use OpenAI/Azure response_format=json_schema to force structured output.
    Returns (response_obj, raw_text, parsed_json).
    """
    response = default_client.chat.completions.create(
        model=default_model,
        messages=messages,
        temperature=temperature,
        # max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": json_schema,
                "strict": True,
            },
        },
    )

    raw_text = response.choices[0].message.content
    if not isinstance(raw_text, str):
        raise ValueError(f"Structured output content is not string: {type(raw_text)}")

    parsed_json = json.loads(raw_text)
    if not isinstance(parsed_json, dict):
        raise ValueError("Structured output is not a JSON object.")

    return response, raw_text, parsed_json

def calculate_proposal_overlap(agent_e, received_output_list, MAX = 5):
        # agent_key = f"{agent_e.agent_name}{agent_e.agent_id}"
        # agent_dict = received_output_list.get(agent_key, {})
        # result = {}
        # diff_mean = {}
        # neighbors = [f'{agent_e.agent_name}{item}' for item in agent_e.observe_id_list if item != agent_e.agent_id]
        # for neighbor in neighbors:
        #     if neighbor in received_output_list:
        #         neighbor_dict = received_output_list[neighbor]
        #         common_keys = set(agent_dict.keys()) & set(neighbor_dict.keys())
        #         if common_keys:
        #             result[neighbor] = {k: neighbor_dict[k] for k in common_keys}
        #             diffs = [(neighbor_dict[k] - agent_dict[k]) for k in common_keys]
        #             diff_mean[neighbor] = sum(diffs)/MAX / len(diffs) if diffs else 0.0
        # overall_mean = sum(diff_mean.values()) / len(diff_mean) if diff_mean else 0.0
        agent_key = f"{agent_e.agent_name}{agent_e.agent_id}"
        neighbors = [f'{agent_e.agent_name}{item}' for item in agent_e.observe_id_list if item != agent_e.agent_id]
        result = {}
        result[agent_key] = {}
        diff_mean = {}
        for neighbor in neighbors:
            result[agent_key][neighbor] = {}
            for check_value in agent_e.observe_id_list:
                proposed_agent_key = f"{agent_e.agent_name}{check_value}"

                agent_dict = received_output_list.get(agent_key, {})
                my_proposal = agent_dict.get(proposed_agent_key)
                if my_proposal is None or not isinstance(my_proposal, (int, float)):
                    print(f"lost proposal: {proposed_agent_key}")
                    my_proposal = 0.0
                try:
                    if neighbor in received_output_list:
                        if proposed_agent_key in received_output_list[neighbor]:
                            val = received_output_list[neighbor][proposed_agent_key]
                            if isinstance(val, (int, float)) and val is not None:
                                neighbor_proposal = val
                            else:
                                neighbor_proposal = 0
                                print(f"lost proposal: {proposed_agent_key}")
                        else:
                            neighbor_proposal = 0
                    else:
                        neighbor_proposal = 0
                    result[agent_key][neighbor][proposed_agent_key] = abs(neighbor_proposal - my_proposal)
                except KeyError as e:
                    print(f"KeyError: {e} not found in received_output_list. All keys: {list(received_output_list.keys())}")
                    neighbor_proposal = 0
                    result[agent_key][neighbor][proposed_agent_key] = abs(neighbor_proposal - my_proposal)

        for main_key, sub_dict in result.items():
            diff_mean[main_key] = {}
            for sub_key, sub_sub_dict in sub_dict.items():
                if isinstance(sub_sub_dict, dict) and sub_sub_dict:
                    values = list(sub_sub_dict.values())
                    avg = sum(values) / len(values)
                    diff_mean[main_key][sub_key] = avg/MAX*100
                else:
                    diff_mean[main_key][sub_key] = None
        values = list(diff_mean[f'{agent_e.agent_name}{agent_e.agent_id}'].values())
        overall_mean = sum(values) / len(values)
        return result, diff_mean, overall_mean

def clean_values(s: str) -> str:
    def replacer(match):
        inside = match.group(1)
        nums = re.findall(r'\d+', inside)
        if len(set(nums)) == 1:
            return nums[0]
        elif len(nums) == 0:
            return "None"
        else:
            print(f"[WARN] 多个数字: {inside} -> {nums}")
            return "None"

    return re.sub(r'<([^>]*)>', replacer, s)

def safe_eval_dict(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    fixed_lines = []
    for line in s.splitlines():
        if ":" in line:
            has_comma = line.strip().endswith(",")
            key, val = line.split(":", 1)

            val = re.sub(r'"([^"]*)"\s*(\d+)', r'"\1 \2"', val)

            if val.count('"') > 2:
                parts = re.findall(r'"([^"]*)"', val)
                merged = " ".join(p.strip() for p in parts if p.strip())
                val = f' "{merged}"'

            new_line = f"{key}:{val}"
            if has_comma and not new_line.strip().endswith(","):
                new_line += ","

            fixed_lines.append(new_line)
        else:
            fixed_lines.append(line)

    fixed = "\n".join(fixed_lines)

    try:
        return ast.literal_eval(fixed)
    except Exception as e:
        print(f"[WARN] 修复失败: {e}\n原始字符串:\n{s}\n修复后:\n{fixed}")
        return s


def _extract_dict_single_int_from_str(s: str):
    result = {}
    key_value_pairs = re.findall(r'"([^"]+)"\s*:\s*([^,}\]]+)|(\w+)\s*:\s*([^,}\]]+)', s)
    for parts in key_value_pairs:
        if parts[0]:
            k, val = parts[0], parts[1].strip()
        else:
            k, val = parts[2], parts[3].strip()
        nums = re.findall(r'\d+', val)
        if len(nums) == 1:
            result[k] = int(nums[0])
        else:
            return None
    return result if result else None


def transfer_string_to_dict(s):
    received_output_list = {}
    for key, raw in s.items():
        if not isinstance(raw, str):
            print(f"[WARN] {key} 的 raw 不是字符串: {type(raw)}，值为: {raw}")
            pass
            continue
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            continue
        dict_str = match.group(0)
        dict_str = clean_values(dict_str)
        dict_str = re.sub(r'\*\*(\d+)\*\*', r'\1', dict_str)

        def replace_single_int_bold(m):
            inner = m.group(1)
            nums = re.findall(r'\d+', inner)
            if len(nums) == 1:
                return nums[0]
            return m.group(0)

        dict_str = re.sub(r'\*\*([^*]+)\*\*', replace_single_int_bold, dict_str)

        # 1) key: → "key":
        safe_str = re.sub(r'(\w+)\s*:', r'"\1":', dict_str)
        safe_str = re.sub(r':\s*([A-Za-z_]\w*)', r':"\1"', safe_str)
        try:
            # tmp_dict = ast.literal_eval(safe_str)
            tmp_dict = safe_eval_dict(safe_str)
        except Exception as e:
            print(f"[ERROR] {key} 解析失败: {e}")
            pass
            continue
        if not isinstance(tmp_dict, dict):
            fallback = _extract_dict_single_int_from_str(dict_str)
            if fallback is not None:
                tmp_dict = fallback
            if not isinstance(tmp_dict, dict):
                print(f"[WARN] {key} 的 output 解析后非 dict (type={type(tmp_dict).__name__})，跳过并置空。raw 片段: {repr(raw[:200])}...")
                received_output_list[key] = {}
                continue
        cleaned = {}

        for k, v in tmp_dict.items():
            if isinstance(v, int):
                cleaned[k] = v
                continue
            if isinstance(v, str):
                nums = re.findall(r'\d+', v)
                if len(set(nums)) == 1:
                    cleaned[k] = int(nums[0])
                elif len(nums) == 0:
                    cleaned[k] = None
                else:
                    print(f"[DEBUG] {key}.{k} -> {v} 多个数字 {nums}")
                    pass
            else:
                cleaned[k] = None
        received_output_list[key] = cleaned
    return received_output_list

def calculate_proposal_overlap_pan(current_key, observe_id_list, received_output_list, MAX=5):
    neighbors = [k for k in received_output_list.keys() if k != current_key]
    result = {current_key: {}}
    diff_mean = {current_key: {}}

    for neighbor in neighbors:
        result[current_key][neighbor] = {}
        for check_value in observe_id_list:
            my_proposal = received_output_list[current_key].get(check_value, 0)
            neighbor_proposal = received_output_list[neighbor].get(check_value, 0)
            if neighbor_proposal is None or my_proposal is None:
                continue
            result[current_key][neighbor][check_value] = abs(neighbor_proposal - my_proposal)

    for neighbor, sub_dict in result[current_key].items():
        if sub_dict:
            values = list(sub_dict.values())
            avg = sum(values) / len(values)
            diff_mean[current_key][neighbor] = avg / MAX * 100
        else:
            diff_mean[current_key][neighbor] = None

    values = [v for v in diff_mean[current_key].values() if v is not None]
    overall_mean = sum(values) / len(values) if values else None

    return result, diff_mean, overall_mean

def transfer_cacc_state_dict_to_list(state_dict):
    pattern = r"\{(?:\s*'vehicle\d+': \([^)]+\),?)+\s*\}"
    match = re.search(pattern, state_dict['content'])
    result = []
    action_result = []
    if match:
        vehicles_dict_str = match.group()
        vehicles_dict = ast.literal_eval(vehicles_dict_str)

        for v, tup in vehicles_dict.items():
            arr = np.array(tup[:2])
            action = tup[2]
            result.append(arr)
            action_result.append(action)
    else:
        print("No vehicle state found!")
    return result, action_result

def make_delta_action(state_from, state_to, action):
    delta1 = state_to[0] - state_from[0]
    delta2 = state_to[1] - state_from[1]
    return np.concatenate([delta1, delta2, action])

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def calculate_transition_overlap(historical_state_dict_list, current_state_dict):
    state_t0, action_t0 = transfer_cacc_state_dict_to_list(historical_state_dict_list[-2])
    state_t1, action_t1 = transfer_cacc_state_dict_to_list(historical_state_dict_list[-1])
    state_t2, action_t2 = transfer_cacc_state_dict_to_list(current_state_dict)
    vec1 = make_delta_action(state_t0, state_t1, action_t1)   # [agent1_delta, agent2_delta, action1]
    vec2 = make_delta_action(state_t1, state_t2, action_t2)   # [agent1_delta, agent2_delta, action2]
    overlap = cosine_similarity(vec1, vec2) * 100
    normalized_overlap = (overlap + 1) / 2
    return normalized_overlap

def label_exist(message, label):
    if not isinstance(message, str):
        return False
    has_flag = f'<{label}>' in message and f'</{label}>' in message
    return has_flag

def initialize_system_prompt(exp, epi, n_agent, n_people, config, agents_list):
    for agent in agents_list:
        task_description = f"""[System Role Instruction]
        You are an administrator of {agent.agent_name} in a small city with a population of {n_people} people. """
        task_description += read_txt_file(pandemic['path'])
        agent.system_prompt = {"role": "system", "content": task_description}
        agent.append_system_prompt_to_file(system_prompt=task_description)
        agent.append_system_prompt_to_SF(system_prompt=task_description)
    return

def RecordOtherProposal(agent_list, proposal_dict, round, episode):
    received_proposal_list_dict = dict()
    for agent_i in agent_list:
        received_proposal_list_dict[agent_i.agent_name] = list()
        for observe_name in agent_i.observe_id_list:
            if observe_name == 'Grocery':
                pass
            proposal = proposal_dict[observe_name]
            user_message = {
            "role": "assistant",
            "name": observe_name,
            "content": f"{observe_name} proposes: \n {proposal}"
            }
            received_proposal_list_dict[agent_i.agent_name].append(user_message)
            agent_i.append_other_proposal_to_SF(proposal_content = user_message)
            agent_i.append_other_proposal_to_file(proposal_content = user_message, proposal_name=observe_name)
    return received_proposal_list_dict


def ReceivedProposalDictList_Pan(agent, received_proposal_dict):
    agent_received_proposal_list = list()
    agent_key = f"{agent.agent_name}{agent.agent_id}"
    my_dict = {
        "role": "assistant",
        "name": f"{agent_key}",
        "content": f"I proppose: {received_proposal_dict[agent_key][0]['content']}"
        }
    agent_received_proposal_list.append(my_dict)
    for proposal_dict_key in received_proposal_dict.keys():
        if proposal_dict_key != agent_key:
            proposal = received_proposal_dict[proposal_dict_key][0]['content']
            proposal_message = {
                "role": "assistant",
                "name": f"{proposal_dict_key}",
                "content": f"{proposal_dict_key} proposes: {proposal}"
            }
            agent_received_proposal_list.append(proposal_message)
    return agent_received_proposal_list
