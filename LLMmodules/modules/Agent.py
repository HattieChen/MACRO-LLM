import openai
# from openai import OpenAI, RateLimitError
from openai import AzureOpenAI, OpenAI, RateLimitError
import time
import json, os
import numpy as np
import configparser
from pathlib import Path
from LLMmodules.TaskConfiguration import catchup
from LLMmodules.model_config import apply_model_override, infer_model_flag, load_runtime_config
from LLMmodules.modules.utils import MAX_LLM_RESPONSE_ATTEMPTS, read_txt_file, extract_from_label, check_format_label, label_exist, build_json_schema_from_format_list, render_tagged_response_from_json, generate_structured_json_openai_response, raise_if_llm_retry_exhausted
from LLMmodules.CACC.catchup_utils import optimal_acc, welford_update_from_stats, welford_ew_update

from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


_cacc_env = os.environ.get('CACC_ENV', 'catchup')
catchup_flag = (_cacc_env == 'catchup')
print(f"Agent catchup_flag: {catchup_flag}")

# True for catchup, False for slowdown
if catchup_flag:
    config_path = PROJECT_ROOT / 'algorithms/envs/NCS/config/config_ma2c_nc_catchup.ini'
    TASK_FLAG = 1
    config = configparser.ConfigParser()
    config.read(str(config_path))
    VEHICLE_NUM = config.getint("ENV_CONFIG", "n_vehicle")
else:
    config_path = PROJECT_ROOT / 'algorithms/envs/NCS/config/config_ma2c_nc_slowdown.ini'
    TASK_FLAG = 2
    config = configparser.ConfigParser()
    config.read(str(config_path))
    VEHICLE_NUM = config.getint("ENV_CONFIG", "n_vehicle")


load_runtime_config(PROJECT_ROOT / "config")
MODEL = apply_model_override(config)
FLAG = infer_model_flag(MODEL, config.get("LLM_CONFIG", "MODEL_FLAG", fallback="Llama"))
OPENAI_MODEL1 = MODEL
azure_client1 = None
azure_client2 = None
client = None
ACC = 2.5

if FLAG == 'Gemini':
    proxy_url = os.environ.get("GOOGLE_PROXY_URL")
    if proxy_url:
        os.environ['http_proxy'] = proxy_url
        os.environ['https_proxy'] = proxy_url
        os.environ['all_proxy'] = proxy_url
    OPENAI_MODEL1 = config.get("LLM_CONFIG", "MODEL")
    client = genai.Client(api_key=require_env("GOOGLE_API_KEY"))

if FLAG == 'GPT':
    API_KEY1 = require_env("OPENAI_API_KEY")
    API_ENDPOINT1 = os.environ.get("OPENAI_ENDPOINT")
    OPENAI_MODEL1 = config.get("LLM_CONFIG", "MODEL")
    azure_client1 = AzureOpenAI(
        api_key=API_KEY1,
        azure_endpoint=API_ENDPOINT1,
        azure_deployment=OPENAI_MODEL1,
        api_version="2025-04-01-preview",
    )
    print(f"{OPENAI_MODEL1} client initialized")


elif FLAG == 'Ali':
    API_KEY1 = require_env("DASHSCOPE_API_KEY")
    BASE_URL1 = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    # OPENAI_MODEL = "deepseek-r1"
    # OPENAI_MODEL = "deepseek-v3"
    # OPENAI_MODEL = "qwen3-235b-a22b"
    # OPENAI_MODEL = "qwen3-32b"
    OPENAI_MODEL1 = config.get("LLM_CONFIG", "MODEL")
    azure_client1 = OpenAI(
    api_key=API_KEY1,
    base_url=BASE_URL1,
    )

elif FLAG == 'Llama':
    # ---------- Llama: Llama-3.3-70b-instruct -----------------------
    API_KEY1 = require_env("OPENROUTER_API_KEY")
    BASE_URL1 = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENAI_MODEL1 = config.get("LLM_CONFIG", "MODEL")
    azure_client1 = OpenAI(
        api_key=API_KEY1,
        base_url=BASE_URL1,
    )
    print(f"{OPENAI_MODEL1} client initialized")

# Generate lst dynamically based on VEHICLE_NUM
lst = list(range(1, VEHICLE_NUM + 1))

class CPPAgent():
    def __init__(self, AGENT_ID: str, element, CONFIG):
        """
        Initialize a CPPAgent instance.

        Args:
        """
        # Initialize VSAgent-specific attributes
        self.agent_id = AGENT_ID
        if self.agent_id == 1:
            self.observe_id_list = [1,2]
        elif self.agent_id == VEHICLE_NUM:
            self.observe_id_list = [VEHICLE_NUM-1, VEHICLE_NUM]
        else:
            self.observe_id_list = [self.agent_id-1, self.agent_id, self.agent_id+1]
        start = lst.index(self.observe_id_list[0])
        end = start + len(self.observe_id_list)
        self.unobserve_id_list_behind = lst[end:]
        self.unobserve_id_list_ahead = lst[:start]
        self.historical_state = []
        self.TOM_history = []
        self.ToM_insights = ''
        self.agent_name = element
        self.current_time = 0  # Current time for viewport or bandwidth operations
        self.wait_time = 300   # Initial wait time for retry operations
        # self.over_episode_insights = []
        self.over_episode_insights = ""
        self.PG_insights = []
        self.PE_insights = []
        self.RP_insights = []
        self.FO_insights = []
        self.spatial_insights = ""
        self.temporal_insights = ""
        self.system_prompt = ""
        self.state_prompt = ""
        self.gen_proposal_over_episode_insights = []
        self.gen_reproposal_over_episode_insights = []
        self.proposal_evaluation_over_episode_insights = []
        self._load_config_env(CONFIG['ENV_CONFIG'])
        self._load_config_llm(CONFIG['LLM_CONFIG'])
        self.conv_path = ''
        self.SF_record_path = ''
        # weighted acceleration
        self.forward_WAA = 0
        self.backward_WAA = 0
        self.forward_WV = 0
        self.backward_WV = 0
        self.updated_rate = 0.7
        # average velocity
        self.forward_WAV = 0
        self.forward_WVV = 0
        self.backward_WAV = 0
        self.backward_WVV = 0
        # average headway
        self.forward_WAH = 0
        self.forward_WVH = 0
        self.backward_WAH = 0
        self.backward_WVH = 0

    def __repr__(self):
        return f"Agent {self.agent_id} with model {self.model} and temperature = {self.temp} "

    def _load_config_env(self, config):
        self.dt = config.getfloat('control_interval_sec')
        self.T = int(config.getint('episode_length_sec') / self.dt)
        self.batch_size = config.getint('batch_size')
        self.h_min = config.getfloat('headway_min')
        self.h_star = config.getfloat('headway_target')
        self.h_norm = config.getfloat('norm_headway')
        self.h_s = config.getfloat('headway_st')
        self.h_g = config.getfloat('headway_go')
        self.v_max = config.getfloat('speed_max')
        self.v_star = config.getfloat('speed_target')
        self.v_norm = config.getfloat('norm_speed')
        self.u_min = config.getfloat('accel_min')
        self.u_max = config.getfloat('accel_max')
        self.name = config.get('scenario').split('_')[1]
        self.exp = config.get('scenario').split('_')[1]
        self.a = config.getfloat('reward_v')
        self.b = config.getfloat('reward_u')
        self.G = config.getfloat('collision_penalty')
        self.n_agent = config.getint('n_vehicle')
        self.agent = config.get('agent')
        self.coop_gamma = config.getfloat('coop_gamma')
        self.seed = config.getint('seed')
        test_seeds = [int(s) for s in config.get('test_seeds').split(',')]

    def _load_config_llm(self, config):
        # self.model = config.get('MODEL')
        self.model = OPENAI_MODEL1
        self.temp = config.getfloat('TEMP')
        self.MAX_MESSAGE = config.getint('MAX_MESSAGE')
        self.REMAIN_MESSAGE = config.getint('REMAIN_MESSAGE')
        self.xgrammar_flag = config.getint('XGRAMMAR_FLAG', fallback=0)
        self.V_FLAG = config.get('VELOCITY_FLAG')
        self.target_meter = config.get('TARGET_DIS')

    def append_user_question_to_file(self, user_question, update = False):
        try:
            with open(self.conv_path, "r", encoding="utf-8") as f:
                conversation_context = json.load(f)
            # NOTE: update the conversation file
            if len(conversation_context["messages"]) >= self.MAX_MESSAGE and update == True:
                msgs = conversation_context["messages"]
                conversation_context["messages"] = [{"role": "system", "content": "Previous messages truncated."}] + msgs[:1] + msgs[-(self.REMAIN_MESSAGE):]
                if "agent_u" not in self.conv_path:
                    self.conv_path = self.conv_path.replace("agent", "agent_u")
            if "messages" not in conversation_context:
                conversation_context["messages"] = []
            user_message = {
                "role": f"user",
                # "name": f"agent{self.agent_id}",
                "content": user_question
            }
            conversation_context["messages"].append(user_message)
            # print(f"Appended user question to file: {user_question}")
            with open(self.conv_path, "w", encoding="utf-8") as f:
                json.dump(conversation_context, f, indent=4, ensure_ascii=False)
            # print(f"Updated conversation context saved to {self.conv_path}")

        except FileNotFoundError:
            print(f"Error: File not found at {self.conv_path}")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file at {self.conv_path}")
        except Exception as e:
            print(f"Unexpected error: type={type(e).__name__}, args={e.args}, msg={e}")


    def gen_proposal_catchup(self, epi, round, last_reward = 0, state=[], record_flag = False, case_flag = 0, weight_flag = 0):
        '''
        Transfer state list to a detailed question

        Args:
            state_list_example: [{'vehicle_1': (20.0, 25.204045333151537, 0.0), 'vehicle_2': (20.0, 25.204045333151537, 0.0)}, {'vehicle_1': (20.0, 25.204045333151537, 0.0), 'vehicle_2': (20.0, 25.204045333151537, 0.0), 'vehicle_3': (20.0, 25.204045333151537, 0.0)}, ..., {'vehicle_7': (20.0, 25.204045333151537, 0.0), 'vehicle_8': (20.0, 25.204045333151537, 0.0)}]
        '''
        # self.append_system_prompt_to_file(system_prompt=read_txt_file(catchup['reward']))
        state_dict = state[self.agent_id-1]
        # observable state info
        # v_lead = state_dict[f"{self.agent_name}{self.agent_id-1}"][1] if f"{self.agent_name}{self.agent_id-1}" in state_dict else 15
        v_lead = state_dict[f"{self.agent_name}{self.agent_id}"][2]
        if self.agent_id == 1:
            state_info = f'''[{self.agent_name}{str(self.agent_id)} Observable State in Episode_{epi}]
            {str(state_dict)}
            The dict represents {{vehicle0 is the virtual target vehicle with constant velocity {state_dict[f"{self.agent_name}{self.agent_id-1}"][1]} m/s.
            {self.agent_name}{self.agent_id} is driving towards virtual target vehicle and the distance from {self.agent_name}{self.agent_id} to the virtual target vehicle is {state_dict[f"{self.agent_name}{self.agent_id}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id}"][3]} m/s^2.
            {self.agent_name}{self.agent_id+1} is driving towards {self.agent_name}{self.agent_id} and the distance from {self.agent_name}{self.agent_id+1} to {self.agent_name}{self.agent_id} is {state_dict[f"{self.agent_name}{self.agent_id+1}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id+1}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id+1}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id+1}"][2]} m/s, its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id+1}"][3]} m/s^2}}.'''

        elif self.agent_id == VEHICLE_NUM:
            state_info = f'''[{self.agent_name}{str(self.agent_id)} Observable State in Episode_{epi}]
            {str(state_dict)}
            The dict represents {{\n{self.agent_name}{self.agent_id-1} is driving towards {self.agent_name}{self.agent_id-2} and the distance from {self.agent_name}{self.agent_id-1} to the {self.agent_name}{self.agent_id-2} is {state_dict[f"{self.agent_name}{self.agent_id-1}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id-1}\'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id-1}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id-1}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id-1}"][3]} m/s^2.
            {self.agent_name}{self.agent_id} is driving towards {self.agent_name}{self.agent_id-1} and the distance from {self.agent_name}{self.agent_id} to {self.agent_name}{self.agent_id-1} is {state_dict[f"{self.agent_name}{self.agent_id}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id}\'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id}"][3]} m/s^2}}.
            '''

        elif self.agent_id == 2:
            state_info = f'''[{self.agent_name}{str(self.agent_id)} Observable State in Episode_{epi}]
            {str(state_dict)}
            The dict represents {{\n{self.agent_name}{self.agent_id-1} is driving towards virtual target vehicle and the distance from {self.agent_name}{self.agent_id-1} to the virtual target vehicle is {state_dict[f"{self.agent_name}{self.agent_id-1}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id-1}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id-1}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id-1}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id-1}"][3]} m/s^2.
            {self.agent_name}{self.agent_id} is driving towards {self.agent_name}{self.agent_id-1} and the distance from {self.agent_name}{self.agent_id} to {self.agent_name}{self.agent_id-1} is {state_dict[f"{self.agent_name}{self.agent_id}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id}"][3]} m/s^2.
            {self.agent_name}{self.agent_id+1} is driving towards {self.agent_name}{self.agent_id} and the distance from {self.agent_name}{self.agent_id+1} to {self.agent_name}{self.agent_id} is {state_dict[f"{self.agent_name}{self.agent_id+1}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id+1}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id+1}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id+1}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id+1}"][3]} m/s^2}}.'''

        else:
            state_info = f'''[{self.agent_name}{str(self.agent_id)} Observable State in Episode_{epi}]
            {str(state_dict)}
            The dict represents {{\n{self.agent_name}{self.agent_id-1} is driving towards {self.agent_name}{self.agent_id-2} and the distance from {self.agent_name}{self.agent_id-1} to {self.agent_name}{self.agent_id-2} is {state_dict[f"{self.agent_name}{self.agent_id-1}"][0]} m; its target distance is 20m; {self.agent_name}{self.agent_id-1}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id-1}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id-1}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id-1}"][3]} m/s^2.
            {self.agent_name}{self.agent_id} is driving towards {self.agent_name}{self.agent_id-1} and the distance from {self.agent_name}{self.agent_id} to {self.agent_name}{self.agent_id-1} is {state_dict[f"{self.agent_name}{self.agent_id}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id}\'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id}"][3]} m/s^2.
            {self.agent_name}{self.agent_id+1} is driving towards {self.agent_name}{self.agent_id} and the distance from {self.agent_name}{self.agent_id+1} to {self.agent_name}{self.agent_id} is {state_dict[f"{self.agent_name}{self.agent_id+1}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id+1}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id+1}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id+1}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id+1}"][3]} m/s^2}}.'''
        self.state_prompt = {"role": "system", "content": state_info}
        self.append_system_prompt_to_file(system_prompt=state_info)
        self.append_system_prompt_to_SF(system_prompt=state_info) # SF record
        self.historical_state.append(self.state_prompt)
        # QUERY: Generate Proposal
        # 0. My current strategy

        # 1. back info
        query_proposal = f"""" [QUERY: Generate Proposal]
        You are a decision-maker for {self.agent_name}{self.agent_id} in a cooperative vehicle control scenario. This is negotiation round {round} with your neighbors. Your task is to propose an acceleration action in next {self.dt} seconds for both yourself ({self.agent_name}{self.agent_id}) """
        # if self.V_FLAG == '1':
        #     if self.agent_id == 1:
        #     #     query_proposal = f"""" [QUERY: Generate Proposal]
        #     # You are a decision-maker for {self.agent_name}{self.agent_id} with strict calculation capacity in a cooperative vehicle control scenario. The target distance to the vehicle ahead is {self.target_meter} meters, the target velocity is {v_lead} m/s. This is the {round} round for negotiation with your neighbors. Your task is to propose a one-second acceleration action for both yourself ({self.agent_name}{self.agent_id}) """
        #         query_proposal = f"""" [QUERY: Generate Proposal]
        #         You are a decision-maker for {self.agent_name}{self.agent_id} with strict calculation capacity in a cooperative vehicle control scenario. This is the {round} round for negotiation with your neighbors. Your task is to propose an acceleration action in next {self.dt} seconds for both yourself ({self.agent_name}{self.agent_id}) """
        #     else:
        #         # v_lead = state[self.agent_id-1][f'{self.agent_name}{self.agent_id-1}'][1]
        #         query_proposal = f"""" [QUERY: Generate Proposal]
        #         You are a decision-maker for {self.agent_name}{self.agent_id} with strict calculation capacity in a cooperative vehicle control scenario. This is the {round} round for negotiation with your neighbors. Your task is to propose an acceleration action in next {self.dt} seconds for both yourself ({self.agent_name}{self.agent_id}) """
        # else:
        #     query_proposal = f"""" [QUERY: Generate Proposal] You are a decision-maker for {self.agent_name}{self.agent_id} with strict calculation capacity in a cooperative vehicle control scenario. The target catchup distance to the vehicle ahead of you is {self.target_meter} meters, and the target velocity is {v_lead} m/s. This is the {round} round for negotiation with your neighbors. Your task is to propose a one-second acceleration action for yourself ({self.agent_name}{self.agent_id}) """
        # 2. neighbor info
        if self.agent_id == 1:
            query_proposal += f""" and your neighbor vehicle ({self.agent_name}{self.agent_id + 1}, behind you). Your target distance to the virtual target vehicle is {self.target_meter} meters, your target velocity is {v_lead} m/s. """
        elif self.agent_id == VEHICLE_NUM:
            query_proposal += f""" and your neighbor vehicle ({self.agent_name}{self.agent_id - 1}, ahead of you). The target distance to the vehicle ahead is {self.target_meter} meters, the target velocity is {v_lead} m/s. """
        else:
            query_proposal += f""" and your neighbor vehicles ({self.agent_name}{self.agent_id - 1}, ahead of you; and {self.agent_name}{self.agent_id + 1}, behind of you). The target distance to the vehicle ahead is {self.target_meter} meters, the target velocity is {v_lead} m/s."""

        query_proposal += f"Decision-Making Steps (Follow strictly, step by step):\n "
        # if state_dict[f"{self.agent_name}{self.agent_id}"][0] > 20:
        #     query_proposal += f"""You need to reduce the distance to the vehicle ahead of you."""
        # elif state_dict[f"{self.agent_name}{self.agent_id}"][0] == 20:
        #     query_proposal += f"""You just achieve the target distance to the vehicle ahead fo you."""
        # else:
        #     query_proposal += f"""You need to increase the distance to the vehicle ahead of you."""

        # 3. refered acceleration
        if self.V_FLAG == '1':
            if case_flag == 0: # 0: self-calculate optimal acceleration with linear weights
                query_proposal += f""" according to the following instructions step-by-step.
                Step 0: Action Selection for {self.agent_name}{self.agent_id}
                If the current distance of {self.agent_name}{self.agent_id} is not equal to {self.target_meter} meters, and the current velocity is not equal to {v_lead} m/s at the same time, do the following steps: 1) Calculate the optimal acceleration based on distance by using [Equation: optimal action according to distance], and obtain distance_optimal_acceleration. 2) Calculate the optimal acceleration based on lead velocity by using [Equation: optimal action according to velocity], and obtain velocity_optimal_acceleration. 3) Use a linear weighted sum to combine these two accelerations into a final acceleration action: final_acceleration=w_distance * distance_optimal_acceleration + w_velocity * velocity_optimal_acceleration, where w_distance and w_velocity are the weights for distance and velocity and w_distance + w_velocity = 1. 4) Define the urgency based on the calculated acceleration: normal: acceleration in range (-0.5, 0.5), warning: acceleration in range (-{ACC}, -0.5] or [0.5, {ACC}). urgent: acceleration <= -{ACC} or acceleration >= {ACC}. 5) Clip the resulting action to the safe range [-{ACC}, {ACC}] \n.
                """
            elif case_flag == 1: # 1. self-calculate optimal acceleration with absolute acceleration as weights
                query_proposal += f""" according to the following instructions step-by-step.
                Step 0: Action Selection for {self.agent_name}{self.agent_id}. If the current distance of {self.agent_name}{self.agent_id} is not equal to {self.target_meter} meters and the current velocity is not equal to {v_lead} m/s at the same time, do the following steps: 1) Calculate the optimal acceleration based on distance by using [Equation: optimal action according to distance], and obtain distance_optimal_acceleration. 2) Calculate the optimal acceleration based on velocity by using [Equation: optimal action according to velocity], and obtain velocity_optimal_acceleration. 3) Calculate the weights for distance and velocity based on the absolute acceleration demand: w_distance = |distance_optimal_acceleration|/(|velocity_optimal_acceleration|+|distance_optimal_acceleration|), w_velocity = |velocity_optimal_acceleration|/(|velocity_optimal_acceleration|+|distance_optimal_acceleration|). final_acceleration=w_distance * distance_optimal_acceleration + w_velocity * velocity_optimal_acceleration, where w_distance and w_velocity are the weights for distance and velocity and w_distance + w_velocity = 1. 4) Define the urgency based on the calculated acceleration: normal: acceleration in range (-0.5, 0.5), warning: acceleration in range (-{ACC}, -0.5] or [0.5, {ACC}). urgent: acceleration <= -{ACC} or acceleration >= {ACC}. 5) Clip the resulting action to the safe range [-{ACC}, {ACC}] \n.
                    """
            else:
                if f"{self.agent_name}{self.agent_id-1}" not in state_dict:
                    if catchup_flag:
                        ahead_velocity = 15
                        ahead_acceleration = 0
                    else:
                        velocity_list = np.linspace(25.204, 15, 30).tolist()
                        if epi < len(velocity_list):
                            ahead_velocity = velocity_list[epi]
                            ahead_acceleration = (15 - 25.204) / (30 - 1)
                        else:
                            ahead_velocity = 15
                            ahead_acceleration = 0
                else:
                    ahead_velocity = state_dict[f"{self.agent_name}{self.agent_id-1}"][1]
                    ahead_acceleration = state_dict[f"{self.agent_name}{self.agent_id-1}"][3]
                my_headway = state_dict[f"{self.agent_name}{self.agent_id}"][0]
                my_velocity = state_dict[f"{self.agent_name}{self.agent_id}"][1]
                my_acceleration = state_dict[f"{self.agent_name}{self.agent_id}"][3]
                my_vh = state_dict[f"{self.agent_name}{self.agent_id}"][4]
                opt_acc, w_headway, headway_optimal_acc, w_velocity, velocity_optimal_acc = optimal_acc(ahead_velocity, my_headway, my_velocity, flag = weight_flag, v_flag = self.V_FLAG, dt=0.5, task_flag=TASK_FLAG)
                # opt_acc = ref_acc(ahead_velocity=ahead_velocity, my_distance=my_headway, my_velocity = my_velocity, my_last_acceleration=my_acceleration, ahead_acceleration=ahead_acceleration)
                # opt_acc = vh_acc(my_velocity=my_velocity, vh=my_vh, dt=0.5)

                # query_proposal += f"""According to the following instructions step-by-step.
                #     Step 0: Action Selection for {self.agent_name}{self.agent_id}
                #     The distance_optimal_acceleration is {headway_optimal_acc}. The velocity_optimal_acceleration is {velocity_optimal_acc}. The initial weight {w_headway} for {headway_optimal_acc} and weight {w_velocity} for {velocity_optimal_acc} based on """
                # if weight_flag == 0:
                #     query_proposal += "linear normalized distance and velocity gaps. Calulate the acceleration as: acceleration = w_headway * headway_optimal_acc + w_velocity * velocity_optimal_acc."
                # elif weight_flag == 1:
                #     query_proposal += "importance normalized by the acceleration values. Calulate the acceleration as: acceleration = w_headway * headway_optimal_acc + w_velocity * velocity_optimal_acc."
                # else:
                #     # query_proposal += "sigmod weighting distance and velocity gaps. Calulate my optimal acceleration as: acceleration = w_headway * headway_optimal_acc + w_velocity * velocity_optimal_acc."
                #     query_proposal += f"sigmod weighting distance and velocity gaps. Obtain my weighted acceleration by: {w_headway} * {headway_optimal_acc} + {w_velocity} * {velocity_optimal_acc} and obtain {opt_acc}"

                query_proposal += f"Step 0: Initial Decision for {self.agent_name}{self.agent_id} at the 1st time step. \n - According to your state, the optimal initial acceleration for 1st time step is {opt_acc}. \n - Define urgency of the action: Normal: acceleration in range (-0.5, 0.5); Warning: acceleration in range (-{ACC}, -0.5] or [0.5, {ACC}); Urgent: acceleration <= -{ACC} or acceleration >= {ACC}. \n - Clip the resulting acceleration for the 1st time step to the safe range [-{ACC}, {ACC}] \n."

        else: # Ignore target velocity setting
            if case_flag == 0:
                query_proposal += f""" according to the following instructions step-by-step.
                Step 0: Action Selection for {self.agent_name}{self.agent_id} If the current distance of {self.agent_name}{self.agent_id} is not equal to {self.target_meter} meters, and the current velocity is not equal to {v_lead} at the same time, please do the following: 1) Calculate the optimal acceleration based on distance by using [Equation: optimal action according to distance], and obtain distance_optimal_acceleration. 2) Calculate the optimal acceleration based on velocity by using [Equation: optimal action according to velocity], and obtain velocity_optimal_acceleration. 3) Use a linear weighted sum to combine these two accelerations into a final acceleration action: final_acceleration=w_distance * distance_optimal_acceleration + w_velocity * velocity_optimal_acceleration, where w_distance and w_velocity are the weights for distance and velocity and w_distance + w_velocity = 1. 4) Define the urgency based on the calculated acceleration: normal: acceleration in range (-0.5, 0.5), warning: acceleration in range (-{ACC}, -0.5] or [0.5, {ACC}). urgent: acceleration <= -{ACC} or acceleration >= {ACC}. 5) Clip the resulting action to the safe range [-{ACC}, {ACC}] \n. """
            elif case_flag == 1: # absolute acceleration as weights
                query_proposal += f""" according to the following instructions step-by-step. \n Step 0: Action Selection for {self.agent_name}{self.agent_id}. If the current distance of {self.agent_name}{self.agent_id} is not equal to {self.target_meter} meters, and the current velocity is not equal to {v_lead} at the same time, please do the following: 1) Calculate the optimal acceleration based on distance by using [Equation: optimal action according to distance], and obtain distance_optimal_acceleration. 2) Calculate the optimal acceleration based on velocity by using [Equation: optimal action according to velocity], and obtain velocity_optimal_acceleration. 3) Calculate the weights for distance and velocity based on the absolute acceleration demand: w_distance = |distance_optimal_acceleration|/(|velocity_optimal_acceleration|+|distance_optimal_acceleration|), w_velocity = |velocity_optimal_acceleration|/(|velocity_optimal_acceleration|+|distance_optimal_acceleration|). final_acceleration=w_distance * distance_optimal_acceleration + w_velocity * velocity_optimal_acceleration, where w_distance and w_velocity are the weights for distance and velocity and w_distance + w_velocity = 1. 4) Define the urgency based on the calculated acceleration: normal: acceleration in range (-0.5, 0.5), warning: acceleration in range (-{ACC}, -0.5] or [0.5, {ACC}). urgent: acceleration <= -{ACC} or acceleration >= {ACC}. 5) Clip the resulting action to the safe range [-{ACC}, {ACC}] \n. """
            elif case_flag == 2:
                if f"{self.agent_name}{self.agent_id-1}" not in state_dict:
                    # ahead_velocity = 15
                    ahead_velocity = v_lead
                    ahead_acceleration = 0
                else:
                    # ahead_velocity = state_dict[f"{self.agent_name}{self.agent_id-1}"][1]
                    ahead_velocity = v_lead
                    ahead_acceleration = state_dict[f"{self.agent_name}{self.agent_id-1}"][3]
                my_headway = state_dict[f"{self.agent_name}{self.agent_id}"][0]
                my_velocity = state_dict[f"{self.agent_name}{self.agent_id}"][1]
                my_acceleration = state_dict[f"{self.agent_name}{self.agent_id}"][3]
                opt_acc, w_headway, headway_optimal_acc, w_velocity, velocity_optimal_acc = optimal_acc(ahead_velocity, my_headway, my_velocity, flag = weight_flag, v_flag=self.V_FLAG, dt = self.dt)

                if int(config["LLM_CONFIG"]["SELF_DEFINE_FLAG"]) == 1:
                    query_proposal += f"""According to the following instructions step-by-step.
                    Step 1: Action Selection for {self.agent_name}{self.agent_id}
                    According to [Equation: optimal action according to distance], the distance_optimal_acceleration is {headway_optimal_acc} to achieve target distance to the vehilce ahead. According to [Equation: optimal action according to velocity], the velocity_optimal_acceleration is {velocity_optimal_acc} to catchup the vehicle ahead of you. You need to set weights of these three acceleration to determine the optimal acceleration.
                    """
                else:
                    query_proposal += f""" Step 1: Action Selection for {self.agent_name}{self.agent_id}
                    According to [Equation: optimal action according to distance], the distance_optimal_acceleration is {headway_optimal_acc}. According to [Equation: optimal action according to velocity], the velocity_optimal_acceleration is {velocity_optimal_acc}. With weight {w_headway} for {headway_optimal_acc} and weight {w_velocity} for {velocity_optimal_acc} based on """
                if weight_flag == 0:
                    query_proposal += "linear normalized distance and velocity gaps."
                elif weight_flag == 1:
                    query_proposal += "importance normalized by the acceleration values."
                else:
                    query_proposal += "sigmod weighting distance and velocity gaps."
                query_proposal += f"Define the urgency based on the calculated acceleration: normal: acceleration in range (-0.5, 0.5), warning: acceleration in range (-{ACC}, -0.5] or [0.5, {ACC}). urgent: acceleration <= -{ACC} or acceleration >= {ACC}. Clip the resulting action to the safe range [-{ACC}, {ACC}] \n."
            else: # 0628 original prompts
                query_proposal += f"""According to the following instructions step-by-step. Step 0: Action Selection for {self.agent_name}{self.agent_id}. If the current distance of {self.agent_name}{self.agent_id} is not equal to {self.target_meter} meters, calculate the optimal acceleration by calling [Equation: optimal action according to distance]. Else if distance equal to {self.target_meter} meters, calculate the optimal acceleration for {self.agent_name}{self.agent_id} by calling [Equation: optimal action according to velocity]. Define the urgency based on the calculated acceleration: normal: acceleration in range (-0.5, 0.5), warning: acceleration in range (-{ACC}, -0.5] or [0.5, {ACC}). urgent: acceleration <= -{ACC} or acceleration >= {ACC}. Clip the resulting action to the safe range [-{ACC}, {ACC}] \n."""

        # test environment flag
        if int(config["LLM_CONFIG"]["KEEP_TEST_FLAG"]) == 1:
            only_for_weighted_method = f"""
            Output with following format strictly.
                <analysis>
                (Calculation steps. final_decision = {w_headway}* {headway_optimal_acc} + {w_velocity}*{velocity_optimal_acc})
                </analysis>
                <proposal>
                (I am {self.agent_name}{self.agent_id}, I choose ... as my acceleration, because ...
                )
                </proposal>
                <output>
                E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>
                }}
                </output>"""
            query_proposal += only_for_weighted_method
            return query_proposal, state_info
        else:
            if self.agent_id == 1:  # for vehicle1
                if int(config["LLM_CONFIG"]["SIMU_STEP_FLAG"]) == 3: # 0721: updated two-step without reward calculation

                    query_proposal += f""" Step 1: Collaborative Acceleration for {self.agent_name}{self.agent_id + 1}.
                    - Based on the clipped acceleration for {self.agent_name}{self.agent_id} selected in Step 0 and [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}], select a collaborative strategy from [Collaboration Instruction] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id + 1}) and compute the optimal collaborative acceleration for {self.agent_name}{self.agent_id + 1} to achieve current Objective in [Temporal Plan].

                    Step 2-4: Validation by 2 Time-Step Simulation: Use the proposed accelerations for time step 1 to simulate the following two time steps (time step 2 and time step 3), in order to verify whether your proposed action for the next {self.dt}s is feasible and safe. They are not part of the final proposed action, and do not need to be included in the output proposal.
                    - Step 2: Simulate the next state (state_2) at time step 2 by calling [Function: Next_State Computation].
                    - Step 3: Calculate collaborative actions for state_2 by [Equation: optimal action according to distance] and [Equation: optimal action according to velocity]. Combine (weight) both to decide the acceleration for time step 2.
                    - Step 4: Call [Function: Next_State Computation] to get state_3 for {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id + 1}.

                    Step 5: Feasibility Check
                    - For both state_2 and state_3:
                    - The distance between vehicles must be greater than 10 meters.
                    - The velocity of each vehicle must be less than 26 m/s.
                    - If any constraint is violated, go back to Step 0 and pick a more conservative action for vehicle1 (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat Steps 0-4 up to 10 times.
                    - If no feasible proposal is found after 10 iterations, output the best proposal found.

                    Output Format (STRICT):
                    The verified proposal in 1st time step are the final proposal. 2nd–3rd time steps are for verification:
                    <analysis>
                    (Show each calculation step in detail:
                    - List all variables and substitute their values
                    - Display all intermediate results
                    - Double-check the final answer)
                    </analysis>
                    <proposal>
                    I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id+1}), because ...
                    For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    For {self.agent_name}{self.agent_id+1} (my neighbor behind of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    </proposal>
                    <output>
                    E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                    "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                    {self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle BEHIND of{self.agent_name}{self.agent_id}>
                    }}
                    </output>
                    """
                elif int(config["LLM_CONFIG"]["SIMU_STEP_FLAG"]) == 2:
                    query_proposal += f""" Step 1: Collaborative Action for {self.agent_name}{self.agent_id + 1}. Based on the selected action for {self.agent_name}{self.agent_id} and my current objective as analyzed in [Temporal Plan], analyze [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}], select a collaborative strategy in [Collaboration Instruction] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id + 1}) to achieve GLOBAL OBJECTIVE as soon as possible. According to the selected collaborative strategies, determine the optimal collaborative action for {self.agent_name}{self.agent_id + 1}.
                    Step 2: Call [Function: Next_State Computation] to calculate the next state at t=1 (state_1) for {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id + 1} using their observable states and selected accelerations. Calculate the reward scores for vehicles: R_v{self.agent_id}_t1 ({self.agent_name}{self.agent_id}), R_v{self.agent_id+1}_t1 ({self.agent_name}{self.agent_id+1}) by calling [Function: Reward_score Calculation].
                    Step 3: Simulate the Following Time Step (t=2). For both {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id + 1}, repeat Steps 0 to 2 to select individually optimal actions by calling [Equation: optimal action according to distance] or [Equation: optimal action according to velocity] for each vehicle at t=2 (using state at t=1 obtained in step 2). Compute the reward scores for t=2: R_v{self.agent_id}_t2 and R_v{self.agent_id + 1}_t2.
                    Step 4: Calculate total system rewards: R1 = R_v{self.agent_id}_t1 + R_v{self.agent_id + 1}_t1 and R2 = R_v{self.agent_id}_t2 + R_v{self.agent_id + 1}_t2.
                    Step 5: Feasibility Check
                    (1) If both R1 > -3000 and R2 > -3000 and R2 >= R1, output the current action proposal.
                    (2) For each vehilce in t=1 and t=2, distance must higher than 10 meters, and the velocity must less than 20 m/s.
                    If any constraint is violated, go back to Step 0 and pick a more conservative action for {self.agent_name}{self.agent_id} (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
                    Output Format (STRICT): Your response must contain three sections, wrapped in the following tags and in the order shown:
                    <analysis>
                    (Calculation steps and analysis.)
                    </analysis>
                    <proposal>
                    I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id+1}), because ...
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id}, acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id+1} (behind of {self.agent_name}{self.agent_id}), acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    </proposal>
                    <output>
                    E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                    "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                    {self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle BEHIND>
                    }}
                    </output>
                    """
                elif int(config["LLM_CONFIG"]["SIMU_STEP_FLAG"]) == 1:
                    query_proposal += f"""
                    Step 1: Collaborative Action for {self.agent_name}{self.agent_id + 1}. Call [Function: Next_State Computation] to calculate the next state (at t=1) for both {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id + 1} using their current states and the {self.agent_name}{self.agent_id}'s selected actions.
                    With the predicted next state for {self.agent_name}{self.agent_id + 1} and the your [Spatial Analysis], determine the optimal collaborative action for {self.agent_name}{self.agent_id + 1}. Calculate the reward scores for both vehicles: R_v{self.agent_id}_t1 ({self.agent_name}{self.agent_id}) and R_v{self.agent_id + 1}_t1 ({self.agent_name}{self.agent_id + 1}) by calling [Function: Reward_score Calculation].
                    Calculate the reward scores for both vehicles: R_v{self.agent_id}_t1 ({self.agent_name}{self.agent_id}) and R_v{self.agent_id + 1}_t1 ({self.agent_name}{self.agent_id + 1}) by calling [Function: Reward_score Calculation].
                    Step 2: Feasibility Check
                    (1) If both R1 > -3000, output the current action proposal.
                    (2) Check whether your chosen action match your current [Temporal Plan].
                    (3) For each vehilce, distance must higher than 10 meters, and the velocity must less than 20 m/s.
                    If any constraint is violated, go back to Step 0 and pick a more conservative action for {self.agent_name}{self.agent_id} (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
                    <analysis>
                    (Calculation steps and analysis.)
                    </analysis>
                    <proposal>
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id}, acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id+1} (behind of {self.agent_name}{self.agent_id}), acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    </proposal>
                    <output>
                    E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                    "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                    {self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle BEHIND>
                    }}
                    </output>
                    """
                else:
                    query_proposal += f"""
                    Step 1: Collaborative Action for {self.agent_name}{self.agent_id + 1}. Call [Function: Next_State Computation] to calculate the next state (at t=1) for both {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id + 1} using their current states and the {self.agent_name}{self.agent_id}'s selected actions.
                    With the predicted next state for {self.agent_name}{self.agent_id + 1} and the your [Spatial Analysis], determine the optimal collaborative action for {self.agent_name}{self.agent_id + 1}. Calculate the reward scores for both vehicles: R_v{self.agent_id}_t1 ({self.agent_name}{self.agent_id}) and R_v{self.agent_id + 1}_t1 ({self.agent_name}{self.agent_id + 1}) by calling [Function: Reward_score Calculation].
                    Step 2: Simulate the Following Time Step (t=2). For both {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id + 1}, repeat Steps 0 and 1 to select individually optimal actions for t=2 (using the most up-to-date state at t=1). Compute the reward scores for t=2: R_v{self.agent_id}_t2 and R_v{self.agent_id + 1}_t2. Calculate total system rewards: R1 = R_v{self.agent_id}_t1 + R_v{self.agent_id + 1}_t1 and R2 = R_v{self.agent_id}_t2 + R_v{self.agent_id + 1}_t2.
                    Step 3: Feasibility Check
                    (1) If both R1 > -3000 and R2 > -3000, output the current action proposal.
                    (2) Check whether your chosen action match your current [Temporal Plan].
                    (3) For each vehilce in t=1 and t=2, distance must higher than 10 meters, and the velocity must less than 20 m/s.
                    If any constraint is violated, go back to Step 0 and pick a more conservative action for {self.agent_name}{self.agent_id} (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
                    Output Format (STRICT): Your response must contain three sections, wrapped in the following tags and in the order shown:
                    <analysis>
                    (Calculation steps and analysis.)
                    </analysis>
                    <proposal>
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id}, acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id+1} (behind of {self.agent_name}{self.agent_id}), acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    </proposal>
                    <output>
                    E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                    "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                    {self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle BEHIND>
                    }}
                    </output>
                    """
            elif self.agent_id == VEHICLE_NUM:  # for vehicle8
                if int(config["LLM_CONFIG"]["SIMU_STEP_FLAG"]) == 3: # 0721: updated two-step without reward calculation
                    query_proposal += f""" Step 1: Collaborative Acceleration for {self.agent_name}{self.agent_id - 1}.
                    - Based on the clipped acceleration for {self.agent_name}{self.agent_id} selected in Step 0 and [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}], select a collaborative strategy from [Collaboration Instruction] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id - 1}) and compute the optimal collaborative acceleration for {self.agent_name}{self.agent_id - 1} to achieve current Objective in [Temporal Plan].

                    Step 2-4: Validation by 2 Time-Step Simulation: Use the proposed accelerations for time step 1 to simulate the following two time steps (time step 2 and time step 3), in order to verify whether your proposed action for the next {self.dt}s is feasible and safe. They are not part of the final proposed action, and do not need to be included in the output proposal.
                    - Step 2: Simulate the next state (state_2) at time step 2 by calling [Function: Next_State Computation].
                    - Step 3: Calculate collaborative actions for state_2 by [Equation: optimal action according to distance] and [Equation: optimal action according to velocity]. Combine (weight) both to decide the acceleration for time step 2.
                    - Step 4: Call [Function: Next_State Computation] to get state_3 for {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id-1}.

                    Step 5: Feasibility Check
                    - For both state_2 and state_3:
                    - The distance between vehicles must be greater than 10 meters.
                    - The velocity of each vehicle must be less than 20 m/s.
                    - If any constraint is violated, go back to Step 0 and pick a more conservative action for vehicle1 (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat Steps 0-4 up to 10 times.
                    - If no feasible proposal is found after 10 iterations, output the best proposal found.

                    Output Format (STRICT):
                    The verified proposal in 1st time step are the final proposal. 2nd–3rd time steps are for verification:
                    <analysis>
                    (Show each calculation step in detail:
                    - List all variables and substitute their values
                    - Display all intermediate results
                    - Double-check the final answer)
                    </analysis>
                    <proposal>
                    I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id}), because ...
                    For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    For {self.agent_name}{self.agent_id-1} (my neighbor ahead of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    </proposal>
                    <output>
                    E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                    "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                    "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>
                    }}</output>"""
                elif int(config["LLM_CONFIG"]["SIMU_STEP_FLAG"]) == 2:
                    query_proposal += f""" Step 1: Collaborative Action for {self.agent_name}{self.agent_id - 1}. Based on the selected action for {self.agent_name}{self.agent_id} and my current objective as analyzed in [Temporal Plan], analyze [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}], select a collaborative strategy in [Collaboration Instruction] for ({self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id}) to achieve the GLOBAL OBJECTIVE as soon as possible. According to the selected collaborative strategies, determine the optimal collaborative action for {self.agent_name}{self.agent_id - 1}.
                    Step 2: Call [Function: Next_State Computation] to calculate the next state at t=1 (state_1) for {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id - 1} using their observable states and selected accelerations. Calculate the reward scores for vehicles: R_v{self.agent_id}_t1 ({self.agent_name}{self.agent_id}), R_v{self.agent_id-1}_t1 ({self.agent_name}{self.agent_id-1}) by calling [Function: Reward_score Calculation].
                    Step 3: Simulate the Following Time Step (t=2). For both {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id - 1}, repeat Steps 0 to 2 to select individually optimal actions by calling [Equation: optimal action according to distance] or [Equation: optimal action according to velocity] for each vehicle at t=2 (using state at t=1 obtained in step 2). Compute the reward scores for t=2: R_v{self.agent_id}_t2 and R_v{self.agent_id - 1}_t2.
                    Step 4: Calculate total system rewards: R1 = R_v{self.agent_id}_t1 + R_v{self.agent_id - 1}_t1 and R2 = R_v{self.agent_id}_t2 + R_v{self.agent_id - 1}_t2.
                    Step 5: Feasibility Check
                    (1) If both R1 > -3000 and R2 > -3000 and R2 >= R1, output the current action proposal.
                    (2) For each vehilce in t=1 and t=2, distance must higher than 10 meters, and the velocity must less than 20 m/s.
                    If any constraint is violated, go back to Step 0 and pick a more conservative action for {self.agent_name}{self.agent_id} (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
                    Output Format (STRICT): Your response must contain three sections, wrapped in the following tags and in the order shown:
                    <analysis>
                    (Calculation steps and analysis.)
                    </analysis>
                    <proposal>
                    I choose [collaboration strategy] for ({self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id}), because ...
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id}, acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id-1} (behind of {self.agent_name}{self.agent_id}), acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    </proposal>
                    <output>
                    E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                    "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                    {self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD>
                    }}
                    </output>
                    """
                elif int(config["LLM_CONFIG"]["SIMU_STEP_FLAG"]) == 1:
                    query_proposal += f"""
                    Step 1: Collaborative Action for {self.agent_name}{self.agent_id - 1}. Call [Function: Next_State Computation] to calculate the next state (at t=1) for both {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id - 1} using their current states and the {self.agent_name}{self.agent_id}'s selected actions.
                    With the predicted next state for {self.agent_name}{self.agent_id - 1} and the your [Spatial Analysis], determine the optimal collaborative action for {self.agent_name}{self.agent_id - 1}. Calculate the reward scores for both vehicles: R_v{self.agent_id}_t1 ({self.agent_name}{self.agent_id}) and R_v{self.agent_id - 1}_t1 ({self.agent_name}{self.agent_id - 1}) by calling [Function: Reward_score Calculation].
                    Step 2: Feasibility Check
                    (1) If both R1 > -3000, output the current action proposal.
                    (2) Check whether your chosen action match your current [Temporal Plan].
                    (3) For each vehilce, distance must higher than 10 meters, and the velocity must less than 20 m/s.
                    If any constraint is violated, go back to Step 0 and pick a more conservative action for {self.agent_name}{self.agent_id} (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
                    Output Format (STRICT): Your response must contain three sections, wrapped in the following tags and in the order shown:
                    <analysis>
                    (Calculation steps and analysis.)
                    </analysis>
                    <proposal>
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id}, acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id-1} (ahead of {self.agent_name}{self.agent_id}), acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    </proposal>
                    <output>
                    E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                    "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD>,
                    "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>
                    }}
                    </output>
                    """
                else:
                    query_proposal += f"""
                    Step 1: Collaborative Action for {self.agent_name}{self.agent_id - 1}. Call [Function: Next_State Computation] to calculate the next state (at t=1) for both {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id - 1} using their current states and the {self.agent_name}{self.agent_id}'s selected actions.
                    With the predicted next state for {self.agent_name}{self.agent_id - 1} and the your [Spatial Analysis], determine the optimal collaborative action for {self.agent_name}{self.agent_id - 1}. Calculate the reward scores for both vehicles: R_v{self.agent_id}_t1 ({self.agent_name}{self.agent_id}) and R_v{self.agent_id - 1}_t1 ({self.agent_name}{self.agent_id - 1}) by calling [Function: Reward_score Calculation].
                    Step 2: Simulate the Following Time Step (t=2). For both {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id - 1}, repeat Steps 0 and 1 to select individually optimal actions for t=2 (using the most up-to-date state at t=1). Compute the reward scores for t=2: R_v{self.agent_id}_t2 and R_v{self.agent_id - 1}_t2. Calculate total system rewards: R1 = R_v{self.agent_id}_t1 + R_v{self.agent_id - 1}_t1 and R2 = R_v{self.agent_id}_t2 + R_v{self.agent_id - 1}_t2.
                    Step 3: Feasibility Check
                    (1) If both R1 > -3000 and R2 > -3000, output the current action proposal.
                    (2) Check whether your chosen action match your current [Temporal Plan].
                    (3) For each vehilce in t=1 and t=2, distance must higher than 10 meters, and the velocity must less than 20 m/s.
                    If any constraint is violated, go back to Step 0 and pick a more conservative action for {self.agent_name}{self.agent_id} (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
                    Output Format (STRICT): Your response must contain three sections, wrapped in the following tags and in the order shown:
                    <analysis>
                    (Calculation steps and analysis.)
                    </analysis>
                    <proposal>
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id}, acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id-1} (ahead of {self.agent_name}{self.agent_id}), acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    </proposal>
                    <output>
                    E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                    "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD>,
                    "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>
                    }}
                    </output>
                    """
            else:   # for middle vehicles
                if int(config["LLM_CONFIG"]["SIMU_STEP_FLAG"]) == 3: # 0721: updated two-step without reward calculation
                    query_proposal += f""" Step 1: Collaborative Acceleration for {self.agent_name}{self.agent_id - 1} and {self.agent_name}{self.agent_id + 1}.
                    - Based on the clipped acceleration for {self.agent_name}{self.agent_id} selected in Step 0 and [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}], select a collaborative strategy from [Collaboration Instruction] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id - 1} and {self.agent_name}{self.agent_id + 1}) and compute the optimal collaborative acceleration for {self.agent_name}{self.agent_id - 1} and {self.agent_name}{self.agent_id + 1} to achieve current Objective in [Temporal Plan].

                    Step 2-4: Validation by 2 Time-Step Simulation: Use the proposed accelerations for time step 1 to simulate the following two time steps (time step 2 and time step 3), in order to verify whether your proposed action for the next {self.dt}s is feasible and safe. They are not part of the final proposed action, and do not need to be included in the output proposal.
                    - Step 2: Simulate the next state (state_2) at time step 2 by calling [Function: Next_State Computation].
                    - Step 3: Calculate collaborative actions for state_2 by [Equation: optimal action according to distance] and [Equation: optimal action according to velocity]. Combine (weight) both to decide the acceleration for time step 2.
                    - Step 4: Call [Function: Next_State Computation] to get state_3 for {self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id-1} and {self.agent_name}{self.agent_id+1}.

                    Step 5: Feasibility Check
                    - For both state_2 and state_3:
                    - The distance between vehicles must be greater than 10 meters.
                    - The velocity of each vehicle must be less than 20 m/s.
                    - If any constraint is violated, go back to Step 0 and pick a more conservative action for vehicle1 (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat Steps 0-4 up to 10 times.
                    - If no feasible proposal is found after 10 iterations, output the best proposal found.

                    Output Format (STRICT):
                    The verified proposal in 1st time step are the final proposal. 2nd–3rd time steps are for verification:
                    <analysis>
                    (Show each calculation step in detail:
                    - List all variables and substitute their values
                    - Display all intermediate results
                    - Double-check the final answer)
                    </analysis>
                    <proposal>
                    I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id}), because ...
                    I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id + 1}), because ...
                    For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    For {self.agent_name}{self.agent_id-1} (my neighbor ahead of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    For {self.agent_name}{self.agent_id+1} (my neighbor behind of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    </proposal>
                    <output>
                    E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                    "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                    "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>,
                    "{self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>
                    }}</output>"""
                elif int(config["LLM_CONFIG"]["SIMU_STEP_FLAG"]) == 2:
                    query_proposal += f""" Step 1: Collaborative Action for {self.agent_name}{self.agent_id - 1} and {self.agent_name}{self.agent_id + 1}. Based on the selected action for {self.agent_name}{self.agent_id} and my current objective as analyzed in [Temporal Plan], analyze [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}], select a collaborative strategy in [Collaboration Instruction] for ({self.agent_name}{self.agent_id - 1}, {self.agent_name}{self.agent_id}) and ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id + 1}) to achieve the GLOBAL OBJECTIVE as soon as possible. According to the selected collaborative strategies, determine the optimal collaborative action for {self.agent_name}{self.agent_id - 1} and {self.agent_name}{self.agent_id + 1}.
                    Step 2: Call [Function: Next_State Computation] to calculate the next state at t=1 (state_1) for {self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id + 1} using their observable states and selected accelerations. Calculate the reward scores for vehicles: R_v{self.agent_id-1}_t1 ({self.agent_name}{self.agent_id-1}), R_v{self.agent_id}_t1 ({self.agent_name}{self.agent_id}), R_v{self.agent_id+1}_t1 ({self.agent_name}{self.agent_id+1}) by calling [Function: Reward_score Calculation].
                    Step 3: Simulate the Following Time Step (t=2). For {self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id + 1}, repeat Steps 0 to 2 to select individually optimal actions by calling [Equation: optimal action according to distance] or [Equation: optimal action according to velocity] for each vehicle at t=2 (using state at t=1 obtained in step 2). Compute the reward scores for t=2: R_v{self.agent_id-1}_t2, R_v{self.agent_id}_t2 and R_v{self.agent_id + 1}_t2.
                    Step 4: Calculate total system rewards: R1 = R_v{self.agent_id-1}_t1 + R_v{self.agent_id}_t1 + R_v{self.agent_id + 1}_t1 and R2 = R_v{self.agent_id-1}_t2 + R_v{self.agent_id}_t2 + R_v{self.agent_id + 1}_t2.
                    Step 5: Feasibility Check
                    (1) If both R1 > -3000 and R2 > -3000 and R2 >= R1, output the current action proposal.
                    (2) For each vehilce in t=1 and t=2, distance must higher than 10 meters, and the velocity must less than 20 m/s.
                    If any constraint is violated, go back to Step 0 and pick a more conservative action for {self.agent_name}{self.agent_id} (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
                    Output Format (STRICT): Your response must contain three sections, wrapped in the following tags and in the order shown:
                    <analysis>
                    (Calculation steps and analysis.)
                    </analysis>
                    <proposal>
                    I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id-1}), because ...
                    I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id+1}), because ...
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id}, acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id-1} (ahead of {self.agent_name}{self.agent_id}), acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id+1} (behind of {self.agent_name}{self.agent_id}), acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    </proposal>
                    <output>
                    E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                    "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD>
                    "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                    "{self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle BEHIND>
                    }}
                    </output>
                    """
                elif int(config["LLM_CONFIG"]["SIMU_STEP_FLAG"]) == 1:
                    query_proposal += f"""
                    Step 1: Collaborative Action for {self.agent_name}{self.agent_id - 1} and {self.agent_name}{self.agent_id + 1}. Call [Function: Next_State Computation] to calculate the next state (at t=1) for {self.agent_name}{self.agent_id - 1}, {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id + 1} using their current states and the {self.agent_name}{self.agent_id}'s selected actions.
                    With the predicted next state and the your [Spatial Analysis], determine the optimal collaborative action for {self.agent_name}{self.agent_id - 1} and {self.agent_name}{self.agent_id + 1}.
                    Calculate the reward scores for vehicles: R_v{self.agent_id-1}_t1 ({self.agent_name}{self.agent_id-1}), R_v{self.agent_id}_t1 ({self.agent_name}{self.agent_id}) and R_v{self.agent_id + 1}_t1 ({self.agent_name}{self.agent_id + 1}) by calling [Function: Reward_score Calculation].
                    Step 2: Feasibility Check
                    (1) If both R1 > -3000, output the current action proposal.
                    (2) Check whether your chosen action match your current [Temporal Plan].
                    (3) For each vehilce, distance must higher than 10 meters, and the velocity must less than 20 m/s.
                    If any constraint is violated, go back to Step 0 and pick a more conservative action for {self.agent_name}{self.agent_id} (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
                    Output Format (STRICT): Your response must contain three sections, wrapped in the following tags and in the order shown:
                    <analysis>
                    (Calculation steps.)
                    </analysis>
                    <proposal>
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id}, acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id-1} (ahead of {self.agent_name}{self.agent_id}), acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id+1} (behind of {self.agent_name}{self.agent_id}), acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    </proposal>
                    <output>
                    E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                    "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD>,
                    "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                    "{self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle BEHIND>
                    }}
                    </output>
                    """
                else:
                    query_proposal += f"""
                    Step 1: Collaborative Action for {self.agent_name}{self.agent_id - 1} and {self.agent_name}{self.agent_id + 1}. Call [Function: Next_State Computation] to calculate the next state (at t=1) for {self.agent_name}{self.agent_id - 1}, {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id + 1} using their current states and the {self.agent_name}{self.agent_id}'s selected actions.
                    With the predicted next state and the your [Spatial Analysis], determine the optimal collaborative action for {self.agent_name}{self.agent_id - 1} and {self.agent_name}{self.agent_id + 1}.
                    Calculate the reward scores for vehicles: R_v{self.agent_id-1}_t1 ({self.agent_name}{self.agent_id-1}), R_v{self.agent_id}_t1 ({self.agent_name}{self.agent_id}) and R_v{self.agent_id + 1}_t1 ({self.agent_name}{self.agent_id + 1}) by calling [Function: Reward_score Calculation].
                    Step 2: Simulate the Following Time Step (t=2). For R_v{self.agent_id-1}_t1 ({self.agent_name}{self.agent_id-1}), {self.agent_name}{self.agent_id} and {self.agent_name}{self.agent_id + 1}, repeat Steps 0 and 1 to select individually optimal actions for t=2 (using the most up-to-date state at t=1). Compute the reward scores for t=2: R_v{self.agent_id-1}_t2, R_v{self.agent_id}_t2 and R_v{self.agent_id + 1}_t2. Calculate total system rewards: R1 = R_v{self.agent_id-1}_t1 + R_v{self.agent_id}_t1 + R_v{self.agent_id + 1}_t1 and R2 = R_v{self.agent_id-1}_t2+ R_v{self.agent_id}_t2 + R_v{self.agent_id + 1}_t2.
                    Step 3: Feasibility Check
                    (1) If both R1 > -3000 and R2 > -3000, output the current action proposal.
                    (2) Check whether your chosen action match your current [Temporal Plan].
                    (3) For each vehilce in t=1 and t=2, distance must higher than 10 meters, and the velocity must less than 20 m/s.
                    If any constraint is violated, go back to Step 0 and pick a more conservative action for {self.agent_name}{self.agent_id} (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
                    Output Format (STRICT): Your response must contain three sections, wrapped in the following tags and in the order shown:
                    <analysis>
                    (Calculation steps.)
                    </analysis>
                    <proposal>
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id}, acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id-1} (ahead of {self.agent_name}{self.agent_id}), acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    - proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id+1} (behind of {self.agent_name}{self.agent_id}), acceleration: [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                    </proposal>
                    <output>
                    E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                    "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD>,
                    "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                    "{self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle BEHIND>
                    }}
                    </output>
                    """

            query_proposal += f"""
            IMPORTANT:
            - All intermediate state predictions and reward calculations must strictly invoke the provided functions; no fabricated data is allowed.
            - When the distance between two vehicles matches the target distance, prioritize collaborative strategies like (ACCELERATION, ACCELERATION), (DECELERATION, DECELERATION), or (MAINTAINING, MAINTAINING) to maintain stable and cooperative behavior.
            - If MY_VELOCITY > VELOCITY_AHEAD, MY_DISTANCE will DECREASE. If MY_VELOCITY < VELOCITY_AHEAD, MY_DISTANCE will INCREASE.
            - If ACCELERATION > 0, MY_VELOCITY will be INCREASED, MY_DISTANCE will be DECREASED.
            - If ACCELERATION < 0, MY_VELOCITY will be DECREASED, MY_DISTANCE will be INCREASED.
            - Before output, double-check whether the reasoning in <proposal></proposal> matches these rules. If not, re-generate the proposal from Step 1.
            - Double-check the correctness of your calculations.
            - Show each calculation step in detail, clearly list all variables and values, and display every intermediate result.
            - Please ensure that your output always includes both opening (<analysis>, <proposal>, <output>) and closing tags (</analysis>, </proposal>, and </output>), such that every tag pair is properly closed and do not omit any closing tags.
            - Do not output any content outside the required tags.
            """


            if '<ProposalGeneration>' in self.over_episode_insights and '</ProposalGeneration>' in self.over_episode_insights:
                query_proposal += f"""
                    {extract_from_label(self.over_episode_insights, 'ProposalGeneration')}
                """
            combined_content = {
                "role": f"user",
                "content": query_proposal,
                }
            if record_flag == True:
                with open(self.conv_path, 'r') as file:
                    data = json.load(file)
                    data["messages"].append(combined_content)
                    with open(self.conv_path, 'w') as file:
                        json.dump(data, file, indent=4)
                self.append_user_question_to_file(user_question=query_proposal)
            # format_list = ['analysis', 'proposal', 'output']
            format_list = ['proposal', 'output', 'analysis'] # NOTE 0930: simplify
            return query_proposal, format_list






    # NOTE: CACC catchup ------------------------------------------------------

    '''
    2025-05-26
    '''
    '''
    def gen_reproposal(self, epi, round, current_rules, state=[]):
        query_proposal = f'You are Vehicle_{self.agent_id}. This is the {round}th round of negotiation in Episode {epi}. Consensus was not achieved among the nodes in the previous round. You need to update your proposal based on the rules summarized in the last round. \n Your observable states are [Vehicle_{str(self.agent_id)} Observable State in Episode_{epi}]. Your last round summarized rules are {current_rules}. \n '
        query_proposal += read_txt_file(task_step['repropose'])
        query_proposal += f"""
                            <output>
                            E<epi_index>_R<round_index>_Vehicle<vehicle_index>_proposal = {{
                                "vehicle_<AHEAD_index>": <proposed acceleration for vehicle AHEAD>,
                                "vehicle_<SELF_index>": <proposed acceleration for itself>,
                                "vehicle_<BEHIND_index>": <proposed acceleration for vehicle AHEAD>,
                            }}
                            </output>
                            Verify that the structured data in <output> ACCURATELY and COMPLETELY represents the decisions proposed in <proposal>. All vehicle references and acceleration values must be EXACTLY correct.
                            """
        combined_content = {
        "role": f"user",
        "name": f"agent{self.agent_id}",
        "content": json.dumps(query_proposal, indent=4),
        }
        with open(self.conv_path, 'r') as file:
            data = json.load(file)
            data["messages"].append(combined_content)
            with open(self.conv_path, 'w') as file:
                json.dump(data, file, indent=4)
        return combined_content
    '''
    def gen_reproposal(self, epi, round, current_rules = [], state=[], last_reward = 0, spatial_dict={}):
        '''
        Transfer state list to a detailed question
        Args:
        '''
        state_dict = state[self.agent_id-1]
        # if int(self.V_FLAG) == 1:
            # query_proposal = read_txt_file(catchup["rewardA"])
            # query_proposal = gen_reward_prompt(self.dt)
        # else:
            # query_proposal = read_txt_file(catchup["reward"])
        # query_proposal += reward_content
        self.update_spatial_insights(epi=epi, state_dict=state_dict, spatial_dict= spatial_dict)
        query_proposal = f"""[Current Spatial Insights]
        {self.spatial_insights}
        [Function: Next_State Computation]
        Given the my current state (distance, velocity), ahead vehicle state (distance_ahead, velocity_ahead), my action (acceleration), ahead vehicle action (proposed_current_ahead_acceleration, that is, you do not know the current_ahead_acceleration, you can propose an optimal current_ahead_acceleration for vehicle_ahead). The control interval is {self.dt} second.
        next_velocity = velocity + acceleration × {self.dt}
        ego_displacement = velocity × {self.dt} + 0.5 × acceleration × {self.dt}²
        ahead_displacement = velocity_ahead × {self.dt} + 0.5 × current_ahead_acceleration × {self.dt}²
        next_distance = distance + (ahead_displacement - ego_displacement)
        return next_velocity, next_distance"""
        if self.agent_id == 1:
            query_proposal += f"""[QUERY: Update Proposal]
                You are the decision maker for {self.agent_name}{self.agent_id} in a multi-agent negotiation scenario. This is the Round_{round} of Episode_{epi}. Your observable states is [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}]. You need to repropose by following steps. The driving direction of {self.agent_name}{self.agent_id} is the virtual target vehicle.

                - Case 1: If the flag between <deal></deal> is True, reuse the last proposal and output as following format:
                <analysis>
                (I will keep my last proposal, because ...)
                </analysis>
                <proposal>
                I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id+1}), because ...
                For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                For {self.agent_name}{self.agent_id+1} (my neighbor behind of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                </proposal>
                <output>
                E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                {self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle BEHIND of{self.agent_name}{self.agent_id}>
                }}
                </output>
                Skip the following steps.

                - Case 2: the flag between <deal></deal> is False:
                Step 1: Define the weights for observable neighbors' proposal and unobservable vehicles weighted average acceleration according to the following rules:
                - Rule 1: According to <reasons></reasons> summarized above, define neighbor_proposed_weight for {self.agent_name}{self.agent_id+1}'s proposal (behind me) in range [0,0.3].
                - Rule 2: According to <reasons></reasons> summarized above, define unobservable_weight for unobservable vehicles {self.unobserve_id_list_behind} (behind me) in range [0,0.3].
                - Rule 3: According to <reasons></reasons> summarized above, define my_weight for my proposal in range [0.3,1].
                - Rule 4: The weights need to satisfy neighbor_proposed_weight + unobservable_weight + my_weight = 1.
                Obtain the weights as following format:
                neighbor_proposed_weight: <neighbor_proposed_weight value>
                unobservable_weight: <unobservable_weight value>
                my_weight: <my_weight value>
                Step 2: Based on defined weights, weighted the proposed acceleration for each vehicles by following equations:
                updated_acceleration = neighbor_proposed_weight * neighbor_proposed_acceleration + my_weight * my_acceleration + unobservable_weight * {self.agent_name}{self.agent_id}'s backward weighted average acceleration
                Ensure each updated_acceleration is clipped in range [-{ACC}, {ACC}] m/s^2.
                Step 3: After determing the acceleration for each vehicle, 1) invoke: [Function: Next_State Computation] to check whether the next state satisfy velocity <= 25 and distance between all vehicles > 10m.
                If any constraint is violated, go back to Step 1 and adjust your proposals slightly and pick another action for {self.agent_name}{self.agent_id}. Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
                Step 4: Final output with following format:
                <analysis>
                (Show anlaysis and calculation step in detail:
                - List all variables and substitute their values
                - Display all intermediate results
                - Double-check the final answer)
                </analysis>
                <proposal>
                I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id+1}), because ...
                For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                For {self.agent_name}{self.agent_id+1} (my neighbor behind of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                </proposal>
                <output>
                E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                {self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle BEHIND of{self.agent_name}{self.agent_id}>
                }}
                </output>
                """
        elif self.agent_id == VEHICLE_NUM:
            query_proposal += f"""
            [QUERY: Update Proposal]
            You are the decision maker for {self.agent_name}{self.agent_id} in a multi-agent negotiation scenario. This is the Round_{round} of Episode_{epi}. Your observable states is [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}]. You need to repropose by following steps. The driving direction of {self.agent_name}{self.agent_id} is {self.agent_name}{self.agent_id-1}.

            - Case 1: If the flag between <deal></deal> is True, reuse the last proposal and output as following format:
            <analysis>
            (I will keep my last proposal, because ...)
            </analysis>
            <proposal>
            I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id}), because ...
            For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            For {self.agent_name}{self.agent_id-1} (my neighbor ahead of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            </proposal>
            <output>
            E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
            "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
            "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>
            }}</output>

            - Case 2: the flag between <deal></deal> is False:
            Step 1: Define the weights for observable neighbors' proposal and unobservable vehicles weighted average acceleration according to the following rules:
            - Rule 1: According to <reasons></reasons> summarized above, define neighbor_proposed_weight for {self.agent_name}{self.agent_id-1}'s proposal (ahead me) in range [0,0.3].
            - Rule 2: According to <reasons></reasons> summarized above, define unobservable_weight for unobservable vehicles {self.unobserve_id_list_ahead} (ahead me) in range [0,0.3].
            - Rule 3: According to <reasons></reasons> summarized above, define my_weight for my proposal in range [0.3,1].
            - Rule 4: The weights need to satisfy neighbor_proposed_weight + unobservable_weight + my_weight = 1.
            Obtain the weights as following format:
            neighbor_proposed_weight: <neighbor_proposed_weight value>
            unobservable_weight: <unobservable_weight value>
            my_weight: <my_weight value>
            Step 2: Based on defined weights, weighted the proposed acceleration for each vehicles by following equations:
            updated_acceleration = neighbor_proposed_weight * neighbor_proposed_acceleration + my_weight * my_acceleration + unobservable_weight * {self.agent_name}{self.agent_id}'s forward weighted average acceleration
            Ensure each updated_acceleration is clipped in range [-{ACC}, {ACC}] m/s^2.
            Step 3: After determing the acceleration for each vehicle, 1) invoke: [Function: Next_State Computation] to check whether the next state satisfy velocity <= 25 and distance between all vehicles > 10m.
            If any constraint is violated, go back to Step 1 and adjust your proposals slightly and pick another action for {self.agent_name}{self.agent_id}. Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
            Step 4: Final output with following format:
            <analysis>
            (Show anlaysis and calculation step in detail:
            - List all variables and substitute their values
            - Display all intermediate results
            - Double-check the final answer)
            </analysis>
            <proposal>
            I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id}), because ...
            For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            For {self.agent_name}{self.agent_id-1} (my neighbor ahead of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            </proposal>
            <output>
            E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
            "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
            "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>
            }}</output>
            """
        else:
            query_proposal += f"""
            [QUERY: Update Proposal]
            You are the decision maker for {self.agent_name}{self.agent_id} in a multi-agent negotiation scenario. This is the Round_{round} of Episode_{epi}. Your observable states is [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}]. You need to repropose by following steps. The driving direction of {self.agent_name}{self.agent_id} is {self.agent_name}{self.agent_id-1}.

            - Case 1: If the flag between <deal></deal> is True, reuse the last proposal and output as following format:
            <analysis>
            (I will keep my last proposal, because ...)
            </analysis>
            <proposal>
            I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id}), because ...
            I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id + 1}), because ...
            For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            For {self.agent_name}{self.agent_id-1} (my neighbor ahead of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            For {self.agent_name}{self.agent_id+1} (my neighbor behind of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            </proposal>
            <output>
            E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
            "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
            "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>,
            "{self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>
            }}</output>

            - Case 2: the flag between <deal></deal> is False:
            Step 1: Define the weights for observable neighbors' proposal and unobservable vehicles weighted average acceleration according to the following rules:
            - Rule 1: According to <reasons></reasons> summarized above, define neighbor_proposed_weight_ahead for {self.agent_name}{self.agent_id-1}'s proposal (ahead me) and neighbor_proposed_weight_behind for {self.agent_name}{self.agent_id+1}'s proposal (behind me) in range [0,0.3].
            - Rule 2: According to <reasons></reasons> summarized above, define unobservable_weight_forward for unobservable vehicles {self.unobserve_id_list_ahead} (ahead me) and unobservable_weight_backward for unobservable vehicles {self.unobserve_id_list_behind} (behind me) in range [0,0.3].
            - Rule 3: According to <reasons></reasons> summarized above, define my_weight for my proposal in range [0.3,1].
            - Rule 4: The weights need to satisfy neighbor_proposed_weight_ahead + neighbor_proposed_weight_behind + unobservable_weight_forward + unobservable_weight_backward + my_weight = 1. If not, repeat defining weights.
            Obtain the weights as following format:
            neighbor_proposed_weight: <neighbor_proposed_weight value>
            unobservable_weight: <unobservable_weight value>
            my_weight: <my_weight value>
            Step 2: Based on defined weights, weighted the proposed acceleration for each vehicles by following equations:
            updated_acceleration = neighbor_proposed_weight_ahead * {self.agent_name}{self.agent_id-1}_proposed_acceleration + neighbor_proposed_weight_behind * {self.agent_name}{self.agent_id+1}_proposed_acceleration + my_weight * my_acceleration + unobservable_weight_forward * {self.agent_name}{self.agent_id-1}'s forward weighted average acceleration + unobservable_weight_backward * {self.agent_name}{self.agent_id+1}'s backward weighted average acceleration
            Ensure each updated_acceleration is clipped in range [-{ACC}, {ACC}] m/s^2.
            Step 3: After determing the acceleration for each vehicle, 1) invoke: [Function: Next_State Computation] to check whether the next state satisfy velocity <= 25 and distance between all vehicles > 10m. If any constraint is violated, go back to Step 1 and adjust your proposals slightly and pick another action for {self.agent_name}{self.agent_id}. Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
            Step 4: Final output with following format:
            <analysis>
            (Show anlaysis and calculation step in detail:
            - List all variables and substitute their values
            - Display all intermediate results
            - Double-check the final answer)
            </analysis>
            <proposal>
            I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id}), because ...
            I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id + 1}), because ...
            For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            For {self.agent_name}{self.agent_id-1} (my neighbor ahead of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            For {self.agent_name}{self.agent_id+1} (my neighbor behind of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            </proposal>
            <output>
            E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
            "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
            "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>,
            "{self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>
            }}</output>
            """
        query_proposal += f"""
        IMPORTANT:
        - No fabricated data is allowed.
        - When the distance between two vehicles matches the target distance, prioritize collaborative strategies like (ACCELERATION, ACCELERATION), (DECELERATION, DECELERATION), or (MAINTAINING, MAINTAINING) to maintain stable and cooperative behavior.
        - If MY_VELOCITY > VELOCITY_AHEAD, MY_DISTANCE will DECREASE. If MY_VELOCITY < VELOCITY_AHEAD, MY_DISTANCE will INCREASE.
        - If ACCELERATION > 0, MY_VELOCITY will be INCREASED, MY_DISTANCE will be DECREASED.
        - If ACCELERATION < 0, MY_VELOCITY will be DECREASED, MY_DISTANCE will be INCREASED.
        - Before output, double-check whether the reasoning in <proposal></proposal> matches these rules. If not, re-generate the proposal from Step 1.
        - Double-check the correctness of your calculations and ensure that the acceleration in <analysis></analysis>, <proposal></proposal>, and <output></output> tags are consistent and correct.
        - Please ensure that your output always includes both opening (<analysis>, <proposal>, <output>) and closing tags (</analysis>, </proposal>, and </output>), such that every tag pair is properly closed and do not omit any closing tags.
        """

        if '<Reproposal>' in self.over_episode_insights and '</Reproposal>' in self.over_episode_insights:
            query_proposal += f"""
                {extract_from_label(self.over_episode_insights, 'Reproposal')}
            """
        reproposal_query_dict = {
        "role": f"user",
        "name": f"agent{self.agent_id}",
        "content": query_proposal,
        }
        with open(self.conv_path, 'r') as file:
            data = json.load(file)
            data["messages"].append(reproposal_query_dict)
            with open(self.conv_path, 'w') as file:
                json.dump(data, file, indent=4)
        return reproposal_query_dict



    def gen_reproposal_A(self, epi, round, current_rules = [], state=[], last_reward = 0, spatial_dict={}):
        '''
        Transfer state list to a detailed question
        Args:
        '''
        state_dict = state[self.agent_id-1]
        # if int(self.V_FLAG) == 1:
            # query_proposal = read_txt_file(catchup["rewardA"])
            # query_proposal = gen_reward_prompt(self.dt)
        # else:
            # query_proposal = read_txt_file(catchup["reward"])
        # query_proposal += reward_content
        self.update_spatial_insights(epi=epi, state_dict=state_dict, spatial_dict= spatial_dict)
        query_proposal = f"""[Current Spatial Insights]
        {self.spatial_insights}
        [Function: Next_State Computation]
        Given the my current state (distance, velocity), ahead vehicle state (distance_ahead, velocity_ahead), my action (acceleration), ahead vehicle action (proposed_current_ahead_acceleration, that is, you do not know the current_ahead_acceleration, you can propose an optimal current_ahead_acceleration for vehicle_ahead). The control interval is {self.dt} second.
        next_velocity = velocity + acceleration × {self.dt}
        ego_displacement = velocity × {self.dt} + 0.5 × acceleration × {self.dt}²
        ahead_displacement = velocity_ahead × {self.dt} + 0.5 × current_ahead_acceleration × {self.dt}²
        next_distance = distance + (ahead_displacement - ego_displacement)
        return next_velocity, next_distance"""
        if self.agent_id == 1:
            query_proposal += f"""[QUERY: Update Proposal]
                You are the decision maker for {self.agent_name}{self.agent_id} in a multi-agent negotiation scenario. This is the Round_{round} of Episode_{epi}. Your observable states is [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}]. You need to repropose by following steps. The driving direction of {self.agent_name}{self.agent_id} is the virtual target vehicle.

                - Case 1: If the flag between <deal></deal> is True, reuse the last proposal and output as following format:
                <analysis>
                (I will keep my last proposal, because ...)
                </analysis>
                <proposal>
                I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id+1}), because ...
                For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                For {self.agent_name}{self.agent_id+1} (my neighbor behind of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                </proposal>
                <output>
                E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                {self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle BEHIND of{self.agent_name}{self.agent_id}>
                }}
                </output>
                Skip the following steps.

                - Case 2: the flag between <deal></deal> is False:
                Step 1: Define the weights for observable neighbors' proposal and unobservable vehicles weighted average acceleration according to the following rules:
                - Rule 1: According to <reasons></reasons> summarized above, define neighbor_proposed_weight for {self.agent_name}{self.agent_id+1}'s proposal (behind me) in range [0,0.3].
                - Rule 2: According to <reasons></reasons> summarized above, define unobservable_weight for unobservable vehicles {self.unobserve_id_list_behind} (behind me) in range [0,0.3].
                - Rule 3: According to <reasons></reasons> summarized above, define my_weight for my proposal in range [0.3,1].
                - Rule 4: The weights need to satisfy neighbor_proposed_weight + unobservable_weight + my_weight = 1.
                Output the insights as following format:
                <insights>
                neighbor_proposed_weight: <neighbor_proposed_weight value>
                unobservable_weight: <unobservable_weight value>
                my_weight: <my_weight value>
                </insights>
                """
        elif self.agent_id == VEHICLE_NUM:
            query_proposal += f"""
            [QUERY: Update Proposal]
            You are the decision maker for {self.agent_name}{self.agent_id} in a multi-agent negotiation scenario. This is the Round_{round} of Episode_{epi}. Your observable states is [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}]. You need to repropose by following steps. The driving direction of {self.agent_name}{self.agent_id} is {self.agent_name}{self.agent_id-1}.

            - Case 1: If the flag between <deal></deal> is True, reuse the last proposal and output as following format:
            <analysis>
            (I will keep my last proposal, because ...)
            </analysis>
            <proposal>
            I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id}), because ...
            For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            For {self.agent_name}{self.agent_id-1} (my neighbor ahead of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            </proposal>
            <output>
            E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
            "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
            "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>
            }}</output>

            - Case 2: the flag between <deal></deal> is False:
            Step 1: Define the weights for observable neighbors' proposal and unobservable vehicles weighted average acceleration according to the following rules:
            - Rule 1: According to <reasons></reasons> summarized above, define neighbor_proposed_weight for {self.agent_name}{self.agent_id-1}'s proposal (ahead me) in range [0,0.3].
            - Rule 2: According to <reasons></reasons> summarized above, define unobservable_weight for unobservable vehicles {self.unobserve_id_list_ahead} (ahead me) in range [0,0.3].
            - Rule 3: According to <reasons></reasons> summarized above, define my_weight for my proposal in range [0.3,1].
            - Rule 4: The weights need to satisfy neighbor_proposed_weight + unobservable_weight + my_weight = 1.
            Output the insights as following format:
            <insights>
            neighbor_proposed_weight: <neighbor_proposed_weight value>
            unobservable_weight: <unobservable_weight value>
            my_weight: <my_weight value>
            </insights>
            """
        else:
            query_proposal += f"""
            [QUERY: Update Proposal]
            You are the decision maker for {self.agent_name}{self.agent_id} in a multi-agent negotiation scenario. This is the Round_{round} of Episode_{epi}. Your observable states is [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}]. You need to repropose by following steps. The driving direction of {self.agent_name}{self.agent_id} is {self.agent_name}{self.agent_id-1}.

            - Case 1: If the flag between <deal></deal> is True, reuse the last proposal and output as following format:
            <analysis>
            (I will keep my last proposal, because ...)
            </analysis>
            <proposal>
            I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id}), because ...
            I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id + 1}), because ...
            For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            For {self.agent_name}{self.agent_id-1} (my neighbor ahead of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            For {self.agent_name}{self.agent_id+1} (my neighbor behind of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            </proposal>
            <output>
            E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
            "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
            "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>,
            "{self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>
            }}</output>

            - Case 2: the flag between <deal></deal> is False:
            Step 1: Define the weights for observable neighbors' proposal and unobservable vehicles weighted average acceleration according to the following rules:
            - Rule 1: According to <reasons></reasons> summarized above, define neighbor_proposed_weight_ahead for {self.agent_name}{self.agent_id-1}'s proposal (ahead me) and neighbor_proposed_weight_behind for {self.agent_name}{self.agent_id+1}'s proposal (behind me) in range [0,0.3].
            - Rule 2: According to <reasons></reasons> summarized above, define unobservable_weight_forward for unobservable vehicles {self.unobserve_id_list_ahead} (ahead me) and unobservable_weight_backward for unobservable vehicles {self.unobserve_id_list_behind} (behind me) in range [0,0.3].
            - Rule 3: According to <reasons></reasons> summarized above, define my_weight for my proposal in range [0.3,1].
            - Rule 4: The weights need to satisfy neighbor_proposed_weight_ahead + neighbor_proposed_weight_behind + unobservable_weight_forward + unobservable_weight_backward + my_weight = 1. If not, repeat defining weights.
            Output the insights as following format:
            <insights>
            neighbor_proposed_weight_ahead: <neighbor_proposed_weight_ahead value>
            neighbor_proposed_weight_behind: <neighbor_proposed_weight_behind value>
            unobservable_weight_forward: <unobservable_weight_forward value>
            unobservable_weight_backward: <unobservable_weight_backward value>
            my_weight: <my_weight value>
            </insights>
            """
        query_proposal += f"""
        IMPORTANT:
        - No fabricated data is allowed.
        - When the distance between two vehicles matches the target distance, prioritize collaborative strategies like (ACCELERATION, ACCELERATION), (DECELERATION, DECELERATION), or (MAINTAINING, MAINTAINING) to maintain stable and cooperative behavior.
        - If MY_VELOCITY > VELOCITY_AHEAD, MY_DISTANCE will DECREASE. If MY_VELOCITY < VELOCITY_AHEAD, MY_DISTANCE will INCREASE.
        - If ACCELERATION > 0, MY_VELOCITY will be INCREASED, MY_DISTANCE will be DECREASED.
        - If ACCELERATION < 0, MY_VELOCITY will be DECREASED, MY_DISTANCE will be INCREASED.
        - Before output, double-check whether the reasoning in <proposal></proposal> matches these rules. If not, re-generate the proposal from Step 1.
        - Double-check the correctness of your calculations and ensure that the acceleration in <analysis></analysis>, <proposal></proposal>, and <output></output> tags are consistent and correct.
        - Please ensure that your output always includes both opening (<analysis>, <proposal>, <output>) and closing tags (</analysis>, </proposal>, and </output>), such that every tag pair is properly closed and do not omit any closing tags.
        """

        if '<Reproposal>' in self.over_episode_insights and '</Reproposal>' in self.over_episode_insights:
            query_proposal += f"""
                {extract_from_label(self.over_episode_insights, 'Reproposal')}
            """
        reproposal_query_dict = {
        "role": f"user",
        "name": f"agent{self.agent_id}",
        "content": query_proposal,
        }
        with open(self.conv_path, 'r') as file:
            data = json.load(file)
            data["messages"].append(reproposal_query_dict)
            with open(self.conv_path, 'w') as file:
                json.dump(data, file, indent=4)
        return reproposal_query_dict

    def gen_reproposal_B(self, epi, round, current_rules = [], state=[], last_reward = 0, spatial_dict={}):
        '''
        Transfer state list to a detailed question
        Args:
        '''
        state_dict = state[self.agent_id-1]
        # if int(self.V_FLAG) == 1:
            # query_proposal = read_txt_file(catchup["rewardA"])
            # query_proposal = gen_reward_prompt(self.dt)
        # else:
            # query_proposal = read_txt_file(catchup["reward"])
        # query_proposal += reward_content
        self.update_spatial_insights(epi=epi, state_dict=state_dict, spatial_dict= spatial_dict)
        if self.agent_id == 1:
            query_proposal += f"""[QUERY: Update Proposal]
                Step 2: Based on weights defined between <insights></insights>, weighted the proposed acceleration for each vehicles by following equations:
                updated_acceleration = neighbor_proposed_weight * neighbor_proposed_acceleration + my_weight * my_acceleration + unobservable_weight * {self.agent_name}{self.agent_id}'s backward weighted average acceleration
                Ensure each updated_acceleration is clipped in range [-{ACC}, {ACC}] m/s^2.
                Step 3: After determing the acceleration for each vehicle, 1) invoke: [Function: Next_State Computation] to check whether the next state satisfy velocity <= 25 and distance between all vehicles > 10m.
                If any constraint is violated, go back to Step 1 and adjust your proposals slightly and pick another action for {self.agent_name}{self.agent_id}. Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
                Step 4: Final output with following format:
                <analysis>
                (Show anlaysis and calculation step in detail:
                - List all variables and substitute their values
                - Display all intermediate results
                - Double-check the final answer)
                </analysis>
                <proposal>
                I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id+1}), because ...
                For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                For {self.agent_name}{self.agent_id+1} (my neighbor behind of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
                </proposal>
                <output>
                E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
                "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
                {self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle BEHIND of{self.agent_name}{self.agent_id}>
                }}
                </output>
                """
        elif self.agent_id == VEHICLE_NUM:
            query_proposal += f"""
            [QUERY: Update Proposal]
            Step 2: Based on weights defined between <insights></insights>, weighted the proposed acceleration for each vehicles by following equations:
            updated_acceleration = neighbor_proposed_weight * neighbor_proposed_acceleration + my_weight * my_acceleration + unobservable_weight * {self.agent_name}{self.agent_id}'s forward weighted average acceleration
            Ensure each updated_acceleration is clipped in range [-{ACC}, {ACC}] m/s^2.
            Step 3: After determing the acceleration for each vehicle, 1) invoke: [Function: Next_State Computation] to check whether the next state satisfy velocity <= 25 and distance between all vehicles > 10m.
            If any constraint is violated, go back to Step 1 and adjust your proposals slightly and pick another action for {self.agent_name}{self.agent_id}. Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
            Step 4: Final output with following format:
            <analysis>
            (Show anlaysis and calculation step in detail:
            - List all variables and substitute their values
            - Display all intermediate results
            - Double-check the final answer)
            </analysis>
            <proposal>
            I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id}), because ...
            For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            For {self.agent_name}{self.agent_id-1} (my neighbor ahead of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            </proposal>
            <output>
            E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
            "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
            "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>
            }}</output>
            """
        else:
            query_proposal += f"""
            [QUERY: Update Proposal]
            Step 2: Based on weights defined between <insights></insights>, weighted the proposed acceleration for each vehicles by following equations:
            updated_acceleration = neighbor_proposed_weight_ahead * {self.agent_name}{self.agent_id-1}_proposed_acceleration + neighbor_proposed_weight_behind * {self.agent_name}{self.agent_id+1}_proposed_acceleration + my_weight * my_acceleration + unobservable_weight_forward * {self.agent_name}{self.agent_id-1}'s forward weighted average acceleration + unobservable_weight_backward * {self.agent_name}{self.agent_id+1}'s backward weighted average acceleration
            Ensure each updated_acceleration is clipped in range [-{ACC}, {ACC}] m/s^2.
            Step 3: After determing the acceleration for each vehicle, 1) invoke: [Function: Next_State Computation] to check whether the next state satisfy velocity <= 25 and distance between all vehicles > 10m. If any constraint is violated, go back to Step 1 and adjust your proposals slightly and pick another action for {self.agent_name}{self.agent_id}. Repeat this process up to 10 times. If no fully feasible proposal is found after 10 iterations, stop and return the best proposal you have.
            Step 4: Final output with following format:
            <analysis>
            (Show anlaysis and calculation step in detail:
            - List all variables and substitute their values
            - Display all intermediate results
            - Double-check the final answer)
            </analysis>
            <proposal>
            I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id-1}, {self.agent_name}{self.agent_id}), because ...
            I am {self.agent_name}{self.agent_id}. I choose [collaboration strategy] for ({self.agent_name}{self.agent_id}, {self.agent_name}{self.agent_id + 1}), because ...
            For {self.agent_name}{self.agent_id} (myself), I choose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            For {self.agent_name}{self.agent_id-1} (my neighbor ahead of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            For {self.agent_name}{self.agent_id+1} (my neighbor behind of {self.agent_name}{self.agent_id}), I propose [acceleration value], because I observe ... and decide to ... The urgency is [normal/warning/urgent].
            </proposal>
            <output>
            E{epi}_R{round}_{self.agent_name}{self.agent_id}_proposal = {{
            "{self.agent_name}{self.agent_id}": <proposed acceleration for itself>,
            "{self.agent_name}{self.agent_id-1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>,
            "{self.agent_name}{self.agent_id+1}": <proposed acceleration for vehicle AHEAD of{self.agent_name}{self.agent_id}>
            }}</output>
            """
        query_proposal += f"""
        IMPORTANT:
        - No fabricated data is allowed.
        - When the distance between two vehicles matches the target distance, prioritize collaborative strategies like (ACCELERATION, ACCELERATION), (DECELERATION, DECELERATION), or (MAINTAINING, MAINTAINING) to maintain stable and cooperative behavior.
        - If MY_VELOCITY > VELOCITY_AHEAD, MY_DISTANCE will DECREASE. If MY_VELOCITY < VELOCITY_AHEAD, MY_DISTANCE will INCREASE.
        - If ACCELERATION > 0, MY_VELOCITY will be INCREASED, MY_DISTANCE will be DECREASED.
        - If ACCELERATION < 0, MY_VELOCITY will be DECREASED, MY_DISTANCE will be INCREASED.
        - Before output, double-check whether the reasoning in <proposal></proposal> matches these rules. If not, re-generate the proposal from Step 1.
        - Double-check the correctness of your calculations and ensure that the acceleration in <analysis></analysis>, <proposal></proposal>, and <output></output> tags are consistent and correct.
        - Please ensure that your output always includes both opening (<analysis>, <proposal>, <output>) and closing tags (</analysis>, </proposal>, and </output>), such that every tag pair is properly closed and do not omit any closing tags.
        """

        if '<Reproposal>' in self.over_episode_insights and '</Reproposal>' in self.over_episode_insights:
            query_proposal += f"""
                {extract_from_label(self.over_episode_insights, 'Reproposal')}
            """
        reproposal_query_dict = {
        "role": f"user",
        "name": f"agent{self.agent_id}",
        "content": query_proposal,
        }
        with open(self.conv_path, 'r') as file:
            data = json.load(file)
            data["messages"].append(reproposal_query_dict)
            with open(self.conv_path, 'w') as file:
                json.dump(data, file, indent=4)
        return reproposal_query_dict








    def append_system_prompt_to_SF(self, system_prompt):
        """
        Append system prompt to the JSON file.
        :param system_prompt: The system prompt content.
        """
        try:
            with open(self.SF_record_path, "r", encoding="utf-8") as f:
                conversation_context = json.load(f)
            if "messages" not in conversation_context:
                conversation_context["messages"] = []
            system_message = {
                "role": "system",
                "content": system_prompt
            }
            conversation_context["messages"].append(system_message)
            with open(self.SF_record_path, "w", encoding="utf-8") as f:
                json.dump(conversation_context, f, indent=4, ensure_ascii=False)
            # print(f"Updated conversation context saved to {self.conv_path}")
            # self.state_prompt = system_message
        except FileNotFoundError:
            print(f"Error: File not found at {self.SF_record_path}")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file at {self.SF_record_path}")
        except Exception as e:
            print(f"Unexpected error: type={type(e).__name__}, args={e.args}, msg={e}")

    def append_system_prompt_to_file(self, system_prompt):
        """
        Append system prompt to the JSON file.
        :param system_prompt: The system prompt content.
        """
        try:
            with open(self.conv_path, "r", encoding="utf-8") as f:
                conversation_context = json.load(f)
            if "messages" not in conversation_context:
                conversation_context["messages"] = []
            system_message = {
                "role": "system",
                "content": system_prompt
            }
            conversation_context["messages"].append(system_message)
            with open(self.conv_path, "w", encoding="utf-8") as f:
                json.dump(conversation_context, f, indent=4, ensure_ascii=False)
            # print(f"Updated conversation context saved to {self.conv_path}")
            # self.state_prompt = system_message
        except FileNotFoundError:
            print(f"Error: File not found at {self.conv_path}")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file at {self.conv_path}")
        except Exception as e:
            print(f"Unexpected error: type={type(e).__name__}, args={e.args}, msg={e}")

    def append_other_proposal_to_SF(self, proposal_content):
        """
        Append others' proposal to the JSON file.
        :param proposal: The proposal content.
        :param proposal_id: The agent id of the proposal.
        """
        try:
            with open(self.SF_record_path, "r", encoding="utf-8") as f:
                conversation_context = json.load(f)
            if "messages" not in conversation_context:
                conversation_context["messages"] = []
            user_message = proposal_content
            conversation_context["messages"].append(user_message)
            with open(self.SF_record_path, "w", encoding="utf-8") as f:
                json.dump(conversation_context, f, indent=4, ensure_ascii=False)

        except FileNotFoundError:
            print(f"Error: File not found at {self.SF_record_path}")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file at {self.SF_record_path}")
        except Exception as e:
            print(f"Unexpected error: type={type(e).__name__}, args={e.args}, msg={e}")

    def append_other_proposal_to_file(self, proposal_content, proposal_name):
        """
        Append others' proposal to the JSON file.
        :param proposal: The proposal content.
        :param proposal_id: The agent id of the proposal.
        """
        try:
            with open(self.conv_path, "r", encoding="utf-8") as f:
                conversation_context = json.load(f)
            if "messages" not in conversation_context:
                conversation_context["messages"] = []
            user_message = {
                "role": "assistant",
                "name": f"{proposal_name}",
                "content": f"{proposal_name} proposes: {proposal_content}"
            }
            conversation_context["messages"].append(user_message)
            with open(self.conv_path, "w", encoding="utf-8") as f:
                json.dump(conversation_context, f, indent=4, ensure_ascii=False)
            # print(f"Updated conversation context saved to {self.conv_path}")

        except FileNotFoundError:
            print(f"Error: File not found at {self.conv_path}")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file at {self.conv_path}")
        except Exception as e:
            print(f"Unexpected error: type={type(e).__name__}, args={e.args}, msg={e}")

    def append_self_message_to_file(self, message_content):
        """
        Append others' proposal to the JSON file.
        :param proposal: The proposal content.
        :param proposal_id: The agent id of the proposal.
        """
        try:
            with open(self.conv_path, "r", encoding="utf-8") as f:
                conversation_context = json.load(f)
            if "messages" not in conversation_context:
                conversation_context["messages"] = []
            my_message = {"role": "assistant",
                                "name": f"vehicle{self.agent_id}",
                                "content": message_content}
            conversation_context["messages"].append(my_message)
            with open(self.conv_path, "w", encoding="utf-8") as f:
                json.dump(conversation_context, f, indent=4, ensure_ascii=False)
            # print(f"Updated conversation context saved to {self.conv_path}")

        except FileNotFoundError:
            print(f"Error: File not found at {self.conv_path}")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file at {self.conv_path}")
        except Exception as e:
            print(f"Unexpected error: type={type(e).__name__}, args={e.args}, msg={e}")

    def append_self_message_to_SF(self, message_content):
        """
        Append others' proposal to the JSON file.
        :param proposal: The proposal content.
        :param proposal_id: The agent id of the proposal.
        """
        try:
            with open(self.SF_record_path, "r", encoding="utf-8") as f:
                conversation_context = json.load(f)
            if "messages" not in conversation_context:
                conversation_context["messages"] = []
            my_message = {"role": "assistant",
                                "name": f"vehicle{self.agent_id}",
                                "content": message_content}
            conversation_context["messages"].append(my_message)
            with open(self.SF_record_path, "w", encoding="utf-8") as f:
                json.dump(conversation_context, f, indent=4, ensure_ascii=False)
            # print(f"Updated conversation context saved to {self.conv_path}")

        except FileNotFoundError:
            print(f"Error: File not found at {self.SF_record_path}")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON file at {self.SF_record_path}")
        except Exception as e:
            print(f"Unexpected error: type={type(e).__name__}, args={e.args}, msg={e}")

    def end_negotiation(self, epi, proposal_records, last_reward = 0):
        """
        The final round of negotiation process."
        :param agent_r: The agent.
        :return: The final decision of the agent.
        """

        if epi == 0 or last_reward == 0:
            if int(config["LLM_CONFIG"]["KEEP_TEST_FLAG"]) == 0:
                endDec = f"""
                [QUERY: End Negotiation and Make Final Decision]
                You are the controller of {self.agent_name}{self.agent_id}. Your and your neighbors' proposals changes these rounds can be found in previous message list. This is the final round of negotiation for this episode. Your task is to determine your final decision by completing the following steps:
                Step 1: Review all the accelerations you proposed for yourself (proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id}) during the previous three rounds of negotiation. Integrate your [Spatial Analysis] and the urgency of each proposed acceleration to assign appropriate weights (ensure that the sum of all weights equals 1) and compute a final weighted acceleration (assign higher weights to decisions with greater urgency). Clip this final acceleration to the range [-{ACC}, {ACC}] m/s², and use it as your final decision for {self.agent_name}{self.agent_id} to end this negotiation. (You do not need to provide decisions for your neighbors.)
                Step 2: Calculate the reward_score of your final decision by calling the [Function: Reward_score Calculation]. Predict the next_state by calling [Function: Next_State Computation].
                Step 3: Check your final decision whether satisfy:
                (1) reward_score > -3000.
                (2) next_velocity <= 20 m/s.
                (3) next_distance >= 10 m.
                If any constraint is violated, go back to Step 1 and adjust your weights/proposals slightly and pick another action for {self.agent_name}{self.agent_id}. Repeat this process up to 10 times. If no completely feasible solution can be found after all attempts, return the initial decision from the first round (proposer: vehicle7, receiver: vehicle7).
                Step 4: Output yourself action in <action></action> using the exact format below:
                <analysis>
                (Explain clearly how you reached your final decision, referencing proposal history, reasoning, and reward score.)
                </analysis>
                <action>(yourself action)</action>

                IMPORTANT:
                - Rewards must be computed directly from predicted states, no arbitrary values or manually chosen scores.
                - Your Spatial Analysis is {self.spatial_insights}.
                - Your Temporal Plan is {self.temporal_insights}.
                - Please ensure that your output always includes both opening (<analysis>, <action>) and closing tags (</analysis>, </action>), such that every tag pair is properly closed and do not omit any closing tags.
                """
            else:
                endDec = f"""
                [QUERY: End Negotiation and Make Final Decision]\n You are {self.agent_name}{self.agent_id}.
                Output the acceleration you first proposed in this episode as following format
                <action>(Your_Final_Decision)</action>
                """
        else:
            if int(config["LLM_CONFIG"]["KEEP_TEST_FLAG"]) == 0:
                endDec = f"""
                [QUERY: End Negotiation and Make Final Decision]\n You are the controller of {self.agent_name}{self.agent_id}. Your and your neighbors' proposals changes these rounds can be found in previous message list. This is the final round of negotiation for this episode. Your task is to determine your final decision by completing the following steps:
                Step 1: Review all the accelerations you proposed for yourself (proposer: {self.agent_name}{self.agent_id}, receiver: {self.agent_name}{self.agent_id}) during the previous three rounds of negotiation. Integrate your [Spatial Analysis] and the urgency of each proposed acceleration to assign appropriate weights (ensure that the sum of all weights equals 1) and compute a final weighted acceleration (Assign higher weights to decisions with greater urgency). Clip this final acceleration to the range [-{ACC}, {ACC}] m/s², and use it as your final decision for {self.agent_name}{self.agent_id} to end this negotiation. (You do not need to provide decisions for your neighbors.)
                Step 2: Based on your analysis to weight each proposed acceleration and clip in range [-{ACC}, {ACC}] m/s^2 determined as the final decision for {self.agent_name}{self.agent_id} (you don't need to provide your neighbors' decision) to end this negotiation.
                Step 3: Calculate the reward_score of your final decision by calling the [Function: Reward_score Calculation], and predict the next_state by calling [Function: Next_State Computation]
                Step 4: Check your final decision whether satisfy:
                (1) reward_score > -3000.
                """
                # NOTE: delete (2) reward_score > {last_reward}
                endDec += f"""(2) next_velocity <= 20 m/s.
                (3) next_distance >= 10 m.
                If any constraint is violated, back to Step 2 and adjust your weights/proposals slightly to pick a more conservative action for {self.agent_name}{self.agent_id} (i.e., slightly reduce the magnitude of the acceleration or deceleration). Repeat this process up to 10 times. If no completely feasible solution can be found after all attempts, return the initial decision from the first round (proposer: vehicle7, receiver: vehicle7).
                Step 5: Output yourself action in <action></action> using the exact format below:
                <analysis>
                (Explain clearly how you reached your final decision, referencing proposal history, reasoning, and reward score.)
                </analysis>
                <action>(yourself action)</action>

                IMPORTANT:
                - Rewards must be computed directly from predicted states, no arbitrary values or manually chosen scores.
                - Your Spatial Analysis is {self.spatial_insights}.
                - Your Temporal Plan is {self.temporal_insights}.
                - Please ensure that your output always includes both opening (<analysis>, <action>) and closing tags (</analysis>, </action>), such that every tag pair is properly closed and do not omit any closing tags.
                """
                # Your Spatial Analysis is {self.spatial_insights}.
                # Your Temporal Plan is {self.temporal_insights}.

            else:
                endDec = f"""
                [QUERY: End Negotiation and Make Final Decision]\n You are {self.agent_name}{self.agent_id}.
                Output the acceleration you first proposed in this episode as following format
                <action>(Your_Final_Decision)</action>
                """
        if int(self.V_FLAG) == 1:
            # endDec += read_txt_file(catchup['rewardA'])
            endDec += f"""[Function: Next_State Computation]
            Given the my current state (distance, velocity), ahead vehicle state (distance_ahead, velocity_ahead), my action (acceleration), ahead vehicle action (proposed_current_ahead_acceleration, that is, you do not know the current_ahead_acceleration, you can propose an optimal current_ahead_acceleration for vehicle_ahead). The control interval is {self.dt} second.
            next_velocity = velocity + acceleration × {self.dt}
            ego_displacement = velocity × {self.dt} + 0.5 × acceleration × {self.dt}²
            ahead_displacement = velocity_ahead × {self.dt} + 0.5 × current_ahead_acceleration × {self.dt}²
            next_distance = distance + (ahead_displacement - ego_displacement)
            return next_velocity, next_distance

            [Function: Reward_score Calculation]
            Given the my current state (distance, velocity), ahead vehicle state (distance_ahead, velocity_ahead), my action (acceleration), ahead vehicle action (acceleration_ahead). The control interval is {self.dt} second.
            Calling function [Function: Next_State Computation] to calculate the next_velocity and next_distance.
            distance_reward = - (next_distance - 20)^2
            velocity_reward = - (next_velocity - 15)^2
            acceleration_reward = - 0.1 * acceleration^2
            if next_distance <= 10 meters: return reward_score = -3000,
            else: return reward_score = distance_reward + velocity_reward + acceleration_reward"""
        else:
            endDec += read_txt_file(catchup["reward"])
        self.append_user_question_to_file(user_question = endDec, update=True)
        with open(self.SF_record_path, 'r') as file:
            data = json.load(file)
        message_list = data["messages"]
        message_list.append({"role":"user", "content": endDec})
        format_list = ["action"]

        for attempt in range(1, MAX_LLM_RESPONSE_ATTEMPTS + 1):
            endDec, endDec_content = self.gen_response_with_format(query_content = message_list, format_list = format_list)
            if endDec == False and endDec_content == False:
                print(f"Error: gen_response_with_format returned False in end_negotiation. Attempt {attempt}")
                raise_if_llm_retry_exhausted(
                    attempt,
                    f"CACC agent {self.agent_id} final negotiation response",
                    "response generator returned (False, False)",
                )
                continue
            if not isinstance(endDec_content, str):
                print(f"Warning: endDec_content is not a string, type: {type(endDec_content)}. Attempt {attempt}")
                raise_if_llm_retry_exhausted(
                    attempt,
                    f"CACC agent {self.agent_id} final negotiation response",
                    f"expected string content, got {type(endDec_content).__name__}",
                )
                continue
            if label_exist(message=endDec_content, label='action') or label_exist(message=endDec_content, label='final_decision'):
                return endDec, endDec_content
            raise_if_llm_retry_exhausted(
                attempt,
                f"CACC agent {self.agent_id} final negotiation response",
                "missing <action> or <final_decision> label",
            )
        # endDec, endDec_content = self.gen_response()




    # def end_action(self):
    #     endAct = read_txt_file(task_step['end_action'])
    #     self.append_user_question_to_file(user_question = endAct, update=True)
    #     endAct, endAct_content = self.gen_response()
    #     return endAct, endAct_content


    # 2025-05-26 comment
    '''
    def gen_gradient(self, proposal_records, current_rules):
        proposal_prompt = "You and your neighbors have proposed the following proposals:\n"
        proposal_prompt += proposal_records
        proposal_prompt += "Your last summarized rules are:\n"
        proposal_prompt += current_rules
        proposal_prompt += read_txt_file(task_step['update_rules'])
        self.append_user_question_to_file(user_question = proposal_prompt)
        response, response_content = self.gen_response()
        return response, response_content\
    '''




    def message_list_query(self, message_list, default_client = azure_client1, default_model = OPENAI_MODEL1):
        max_retries = 10
        retry_delay = 2
        response = None

        for attempt in range(1, max_retries + 1):
            try:
                if FLAG == 'GPT':
                    response = default_client.chat.completions.create(
                        model= default_model,
                        messages=message_list,
                        temperature=self.temp,
                        # top_p = topp
                        # max_tokens=50
                    )
                    if response is None:
                        print("Error: response is None after all retries")
                        # import pdb; pdb.set_trace()
                        continue

                    if isinstance(response, str):
                        print(f"Warning: response is a string instead of object: {response[:100]}...")
                        # import pdb; pdb.set_trace()
                        continue

                    if not hasattr(response, 'choices') or not response.choices:
                        print(f"Error: response does not have choices attribute or choices is empty. Response type: {type(response)}")
                        # import pdb; pdb.set_trace()
                        continue
                    break
                elif FLAG == 'Ali':
                    # output_content, thinking_content = self.client_run(messages=message_list)
                    response = default_client.chat.completions.create(
                        model= default_model,
                        messages=message_list,
                            temperature=self.temp,
                        # top_p = topp
                        # max_tokens=50
                    )

                    if response is None:
                        print("Error: response is None after all retries")
                        pass
                        continue

                    if isinstance(response, str):
                        print(f"Warning: response is a string instead of object: {response[:100]}...")
                        pass
                        continue

                    if not hasattr(response, 'choices') or not response.choices:
                        print(f"Error: response does not have choices attribute or choices is empty. Response type: {type(response)}")
                        pass
                        continue
                    break
                elif FLAG == 'Llama':
                    response = azure_client1.chat.completions.create(
                        model= default_model,
                        messages=message_list,
                        temperature=self.temp,
                        # top_p = topp
                        # max_tokens=50
                    )
                    if response is not None and hasattr(response, 'choices') and response.choices:
                        break
                    print("Warning: Llama response empty or no choices, retrying...")
                    continue
            except RateLimitError as e:
                if attempt < max_retries:
                    print(f"[Retry {attempt}/{max_retries}] Rate limit error, retrying after {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Exceeded maximum retries ({max_retries}). Raising error.")
                    raise
            # Handle content-policy request errors.
            except openai.BadRequestError as e:
                return False, False
            except openai.APIStatusError as e:
                if e.status_code >= 500:
                    # Server error (5xx), retry with exponential backoff
                    if attempt < max_retries:
                        backoff_delay = min(2 ** (attempt - 1), 16)
                        print(f"[Retry {attempt}/{max_retries}] Server error (HTTP {e.status_code}), retrying after {backoff_delay}s...")
                        time.sleep(backoff_delay)
                    else:
                        print(f"Server error after {max_retries} retries (HTTP {e.status_code}): {e.message}")
                        raise
                else:
                    # Client error (4xx), don't retry
                    print(f"Client error (HTTP {e.status_code}): {e.message}")
                    raise
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    print(f"[Retry {attempt}/{max_retries}] API response not valid JSON (e.g. server error/truncated), retrying after {retry_delay}s: {e}")
                    time.sleep(retry_delay)
                else:
                    print(f"Exceeded max retries ({max_retries}) after JSONDecodeError. Last error: {e}")
                    raise
            except Exception as e:
                print(f"Unexpected error: type={type(e).__name__}, args={e.args}, msg={e}")
                raise

        if FLAG == 'GPT':
            # return response, response.messages[-1]["content"]
            return response, response.choices[0].message.content
        elif FLAG == 'Ali':
            # return output_content + thinking_content, output_content
            return response, response.choices[0].message.content
        elif FLAG == 'Llama':
            return response, response.choices[0].message.content
        else:
            print(f"Warning: Unknown FLAG value: {FLAG}")
            return False, False



    def initial_over_episode_insights(self, epi, round, state=[]):
        '''
        Transfer state list to a detailed question
        '''
        # self.append_system_prompt_to_file(system_prompt=read_txt_file(catchup['reward']))
        with open(self.conv_path, "r", encoding="utf-8") as f:
            conversation_context = json.load(f)
        initial_over_episode_query = conversation_context["messages"]

        state_dict = state[self.agent_id-1]
        v_lead = state_dict[f"{self.agent_name}{self.agent_id}"][2]
        # 0. state_info
        if self.agent_id == 1:
            state_info = f'''[{self.agent_name}{str(self.agent_id)} Observable State in Episode_{epi}]
            {str(state_dict)}
            The dict represents {{vehicle0 is the virtual target vehicle with constant velocity {state_dict[f"{self.agent_name}{self.agent_id-1}"][1]} m/s.
            {self.agent_name}{self.agent_id} is driving towards virtual target vehicle and the distance from {self.agent_name}{self.agent_id} to the virtual target vehicle is {state_dict[f"{self.agent_name}{self.agent_id}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id}"][3]} m/s^2.
            {self.agent_name}{self.agent_id+1} is driving towards {self.agent_name}{self.agent_id} and the distance from {self.agent_name}{self.agent_id+1} to {self.agent_name}{self.agent_id} is {state_dict[f"{self.agent_name}{self.agent_id+1}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id+1}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id+1}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id+1}"][2]} m/s, its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id+1}"][3]} m/s^2}}.'''
        elif self.agent_id == VEHICLE_NUM:
            state_info = f'''[{self.agent_name}{str(self.agent_id)} Observable State in Episode_{epi}]
            {str(state_dict)}
            The dict represents {{\n{self.agent_name}{self.agent_id-1} is driving towards {self.agent_name}{self.agent_id-2} and the distance from {self.agent_name}{self.agent_id-1} to the {self.agent_name}{self.agent_id-2} is {state_dict[f"{self.agent_name}{self.agent_id-1}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id-1}\'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id-1}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id-1}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id-1}"][3]} m/s^2.
            {self.agent_name}{self.agent_id} is driving towards {self.agent_name}{self.agent_id-1} and the distance from {self.agent_name}{self.agent_id} to {self.agent_name}{self.agent_id-1} is {state_dict[f"{self.agent_name}{self.agent_id}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id}\'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id}"][3]} m/s^2}}.
            '''
        elif self.agent_id == 2:
            state_info = f'''[{self.agent_name}{str(self.agent_id)} Observable State in Episode_{epi}]
            {str(state_dict)}
            The dict represents {{\n{self.agent_name}{self.agent_id-1} is driving towards virtual target vehicle and the distance from {self.agent_name}{self.agent_id-1} to the virtual target vehicle is {state_dict[f"{self.agent_name}{self.agent_id-1}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id-1}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id-1}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id-1}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id-1}"][3]} m/s^2.
            {self.agent_name}{self.agent_id} is driving towards {self.agent_name}{self.agent_id-1} and the distance from {self.agent_name}{self.agent_id} to {self.agent_name}{self.agent_id-1} is {state_dict[f"{self.agent_name}{self.agent_id}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id}"][3]} m/s^2.
            {self.agent_name}{self.agent_id+1} is driving towards {self.agent_name}{self.agent_id} and the distance from {self.agent_name}{self.agent_id+1} to {self.agent_name}{self.agent_id} is {state_dict[f"{self.agent_name}{self.agent_id+1}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id+1}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id+1}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id+1}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id+1}"][3]} m/s^2}}.'''
        else:
            state_info = f'''[{self.agent_name}{str(self.agent_id)} Observable State in Episode_{epi}]
            {str(state_dict)}
            The dict represents {{\n{self.agent_name}{self.agent_id-1} is driving towards {self.agent_name}{self.agent_id-2} and the distance from {self.agent_name}{self.agent_id-1} to {self.agent_name}{self.agent_id-2} is {state_dict[f"{self.agent_name}{self.agent_id-1}"][0]} m; its target distance is 20m; {self.agent_name}{self.agent_id-1}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id-1}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id-1}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id-1}"][3]} m/s^2.
            {self.agent_name}{self.agent_id} is driving towards {self.agent_name}{self.agent_id-1} and the distance from {self.agent_name}{self.agent_id} to {self.agent_name}{self.agent_id-1} is {state_dict[f"{self.agent_name}{self.agent_id}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id}\'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id}"][3]} m/s^2.
            {self.agent_name}{self.agent_id+1} is driving towards {self.agent_name}{self.agent_id} and the distance from {self.agent_name}{self.agent_id+1} to {self.agent_name}{self.agent_id} is {state_dict[f"{self.agent_name}{self.agent_id+1}"][0]} m; its target distance is 20 m; {self.agent_name}{self.agent_id+1}'s current velocity is {state_dict[f"{self.agent_name}{self.agent_id+1}"][1]} m/s; its target velocity is {state_dict[f"{self.agent_name}{self.agent_id+1}"][2]} m/s; its last acceleration is {state_dict[f"{self.agent_name}{self.agent_id+1}"][3]} m/s^2}}.'''
        self.state_prompt = {"role": "system", "content": state_info}
        initial_over_episode_query.append(self.state_prompt)
        # 1. info
        query_over_episode = f"""" [QUERY: Generate Over-Episode Strategy]
        You are the strategist for {self.agent_name}{self.agent_id}."""
        if self.agent_id == 1:
            query_over_episode += f"""{self.agent_name}{self.agent_id}'s neighbor is {self.agent_name}{self.agent_id + 1} (behind of {self.agent_name}{self.agent_id}). """
        elif self.agent_id == VEHICLE_NUM:
            query_over_episode += f""" {self.agent_name}{self.agent_id}'s neighbor is {self.agent_name}{self.agent_id - 1} (ahead of {self.agent_name}{self.agent_id}). """
        else:
            query_over_episode += f""" {self.agent_name}{self.agent_id}'s neighbor are {self.agent_name}{self.agent_id - 1} (ahead of {self.agent_name}{self.agent_id}) and {self.agent_name}{self.agent_id + 1} (behind {self.agent_name}{self.agent_id}). Based on the [Objective Instruction] and [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}], analyze the spatial and temporal insights as follows:"""

        # # Unobservable states
        if self.unobserve_id_list_ahead == [] and self.unobserve_id_list_behind != []:
            self.spatial_insights = f"""Unobservable vehicles {self.unobserve_id_list_behind} are behind {self.agent_name}{self.agent_id}'s observable vehicles {self.observe_id_list}, with more than 2 hops communication distance. {self.agent_name}{self.agent_id} cannot directly obtain the states and actions of these unobservable vehicles. This is episode {epi}. Your Spatial Insights are summarized as follows:
            <spatial-insights>
            - {self.agent_name}{self.agent_id}'s forward features: weighted average acceleration: {self.forward_WAA}; average distance: {self.forward_WAH}; average velocity: {self.forward_WAV}.
            - {self.agent_name}{self.agent_id}'s backward features: weighted average acceleration: {self.backward_WAA}; average distance: {self.backward_WAH}; avearge velocity: {self.backward_WAV}.
            - {self.agent_name}{self.agent_id}'s forward variances: weighted acceleration variance: {self.forward_WV}; distance variance: {self.forward_WVH}; velocity variance {self.forward_WVV}.
            - {self.agent_name}{self.agent_id}'s backward variances: weighted acceleration variance: {self.backward_WV}; distance variance: {self.backward_WVH}; velocity variance: {self.backward_WVV}.
            </spatial-insights>"""
        elif self.unobserve_id_list_ahead != [] and self.unobserve_id_list_behind == []:
            self.spatial_insights = f"""Unobservable vehicles {self.unobserve_id_list_ahead} are ahead {self.agent_name}{self.agent_id}'s observable vehicles {self.observe_id_list}, with more than 2 hops communication distance. {self.agent_name}{self.agent_id} cannot directly obtain the states and actions of these unobservable vehicles. This is episode {epi}. Your Spatial Insights are summarized as follows:
            <spatial-insights>
            <spatial-insights>
            - {self.agent_name}{self.agent_id}'s forward features: weighted average acceleration: {self.forward_WAA}; average distance: {self.forward_WAH}; average velocity: {self.forward_WAV}.
            - {self.agent_name}{self.agent_id}'s backward features: weighted average acceleration: {self.backward_WAA}; average distance: {self.backward_WAH}; avearge velocity: {self.backward_WAV}.
            - {self.agent_name}{self.agent_id}'s forward variances: weighted acceleration variance: {self.forward_WV}; distance variance: {self.forward_WVH}; velocity variance {self.forward_WVV}.
            - {self.agent_name}{self.agent_id}'s backward variances: weighted acceleration variance: {self.backward_WV}; distance variance: {self.backward_WVH}; velocity variance: {self.backward_WVV}.
            </spatial-insights>
            </spatial-insights>"""
        else:
            self.spatial_insights = f"""Unobservable vehicles {self.unobserve_id_list_ahead} are ahead {self.agent_name}{self.agent_id}'s observable vehicles {self.observe_id_list} and unobservable vehicles {self.unobserve_id_list_behind} are behind {self.agent_name}{self.agent_id}'s observable vehicles {self.observe_id_list}, with more than 2 hops communication distance. {self.agent_name}{self.agent_id} cannot directly obtain the states and actions of these unobservable vehicles. This is episode {epi}. Your Spatial Insights are summarized as follows:
            <spatial-insights>
            - {self.agent_name}{self.agent_id}'s forward features: weighted average acceleration: {self.forward_WAA}; average distance: {self.forward_WAH}; average velocity: {self.forward_WAV}.
            - {self.agent_name}{self.agent_id}'s backward features: weighted average acceleration: {self.backward_WAA}; average distance: {self.backward_WAH}; avearge velocity: {self.backward_WAV}.
            - {self.agent_name}{self.agent_id}'s forward variances: weighted acceleration variance: {self.forward_WV}; distance variance: {self.forward_WVH}; velocity variance {self.forward_WVV}.
            - {self.agent_name}{self.agent_id}'s backward variances: weighted acceleration variance: {self.backward_WV}; distance variance: {self.backward_WVH}; velocity variance: {self.backward_WVV}.
            </spatial-insights>"""
        if catchup_flag:
            query_over_episode += f""" Step 1. Forward-Looking Temporal Phased Strategy: Based on the current observed state, design a high-level, multi-step strategy for {self.agent_name}{self.agent_id} to reach its target distance = 20m and target velocity = {v_lead} m/s simultaneously. For each phase, specify the required strategy (e.g., aggressive/slight/collaborative acceleration or deceleration, maintaining current states, and so on). Output the temporal analysis within <temporal-analysis> tags as follows:
            <temporal-analysis>
            {self.agent_name}{self.agent_id}'s current state is ... the target state is ...
            (First, it needs to [local adjustment/collaborative action] until [state condition or milestone]. Detailed reason: ...
            Second, it needs to ... ...
            Finally, it can maintain its acceleration = 0 to keep the target velocity and distance as desired values.)
            </temporal-analysis>
            For example:
            <temporal-analysis>
            vehicle2's current state is (30m, 15.0m/s). the final stable state is (20m, 15m/s). It hasn't achieved the target distance.
            I observe that vehicle1 is accelerating, and I need to collaborative accelerate with vehicle1 first.
            First, vehicle2 needs to accelerate collaborting slightly with vehicle1 until vehicle2's distance to vehilce1 is reduced to 20m.
            Second, vehicle2 needs to decelerate slightly until the velocity is reduced to 15m/s.
            Finally, vehicle2 can maintain its acceleration = 0 to keep the target velocity and distance as desired values.
            </temporal-analysis>

            IMPORTANT:
            - If MY_VELOCITY > VELOCITY_AHEAD, MY_DISTANCE will be DECREASED. If MY_VELOCITY < VELOCITY_AHEAD, MY_DISTANCE will be INCREASED.
            - If ACCELERATION > 0, MY_VELOCITY will be INCREASED, MY_DISTANCE will be DECREASED.
            - If ACCELERATION < 0, MY_VELOCITY will be DECREASED, MY_DISTANCE will be INCREASED.
            - Please ensure that your output always includes both opening (<temporal-analysis>) and closing tags (</temporal-analysis>), such that every tag pair is properly closed and do not omit any closing tags.
            """
        else:
            query_over_episode += f"""
            Step 1: Forward-Looking Temporal Phased Strategy: Based on [{self.agent_name}{self.agent_id} Observable State in Episode_{epi}], design a high-level, multi-step strategy for {self.agent_name}{self.agent_id} to reach its target distance = 20m and target velocity = {v_lead} simultaneously. For each phase, specify the required strategy (e.g., aggressive/slight/collaborative acceleration or deceleration, maintaining current states, and so on). Output the temporal analysis within <temporal-analysis> tags as follows:
            <temporal-analysis>
            {self.agent_name}{self.agent_id}'s current state is ... the target state is ...
            (First, it needs to [local adjustment/collaborative action] until [state condition or milestone]. Detailed reason: ...
            Second, it needs to ...
            Finally, it can maintain its acceleration = 0 to keep the target velocity and distance as desired values.)
            </temporal-analysis>
            For example:
            <temporal-analysis>
            vehicle2's current state is (23m, 16.0m/s). the final stable state is (20m, 17m/s). It hasn't achieved the target distance.
            I observe that vehicle1 is accelerating, and I need to collaborative accelerate with vehicle1 first.
            First, vehicle2 needs to accelerate collaborting slightly with vehicle1 until vehicle2's distance to vehilce1 is reduced to 20m.
            Second, vehicle2 needs to decelerate slightly until the velocity is reduced to 17m/s.
            Finally, vehicle2 can maintain its acceleration = 0 to keep the target velocity and distance as desired values.
            </temporal-analysis>

            IMPORTANT:
            - If MY_VELOCITY > VELOCITY_AHEAD, MY_DISTANCE will be DECREASED. If MY_VELOCITY < VELOCITY_AHEAD, MY_DISTANCE will be INCREASED.
            - If ACCELERATION > 0, MY_VELOCITY will be INCREASED, MY_DISTANCE will be DECREASED.
            - If ACCELERATION < 0, MY_VELOCITY will be DECREASED, MY_DISTANCE will be INCREASED.
            - Please ensure that your output always includes both opening (<temporal-analysis>) and closing tags (</temporal-analysis>), such that every tag pair is properly closed and do not omit any closing tags.
            """
        over_episode_query_dict = {
            "role": "user",
            "content": query_over_episode
        }
        self.append_user_question_to_file(user_question = query_over_episode)
        initial_over_episode_query.append(over_episode_query_dict)
        # if '<ProposalGeneration>' in self.over_episode_insights and '</ProposalGeneration>' in self.over_episode_insights:
        #     query_proposal += f"""
        #         {extract_from_label(self.over_episode_insights, 'ProposalGeneration')}
        #     """
        # initial_over_episode_query = back_info + self.state_prompt
        format_list = ['temporal-analysis']
        return initial_over_episode_query, format_list





    def gen_response_with_format(self, query_content, format_list):
        normalized_format_list = []
        for fmt in format_list:
            if isinstance(fmt, str):
                fmt = fmt.strip()
                if fmt and fmt not in normalized_format_list:
                    normalized_format_list.append(fmt)

        def _format_satisfied(content):
            label_set = set(normalized_format_list)
            if len(normalized_format_list) == 0:
                return True
            if label_set == {"action", "final_decision"}:
                return label_exist(message=content, label='action') or label_exist(message=content, label='final_decision')
            if {"deal", "action", "reasons"}.issubset(label_set):
                has_deal = label_exist(message=content, label='deal')
                has_action = label_exist(message=content, label='action')
                has_reasons = label_exist(message=content, label='reasons')
                return has_deal and (has_action or has_reasons)
            for format_string in normalized_format_list:
                if not check_format_label(content, format_string):
                    return False
            return True

        def _print_missing_labels(content, retry_count):
            print(f"{self.agent_id} error, retry {retry_count}")
            for format_string in normalized_format_list:
                if f"<{format_string}>" not in content:
                    print(f"<{format_string}> not in response_content")
                if f"</{format_string}>" not in content:
                    print(f"</{format_string}> not in response_content")

        # xgrammar_flag = 1: use schema-constrained generation first (currently enabled for GPT path)
        if self.xgrammar_flag == 1 and FLAG == 'GPT' and len(normalized_format_list) > 0:
            json_schema = build_json_schema_from_format_list(normalized_format_list)
            if json_schema is not None:
                for attempt in range(1, MAX_LLM_RESPONSE_ATTEMPTS + 1):
                    try:
                        response_full, _, parsed_json = generate_structured_json_openai_response(
                            default_client=azure_client1,
                            default_model=OPENAI_MODEL1,
                            messages=query_content,
                            json_schema=json_schema,
                            temperature=self.temp,
                        )
                        response_content = render_tagged_response_from_json(parsed_json, normalized_format_list)
                    except Exception as e:
                        print(f"{self.agent_id} structured generation error, retry {attempt}, err={e}")
                        raise_if_llm_retry_exhausted(
                            attempt,
                            f"CACC agent {self.agent_id} structured response generation",
                            f"{type(e).__name__}: {e}",
                        )
                        continue

                    if _format_satisfied(response_content):
                        return response_full, response_content
                    print(f"{self.agent_id} structured format check failed, retry {attempt}")
                    _print_missing_labels(response_content, attempt)
                    raise_if_llm_retry_exhausted(
                        attempt,
                        f"CACC agent {self.agent_id} structured response formatting",
                        f"missing required labels {normalized_format_list}",
                    )

        # legacy path: prompt format + retry
        for attempt in range(1, MAX_LLM_RESPONSE_ATTEMPTS + 1):
            response_full, response_content = self.message_list_query(message_list = query_content)
            # import pdb; pdb.set_trace()
            if response_full == False and response_content == False:
                return False, False
            if _format_satisfied(response_content):
                return response_full, response_content
            # import pdb; pdb.set_trace()
            _print_missing_labels(response_content, attempt)
            raise_if_llm_retry_exhausted(
                attempt,
                f"CACC agent {self.agent_id} response formatting",
                f"missing required labels {normalized_format_list}",
            )

    def update_spatial_insights(self, epi, state_dict, spatial_dict):
        if self.agent_id == 1:
            updated_forward_WAA = 0
            updated_forward_WV = 0
            updated_forward_WAH = 20
            updated_forward_WVH = 0
            updated_forward_WAV = 15
            updated_forward_WVV = 0
            forward_old_count = self.agent_id - 1
            backward_old_count = VEHICLE_NUM - self.agent_id-1
            updated_backward_WAA, updated_backward_WV = welford_ew_update(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id+1}']['backward_WAA'], var_prev = spatial_dict[f'{self.agent_name}{self.agent_id+1}']['backward_WV'], x_new = state_dict[f"{self.agent_name}{self.agent_id+1}"][3], weight = 0.7)
            updated_backward_WAH, updated_backward_WVH = welford_update_from_stats(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id+1}']['backward_WAH'], var_prev = spatial_dict[f'{self.agent_name}{self.agent_id+1}']['backward_WVH'], count_prev = backward_old_count, x_new = state_dict[f"{self.agent_name}{self.agent_id+1}"][0])
            updated_backward_WAV, updated_backward_WVV = welford_update_from_stats(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id+1}']['backward_WAV'],
            var_prev = spatial_dict[f'{self.agent_name}{self.agent_id+1}']['backward_WVV'], count_prev = backward_old_count, x_new = state_dict[f"{self.agent_name}{self.agent_id+1}"][1])

            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WAA'] = updated_forward_WAA
            self.forward_WAA = updated_forward_WAA
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WV'] = updated_forward_WV
            self.forward_WV = updated_forward_WV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WAA'] = updated_backward_WAA
            self.backward_WAA = updated_backward_WAA
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WV'] = updated_backward_WV
            self.backward_WV = updated_backward_WV
            # distance
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WAH'] = updated_forward_WAH
            self.forward_WAH = updated_forward_WAH
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WVH'] = updated_forward_WVH
            self.forward_WVH = updated_forward_WVH
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WAH'] = updated_backward_WAH
            self.backward_WAH = updated_backward_WAH
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WVH'] = updated_backward_WVH
            self.backward_WVH = updated_backward_WVH
            # velocity
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WAV'] = updated_forward_WAV
            self.forward_WAV = updated_forward_WAV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WVV'] = updated_forward_WVV
            self.forward_WVV = updated_forward_WVV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WAV'] = updated_backward_WAV
            self.backward_WAV = updated_backward_WAV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WVV'] = updated_backward_WVV
            self.backward_WVV = updated_backward_WVV

            self.spatial_insights = f"""Unobservable vehicles {self.unobserve_id_list_behind} are behind {self.agent_name}{self.agent_id}'s observable vehicles {self.observe_id_list}, with more than 2 hops communication distance. {self.agent_name}{self.agent_id} cannot directly obtain the states and actions of these unobservable vehicles. This is episode {epi}. Your Spatial Insights are summarized as follows:
            <spatial-insights>
            - {self.agent_name}{self.agent_id}'s forward features: weighted average acceleration: {self.forward_WAA}; average distance: {self.forward_WAH}; average velocity: {self.forward_WAV}.
            - {self.agent_name}{self.agent_id}'s backward features: weighted average acceleration: {self.backward_WAA}; average distance: {self.backward_WAH}; avearge velocity: {self.backward_WAV}.
            - {self.agent_name}{self.agent_id}'s forward variances: weighted acceleration variance: {self.forward_WV}; distance variance: {self.forward_WVH}; velocity variance {self.forward_WVV}.
            - {self.agent_name}{self.agent_id}'s backward variances: weighted acceleration variance: {self.backward_WV}; distance variance: {self.backward_WVH}; velocity variance: {self.backward_WVV}.
            </spatial-insights>"""

        elif self.agent_id == VEHICLE_NUM:
            forward_old_count = self.agent_id - 1
            # backward_old_count = 0  # Last vehicle has no vehicles behind
            backward_old_count = VEHICLE_NUM - self.agent_id-1
            updated_forward_WAA, updated_forward_WV = welford_ew_update(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WAA'], var_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WV'], x_new = state_dict[f"{self.agent_name}{self.agent_id-1}"][3], weight = 0.7)

            updated_forward_WAH, updated_forward_WVH = welford_update_from_stats(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WAH'], var_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WVH'], count_prev = forward_old_count, x_new = state_dict[f"{self.agent_name}{self.agent_id-1}"][0])

            updated_forward_WAV, updated_forward_WVV = welford_update_from_stats(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WAV'], var_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WVV'], count_prev = forward_old_count, x_new = state_dict[f"{self.agent_name}{self.agent_id-1}"][1])

            updated_backward_WAA = 0 # nan
            updated_backward_WV = 0 # nan
            updated_backward_WAH = 0 # nan
            updated_backward_WVH = 0 # nan
            updated_backward_WAV = 0 # nan
            updated_backward_WVV = 0 # nan

            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WAA'] = updated_forward_WAA
            self.forward_WAA = updated_forward_WAA
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WV'] = updated_forward_WV
            self.forward_WV = updated_forward_WV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WAA'] = updated_backward_WAA
            self.backward_WAA = updated_backward_WAA
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WV'] = updated_backward_WV
            self.backward_WV = updated_backward_WV
            # distance
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WAH'] = updated_forward_WAH
            self.forward_WAH = updated_forward_WAH
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WVH'] = updated_forward_WVH
            self.forward_WVH = updated_forward_WVH
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WAH'] = updated_backward_WAH
            self.backward_WAH = updated_backward_WAH
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WVH'] = updated_backward_WVH
            self.backward_WVH = updated_backward_WVH
            # velocity
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WAV'] = updated_forward_WAV
            self.forward_WAV = updated_forward_WAV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WVV'] = updated_forward_WVV
            self.forward_WVV = updated_forward_WVV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WAV'] = updated_backward_WAV
            self.backward_WAV = updated_backward_WAV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WVV'] = updated_backward_WVV
            self.backward_WVV = updated_backward_WVV
            self.spatial_insights = f"""Unobservable vehicles {self.unobserve_id_list_ahead} are ahead {self.agent_name}{self.agent_id}'s observable vehicles {self.observe_id_list}, with more than 2 hops communication distance. {self.agent_name}{self.agent_id} cannot directly obtain the states and actions of these unobservable vehicles. This is episode {epi}. Your Spatial Insights are summarized as follows:
            <spatial-insights>
            - {self.agent_name}{self.agent_id}'s forward features: weighted average acceleration: {self.forward_WAA}; average distance: {self.forward_WAH}; average velocity: {self.forward_WAV}.
            - {self.agent_name}{self.agent_id}'s backward features: weighted average acceleration: {self.backward_WAA}; average distance: {self.backward_WAH}; avearge velocity: {self.backward_WAV}.
            - {self.agent_name}{self.agent_id}'s forward variances: weighted acceleration variance: {self.forward_WV}; distance variance: {self.forward_WVH}; velocity variance {self.forward_WVV}.
            - {self.agent_name}{self.agent_id}'s backward variances: weighted acceleration variance: {self.backward_WV}; distance variance: {self.backward_WVH}; velocity variance: {self.backward_WVV}.
            </spatial-insights>"""
        elif self.agent_id == 7:
            forward_old_count = self.agent_id - 1
            backward_old_count = VEHICLE_NUM - self.agent_id-1

            updated_forward_WAA, updated_forward_WV = welford_ew_update(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WAA'], var_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WV'], x_new = state_dict[f"{self.agent_name}{self.agent_id-1}"][3], weight = 0.7)

            updated_forward_WAH, updated_forward_WVH = welford_update_from_stats(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WAH'], var_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WVH'], count_prev = forward_old_count, x_new = state_dict[f"{self.agent_name}{self.agent_id-1}"][0])

            updated_forward_WAV, updated_forward_WVV = welford_update_from_stats(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WAV'], var_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WVV'], count_prev = forward_old_count, x_new = state_dict[f"{self.agent_name}{self.agent_id-1}"][1])

            updated_backward_WAA = state_dict[f"{self.agent_name}{self.agent_id+1}"][3]
            updated_backward_WV = 0
            updated_backward_WAH = state_dict[f"{self.agent_name}{self.agent_id+1}"][0]
            updated_backward_WVH = 0
            updated_backward_WAV = state_dict[f"{self.agent_name}{self.agent_id+1}"][1]
            updated_backward_WVV = 0

            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WAA'] = updated_forward_WAA
            self.forward_WAA = updated_forward_WAA
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WV'] = updated_forward_WV
            self.forward_WV = updated_forward_WV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WAA'] = updated_backward_WAA
            self.backward_WAA = updated_backward_WAA
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WV'] = updated_backward_WV
            self.backward_WV = updated_backward_WV
            # distance
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WAH'] = updated_forward_WAH
            self.forward_WAH = updated_forward_WAH
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WVH'] = updated_forward_WVH
            self.forward_WVH = updated_forward_WVH
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WAH'] = updated_backward_WAH
            self.backward_WAH = updated_backward_WAH
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WVH'] = updated_backward_WVH
            self.backward_WVH = updated_backward_WVH
            # velocity
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WAV'] = updated_forward_WAV
            self.forward_WAV = updated_forward_WAV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WVV'] = updated_forward_WVV
            self.forward_WVV = updated_forward_WVV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WAV'] = updated_backward_WAV
            self.backward_WAV = updated_backward_WAV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WVV'] = updated_backward_WVV
            self.backward_WVV = updated_backward_WVV
            self.spatial_insights = f"""Unobservable vehicles {self.unobserve_id_list_ahead} are ahead {self.agent_name}{self.agent_id}'s observable vehicles {self.observe_id_list}, with more than 2 hops communication distance. {self.agent_name}{self.agent_id} cannot directly obtain the states and actions of these unobservable vehicles. This is episode {epi}. Your Spatial Insights are summarized as follows:
            <spatial-insights>
            - {self.agent_name}{self.agent_id}'s forward features: weighted average acceleration: {self.forward_WAA}; average distance: {self.forward_WAH}; average velocity: {self.forward_WAV}.
            - {self.agent_name}{self.agent_id}'s backward features: weighted average acceleration: {self.backward_WAA}; average distance: {self.backward_WAH}; avearge velocity: {self.backward_WAV}.
            - {self.agent_name}{self.agent_id}'s forward variances: weighted acceleration variance: {self.forward_WV}; distance variance: {self.forward_WVH}; velocity variance {self.forward_WVV}.
            - {self.agent_name}{self.agent_id}'s backward variances: weighted acceleration variance: {self.backward_WV}; distance variance: {self.backward_WVH}; velocity variance: {self.backward_WVV}.
            </spatial-insights>"""

        else:
            forward_old_count = self.agent_id - 1
            backward_old_count = VEHICLE_NUM - self.agent_id-1
            # forward_acceleration
            updated_forward_WAA, updated_forward_WV = welford_ew_update(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WAA'], var_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WV'], x_new = state_dict[f"{self.agent_name}{self.agent_id-1}"][3], weight = 0.7)
            # forward_headway
            updated_forward_WAH, updated_forward_WVH = welford_update_from_stats(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WAH'], var_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WVH'], count_prev = forward_old_count, x_new = state_dict[f"{self.agent_name}{self.agent_id-1}"][0])
            # forward_velocity
            updated_forward_WAV, updated_forward_WVV = welford_update_from_stats(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WAV'], var_prev = spatial_dict[f'{self.agent_name}{self.agent_id-1}']['forward_WVV'], count_prev = forward_old_count, x_new = state_dict[f"{self.agent_name}{self.agent_id-1}"][1])
            # backward_acceleration
            updated_backward_WAA, updated_backward_WV = welford_ew_update(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id+1}']['backward_WAA'],
            var_prev = spatial_dict[f'{self.agent_name}{self.agent_id+1}']['backward_WV'], x_new = state_dict[f"{self.agent_name}{self.agent_id+1}"][3], weight = 0.7)
            # backward_headway
            updated_backward_WAH, updated_backward_WVH = welford_update_from_stats(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id+1}']['backward_WAH'],
            var_prev = spatial_dict[f'{self.agent_name}{self.agent_id+1}']['backward_WVH'], count_prev = backward_old_count, x_new = state_dict[f"{self.agent_name}{self.agent_id+1}"][0])
            # backward_velocity
            updated_backward_WAV, updated_backward_WVV = welford_update_from_stats(mean_prev = spatial_dict[f'{self.agent_name}{self.agent_id+1}']['backward_WAV'],
            var_prev = spatial_dict[f'{self.agent_name}{self.agent_id+1}']['backward_WVV'], count_prev = backward_old_count, x_new = state_dict[f"{self.agent_name}{self.agent_id+1}"][1])

            # acceleration
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WAA'] = updated_forward_WAA
            self.forward_WAA = updated_forward_WAA
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WV'] = updated_forward_WV
            self.forward_WV = updated_forward_WV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WAA'] = updated_backward_WAA
            self.backward_WAA = updated_backward_WAA
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WV'] = updated_backward_WV
            self.backward_WV = updated_backward_WV
            # distance
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WAH'] = updated_forward_WAH
            self.forward_WAH = updated_forward_WAH
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WVH'] = updated_forward_WVH
            self.forward_WVH = updated_forward_WVH
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WAH'] = updated_backward_WAH
            self.backward_WAH = updated_backward_WAH
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WVH'] = updated_backward_WVH
            self.backward_WVH = updated_backward_WVH
            # velocity
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WAV'] = updated_forward_WAV
            self.forward_WAV = updated_forward_WAV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['forward_WVV'] = updated_forward_WVV
            self.forward_WVV = updated_forward_WVV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WAV'] = updated_backward_WAV
            self.backward_WAV = updated_backward_WAV
            spatial_dict[f'{self.agent_name}{self.agent_id}']['backward_WVV'] = updated_backward_WVV
            self.backward_WVV = updated_backward_WVV
            self.spatial_insights = f"""Unobservable vehicles {self.unobserve_id_list_ahead} are ahead {self.agent_name}{self.agent_id}'s observable vehicles {self.observe_id_list} and unobservable vehicles {self.unobserve_id_list_behind} are behind {self.agent_name}{self.agent_id}'s observable vehicles {self.observe_id_list}, with more than 2 hops communication distance. {self.agent_name}{self.agent_id} cannot directly obtain the states and actions of these unobservable vehicles. This is episode {epi}. Your Spatial Insights are summarized as follows:
            <spatial-insights>
            - {self.agent_name}{self.agent_id}'s forward features: weighted average acceleration: {self.forward_WAA}; average distance: {self.forward_WAH}; average velocity: {self.forward_WAV}.
            - {self.agent_name}{self.agent_id}'s backward features: weighted average acceleration: {self.backward_WAA}; average distance: {self.backward_WAH}; avearge velocity: {self.backward_WAV}.
            - {self.agent_name}{self.agent_id}'s forward variances: weighted acceleration variance: {self.forward_WV}; distance variance: {self.forward_WVH}; velocity variance {self.forward_WVV}.
            - {self.agent_name}{self.agent_id}'s backward variances: weighted acceleration variance: {self.backward_WV}; distance variance: {self.backward_WVH}; velocity variance: {self.backward_WVV}.
            </spatial-insights>"""
        return spatial_dict
