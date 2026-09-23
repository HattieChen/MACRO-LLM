

import time
import sys
#from algorithms.algorithm import ReplayBuffer
import torch
import numpy as np
from datetime import datetime
import os
from LLMmodules.TaskConfiguration import catchup, slowdown
from LLMmodules.model_config import apply_model_override
# NOTE: LLM ------------------------------------------------------------
from LLMmodules.modules.Agent import CPPAgent as MAgent # GPT
import configparser
from pathlib import Path
from LLMmodules.modules.utils import MAX_LLM_RESPONSE_ATTEMPTS, initialize_conversation_context, read_txt_file,extract_from_label,initialize_comm_matrix, record_proposal_to_agents_comm_range, string_extract_number, collect_output_for_each_vehicle, raise_if_llm_retry_exhausted
from LLMmodules.CACC.catchup_utils import observe_current_state, reward_in_one_time_step, gen_reward_prompt
from LLMmodules.modules.negotiation import Negotiation
from LLMmodules.modules.SelfReflection import SelfReflection
# from modules.video.vs_utils import get_file_list, initialize_vr_conversation_context, read_record_conversation
# from modules.utils import gen_sys_info_str, gen_sys_info_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class OnPolicyRunner_LLM:
    def __init__(self, env_learn, env_test, env_args):
        # agent.load_state_dict(torch.load('./123.pt'))
        # load_state_dict(torch.load('./Models/49999best_actor.pt'))

        self.env_name = env_args.env
        self.max_steps = env_args.max_steps
        self.n_test = 1
        self.device = "cpu"
        self.env_learn = env_learn
        self.env_test = env_test
        self.n_agent = env_test.n_agent

    def run(self):
        return self.test()

    def test(self):
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        # global initialization ---------------------------------------------------------
        OBSERVE_RANGE = 1
        SPLIT_REPROPOSAL_FLAG = False
        time_t = time.time()
        length = self.max_steps
        returns = []
        scaled = []
        lengths = []
        episodes = []
        reward_list = []
        LLM_FILEPATH = PROJECT_ROOT / "experiments/RewardTest"
        COMM_MATRIX = initialize_comm_matrix(num_agents=self.n_agent, observable_range=OBSERVE_RANGE)
        # self-reflection records
        if self.env_name == 'catchup':
            CONFIG_PATH = "algorithms/envs/NCS/config/config_ma2c_nc_catchup.ini"
        elif self.env_name == 'slowdown':
            CONFIG_PATH = "algorithms/envs/NCS/config/config_ma2c_nc_slowdown.ini"
        else:
            raise ValueError(f"Unsupported environment: {self.env_name}")

        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        apply_model_override(config)
        overall_episode_rewards = dict()
        conversation_filepath_dict = dict()
        # LLM initialization --------------------------------------
        AGENT_LIST = []
        if int((config["LLM_CONFIG"]["KEEP_TEST_FLAG"])) == 1:
            MAX_ROUND = 1
            print("test weigted method")
        elif int((config["LLM_CONFIG"]["KEEP_TEST_FLAG"])) == 2:
            MAX_ROUND = 2
            print("for rebuttal-2")
        else:
            MAX_ROUND = 3
            print("original")
        # MAX_ROUND = 1

        CASE_FLAG = float(config["LLM_CONFIG"]["CASE_FLAG"])
        W_FLAG = float(config["LLM_CONFIG"]["W_FLAG"])
        V_FLAG = int(config["LLM_CONFIG"]["VELOCITY_FLAG"])
        # MODEL = float(config["LLM_CONFIG"]["MODEL"])
        DT = float(config["ENV_CONFIG"]["control_interval_sec"])

        for agent_i in range(self.n_agent):
            agent_id = agent_i+1
            agent = MAgent(AGENT_ID=agent_id, element='vehicle', CONFIG=config)
            AGENT_LIST.append(agent)

        # test cacc ------------------------------
        for i in range(self.n_test):
            # initialization --------------------------------------
            spatial_dict = {}

            # STEP1: initial states -----------------------------------
            episode = []
            env = self.env_test
            env.reset()
            d, ep_ret, ep_len = np.array([False]), 0, 0
            # record filepath ------------------------------
            date_str = datetime.today().strftime("%m%d")
            index = 1
            while True:
                file_name = f"{date_str}-reward-{index}.txt"
                file_path = os.path.join(str(LLM_FILEPATH), file_name)
                if not os.path.exists(file_path):
                    break
                index += 1
            proposal_record_dict = dict()
            while not (ep_len == length):
                within_episode_strategy_dict = dict()
                # NOTE: agent 0 is the target agent (for CACC catchup task)
                # without any description
                conv_filepath_list, message_list, SF_path_list = initialize_conversation_context(exp='catchup', epi=ep_len, n_agent=self.n_agent, config=config)
                for agent_i_index, agent_i in enumerate(AGENT_LIST):
                    agent_i.conv_path = conv_filepath_list[agent_i_index]
                    agent_i.SF_record_path = SF_path_list[agent_i_index]

                Episode_des = f'Episode {ep_len}:\n'
                # conversation_filepath_dict[f"Episode{ep_len}"] = conv_filepath_list
                conversation_filepath_dict[f"Episode{ep_len}"] = SF_path_list
                s = env.get_state_() # dim = 2 or 3 (vectorized)
                s = torch.as_tensor(s, dtype=torch.float, device=self.device)
                llm_act = list()
                llm_act_acc = np.zeros(self.n_agent)

                # llm_act_time = np.zeros(self.n_agent)
                # LLM decision ------------------------------------------
                current_state = env.transfer_state_llm(state_tensor=s)
                # STEP1: construct task -----------------------------
                # initial state -------------------
                # initial_state = observe_catchup_state(state_list=origin_state, agent_list=AGENT_LIST, observable_range=OBSERVE_RANGE)
                # Initialize the reward function from the current state.
                proposal_dict = dict()
                proposal_epi_dict = dict()
                rules_dict = dict()
                reasons_dict = dict()
                insights_dict = dict()
                deal_flag = np.full(self.n_agent, False)
                AGENT_DEC = np.zeros(self.n_agent)
                negotiation_dict = dict()
                for agent_index_i, agent_i in enumerate(AGENT_LIST):
                    task_description = f"[System Role Instruction]\n You are {agent_i.agent_name}{agent_i.agent_id} "
                    if self.env_name == 'catchup':
                        task_description += read_txt_file(catchup['path'])
                    elif self.env_name == 'slowdown':
                        task_description += read_txt_file(slowdown['path'])
                    else:
                        raise ValueError(f"Unsupported environment: {self.env_name}")
                    agent_i.system_prompt = {"role": "system", "content": task_description}
                    agent_i.append_system_prompt_to_file(system_prompt=task_description)
                # TODO: initial system prmopt -----------------------------
                observe_current_state_list = observe_current_state(episode = ep_len, state_list=current_state, agent_list=AGENT_LIST, observable_range=1, env_flag = self.env_name) # observe current state
                # Generate over-episode self-reflection.

                if ep_len > 1:
                    self_reflection = SelfReflection(agent_list=AGENT_LIST, current_episode=ep_len, negotiation_PATH_dict = conversation_filepath_dict, overall_episode_rewards_dict = overall_episode_rewards, new_state = observe_current_state_list, spatial_dict=spatial_dict, VEHICLE_NUM=self.n_agent)
                    reflection_dict, reflect_flag = self_reflection.self_reflect_all_agents()



                for round in range(MAX_ROUND+1):
                    proposal_round_dict = dict()
                    if round == MAX_ROUND:
                        continue
                    if round == 0:

                        # Initial proposal generation.
                        for agent_index_i, agent_i in enumerate(AGENT_LIST):

                            spatial_dict[f"{agent_i.agent_name}{agent_i.agent_id}"] = {"forward_WAA": agent_i.forward_WAA, "forward_WV": agent_i.forward_WV, "backward_WAA": agent_i.backward_WAA, "backward_WV": agent_i.backward_WV, "forward_WAH": agent_i.forward_WAH, "forward_WVH": agent_i.forward_WVH, "backward_WAH": agent_i.backward_WAH, "backward_WVH": agent_i.backward_WVH, "forward_WAV": agent_i.forward_WAV, "forward_WVV": agent_i.forward_WVV, "backward_WAV": agent_i.backward_WAV, "backward_WVV": agent_i.backward_WVV}

                            if ep_len == 0:
                                # Initial over-episode analysis.
                                ini_spa_tem_query_list, spa_tem_format_list = agent_i.initial_over_episode_insights(epi=ep_len, round=round, state=observe_current_state_list)
                                over_episode_full, over_episode_content = agent_i.gen_response_with_format(query_content = ini_spa_tem_query_list, format_list = spa_tem_format_list)
                                if over_episode_full is False and over_episode_content is False:
                                    print(f"[CACC test] agent_i.gen_response_with_format 返回 (False, False)：未生成 over_episode 内容。收到类型: full={type(over_episode_full).__name__}, content={type(over_episode_content).__name__}。请查看上方 [LLM ...] 日志确认 API 失败原因。")
                                    raise RuntimeError(
                                        "Initial over-episode LLM 调用未返回有效内容 (False, False)。常见原因：API 限流/配额用尽、内容审核 BadRequest。请查看控制台上方 [LLM ...] 日志。"
                                    )
                                # agent_i.spatial_insights = extract_from_label(over_episode_content, "spatial-analysis")
                                agent_i.temporal_insights = extract_from_label(over_episode_content, "temporal-analysis")
                                # record over-episode insights
                                agent_i.append_self_message_to_file(message_content=over_episode_content)

                                gen_proposal_query_content, iniProposal_format_list = agent_i.gen_proposal_catchup(epi = ep_len, round = round, last_reward = 0, state = observe_current_state_list, case_flag = CASE_FLAG, weight_flag = W_FLAG)
                                gen_proposal_query_content += f"My spatial insights are \n{agent_i.spatial_insights}"

                            elif ep_len > 1 and agent_i.over_episode_insights != "":
                                gen_proposal_query_content, iniProposal_format_list = agent_i.gen_proposal_catchup(epi = ep_len, round = round, last_reward = epi_rewards[agent_index_i], state = observe_current_state_list, case_flag = CASE_FLAG, weight_flag = W_FLAG)
                                gen_proposal_query_content += f"My spatial insights are \n{agent_i.spatial_insights}"

                            else:
                                gen_proposal_query_content, iniProposal_format_list = agent_i.gen_proposal_catchup(epi = ep_len, round = round, last_reward = epi_rewards[agent_index_i], state = observe_current_state_list, case_flag = CASE_FLAG, weight_flag = W_FLAG)
                                gen_proposal_query_content += f"My spatial insights are \n{agent_i.spatial_insights}"

                            # record conversation
                            agent_i.append_user_question_to_file(user_question = gen_proposal_query_content)
                            gen_proposal_query = []
                            # gen_proposal_query.append(background_dict)
                            gen_proposal_query.append(agent_i.system_prompt)
                            if int(V_FLAG) == 1:
                                # gen_proposal_query.append({"role":"system", "content": read_txt_file(catchup["rewardA"])})
                                gen_proposal_query.append({"role":"system", "content": gen_reward_prompt(DT)})
                            else:
                                gen_proposal_query.append({"role":"system", "content": read_txt_file(catchup["reward"])})

                            gen_proposal_query.append({"role":"assistant", "content": f"""Your current [Spatial Analysis] is {agent_i.spatial_insights}"""})
                            gen_proposal_query.append({"role":"assistant", "content": f"""Your current [Temporal Plan] is {agent_i.temporal_insights}"""})
                            gen_proposal_query.append(agent_i.state_prompt)
                            gen_proposal_query.append({"role":"user", "content": gen_proposal_query_content})
                            proposal_full, proposal_content = agent_i.gen_response_with_format(query_content = gen_proposal_query, format_list = iniProposal_format_list)

                            analysis = extract_from_label(proposal_content, 'analysis') # NOTE 1220: do not need to extract analysis for initial proposal
                            # record analysis
                            agent_i.append_self_message_to_file(message_content=analysis)
                            # SF record
                            analysis_SF = f"[{agent_i.agent_name}{agent_i.agent_id} Analysis for Initial Proposal in Episode_{ep_len}]: \n" + analysis
                            agent_i.append_self_message_to_SF(message_content=analysis_SF)
                            proposal = extract_from_label(proposal_content, 'proposal')
                            proposal += f"My spatial insights are \n{agent_i.spatial_insights}"
                            output = extract_from_label(proposal_content, 'output')
                            # Check the initial proposal.
                            if ep_len == 0 and round == 0 and agent_index_i == 0:
                                print('initial proposal:', proposal)
                            # record the initial proposal to conversation list
                            proposal_round_dict[f'{agent_i.agent_name}{agent_i.agent_id}'] = output
                            proposal_dict[f'{agent_i.agent_name}{agent_i.agent_id}'] = proposal
                            agent_i.append_self_message_to_file(message_content = proposal)   # record proposals after generation
                            # SF record
                            proposal_SF = f"[{agent_i.agent_name}{agent_i.agent_id}'s round{round} proposal in Episode_{ep_len}]: \n"+proposal
                            agent_i.append_self_message_to_SF(message_content=proposal_SF)


                        proposal_epi_dict[f'Round{round}'] = proposal_round_dict
                        # record_proposal_to_agents_comm_range(agent_list=AGENT_LIST, comm_range=1, proposal_dict=proposal_dict) # 0622
                        # SF record
                        received_proposal_list_dict = record_proposal_to_agents_comm_range(agent_list=AGENT_LIST, comm_range=1, proposal_dict=proposal_dict, round=round, episode = ep_len)

                        received_output_list_dict = collect_output_for_each_vehicle(proposal_round_dict=proposal_round_dict)

                    else:
                        # Consensus is evaluated independently in each negotiation round.
                        deal_flag.fill(False)
                        negotiation = Negotiation(agents_list=AGENT_LIST, comm_matrix=COMM_MATRIX, max_rounds=MAX_ROUND)
                        # Evaluate proposals for agreement.
                        for agent_index_i, agent_i in enumerate(AGENT_LIST):
                            agent_received_proposal_list = [
                                agent_i.system_prompt, agent_i.state_prompt
                            ]
                            agent_received_proposal_list += received_proposal_list_dict[f"{agent_i.agent_name}{agent_i.agent_id}"]
                            agent_received_output_dict = received_output_list_dict[f"{agent_i.agent_name}{agent_i.agent_id}"]
                            evaluate_message_list = negotiation.evaluate_proposal(agent_e=agent_i, received_proposal_list= agent_received_proposal_list, received_output_list=agent_received_output_dict)

                            # record evaluation prompts
                            # agent_i.append_user_question_to_file(user_question = evaluate_prompt_dict["content"])
                            evaluate_format_list = ['deal', 'action', 'reasons']
                            for attempt in range(1, MAX_LLM_RESPONSE_ATTEMPTS + 1):
                                evaluate, evaluate_content = agent_i.gen_response_with_format(
                                    query_content=evaluate_message_list,
                                    format_list=evaluate_format_list,
                                )
                                if evaluate is False and evaluate_content is False:
                                    print(f"{agent_i.agent_id} retry {attempt}, evaluation returned False")
                                    raise_if_llm_retry_exhausted(
                                        attempt,
                                        f"CACC agent {agent_i.agent_id} proposal evaluation",
                                        "response generator returned (False, False)",
                                    )
                                    continue
                                has_reasons = '<reasons>' in evaluate_content and '</reasons>' in evaluate_content
                                has_deal = '<deal>' in evaluate_content and '</deal>' in evaluate_content

                                if has_deal:
                                    if extract_from_label(message=evaluate_content, label='deal') == 'True':
                                        # record evaluation analysis and action
                                        agent_i.append_self_message_to_file(message_content=evaluate_content)
                                        has_action = '<action>' in evaluate_content and '</action>' in evaluate_content
                                        evaluate_content_SF = f"[{agent_i.agent_name}{agent_i.agent_id}'s proposal evaluation summarized from round{round} proposals in Episode_{ep_len}]: \n{evaluate_content}"
                                        agent_i.append_self_message_to_SF(message_content=evaluate_content_SF)
                                        if has_action:
                                            deal_flag[agent_index_i] = True
                                            output_action = string_extract_number(extract_from_label(evaluate_content, 'action'))
                                            if len(output_action) == 1:
                                                llm_act_acc[agent_index_i] = output_action[0]
                                                break
                                            else:
                                                print(f'{agent_i.agent_id} retry {attempt}, Deal True with action length error: {len(output_action)}')
                                        else:
                                            if '<action>' not in evaluate_content:
                                                print(f'{agent_i.agent_id} retry {attempt}, deal True with <action> error')
                                            if '</action>' not in evaluate_content:
                                                print(f'{agent_i.agent_id} retry {attempt}, deal True with </action> error')
                                    elif extract_from_label(message=evaluate_content, label='deal') == 'False' and has_reasons:
                                        # record evaluation analysis and reasons
                                        agent_i.append_self_message_to_file(message_content=evaluate_content)
                                        # SF records
                                        evaluate_content_SF = f"[{agent_i.agent_name}{agent_i.agent_id}'s proposal evaluation summarized from round{round} proposals in Episode_{ep_len}]: \n {evaluate_content}"
                                        agent_i.append_self_message_to_SF(message_content=evaluate_content_SF)
                                        break
                                    else:
                                        if '<reasons>' not in evaluate_content:
                                            print(f"{agent_i.agent_id} retry {attempt}, Deal false <reasons> error")
                                        if '</reasons>' not in evaluate_content:
                                            print(f"{agent_i.agent_id} retry {attempt}, Deal false </reasons> error")
                                raise_if_llm_retry_exhausted(
                                    attempt,
                                    f"CACC agent {agent_i.agent_id} proposal evaluation",
                                    "invalid deal result or action value",
                                )
                            within_episode_strategy_dict[f"{agent_i.agent_name}{agent_i.agent_id}"] = evaluate_content
                        # CHECK: whether consensus is reached
                        if all(deal_flag):
                            print("All agents have reached an agreement.")
                            break
                        else:
                            # No consensus, UPDATE: reproposal
                            # structured_proposals = structed_received_proposals(proposal_record_dict=proposal_epi_dict[f'Round{round-1}'], comm_matrix=COMM_MATRIX, agent_list=AGENT_LIST)
                            # Update: rules and proposal
                            for agent_index_i, agent_i in enumerate(AGENT_LIST):
                                agent_received_output_dict = received_output_list_dict[f"{agent_i.agent_name}{agent_i.agent_id}"]
                                # GEN: reproposal
                                # record reproposal query content

                                # Split insights and reproposals.
                                if SPLIT_REPROPOSAL_FLAG:
                                    if ep_len == 0:
                                        reproposal_query_dict = agent_i.gen_reproposal_A(epi = ep_len, round = round, state = observe_current_state_list, spatial_dict = spatial_dict)
                                    else:
                                        reproposal_query_dict = agent_i.gen_reproposal_A(epi = ep_len, round = round, last_reward = epi_rewards[agent_index_i], state = observe_current_state_list, spatial_dict = spatial_dict)
                                    reproposal_query_content = [agent_i.system_prompt, agent_i.state_prompt]
                                    reproposal_query_content += received_proposal_list_dict[f"{agent_i.agent_name}{agent_i.agent_id}"]
                                    within_episode_strategy = {
                                        "role": "assistant",
                                        "name": f"{agent_i.agent_name}{agent_i.agent_id}",
                                        "content": within_episode_strategy_dict[f"{agent_i.agent_name}{agent_i.agent_id}"]
                                    }
                                    reproposal_query_content.append(within_episode_strategy)
                                    reproposal_query_content.append(reproposal_query_dict)
                                    proposal_full, proposal_content = agent_i.gen_response_with_format(query_content = reproposal_query_content, format_list = ['proposal', 'analysis', 'output'])
                                    pass
                                    if ep_len == 0:
                                        reproposal_query_dict = agent_i.gen_reproposal_B(epi = ep_len, round = round, state = observe_current_state_list, spatial_dict = spatial_dict)
                                    else:
                                        reproposal_query_dict = agent_i.gen_reproposal_B(epi = ep_len, round = round, last_reward = epi_rewards[agent_index_i], state = observe_current_state_list, spatial_dict = spatial_dict)
                                    reproposal_query_content = [agent_i.system_prompt, agent_i.state_prompt]

                                # End split reproposal branch.
                                else:
                                    if ep_len == 0:
                                        reproposal_query_dict = agent_i.gen_reproposal(epi = ep_len, round = round, state = observe_current_state_list, spatial_dict = spatial_dict)
                                    else:
                                        reproposal_query_dict = agent_i.gen_reproposal(epi = ep_len, round = round, last_reward = epi_rewards[agent_index_i], state = observe_current_state_list, spatial_dict = spatial_dict)
                                # import pdb; pdb.set_trace()
                                    reproposal_query_content = [agent_i.system_prompt, agent_i.state_prompt]
                                    reproposal_query_content += received_proposal_list_dict[f"{agent_i.agent_name}{agent_i.agent_id}"]
                                within_episode_strategy = {
                                    "role": "assistant",
                                    "name": f"{agent_i.agent_name}{agent_i.agent_id}",
                                    "content": within_episode_strategy_dict[f"{agent_i.agent_name}{agent_i.agent_id}"]
                                }
                                reproposal_query_content.append(within_episode_strategy)
                                reproposal_query_content.append(reproposal_query_dict)

                                reproposal_format_list = ['proposal', 'analysis', 'output']
                                # import pdb; pdb.set_trace()
                                proposal_full, proposal_content = agent_i.gen_response_with_format(query_content = reproposal_query_content, format_list = reproposal_format_list)

                                # record the proposal
                                # agent_i.append_self_message_to_file(message_content=extract_from_label(proposal_content, 'analysis'))
                                analysis = extract_from_label(proposal_content, 'analysis')
                                proposal = extract_from_label(proposal_content, 'proposal')
                                proposal_dict[f'{agent_i.agent_name}{agent_i.agent_id}'] = proposal
                                # record the output
                                output = extract_from_label(proposal_content, 'output')

                                proposal_round_dict[f'{agent_i.agent_name}{agent_i.agent_id}'] = output
                                agent_i.append_self_message_to_file(message_content=analysis)
                                agent_i.append_self_message_to_file(message_content = proposal)
                                # SF record
                                analysis_SF = f"[{agent_i.agent_name}{agent_i.agent_id}'s insights summarized from the round{round-1} in Episode_{ep_len}]: \n {analysis}"
                                agent_i.append_self_message_to_SF(message_content = analysis_SF)
                                proposal_SF = f"[{agent_i.agent_name}{agent_i.agent_id}'s round{round} proposal in Episode_{ep_len}]: \n {proposal}"
                                agent_i.append_self_message_to_SF(message_content = proposal_SF)

                        proposal_epi_dict[f'Round{round}'] = proposal_round_dict
                        record_proposal_to_agents_comm_range(agent_list=AGENT_LIST, comm_range=1, proposal_dict=proposal_dict, round = round, episode=ep_len)

                        received_output_list_dict = collect_output_for_each_vehicle(proposal_round_dict=proposal_round_dict)
                    # self_reflection = SelfReflection(agent_list=AGENT_LIST, current_episode=ep_len, negotiation_PATH_dict = conversation_filepath_dict, overall_episode_rewards_dict = overall_episode_rewards)
                    # reflection_dict, reflect_flag = self_reflection.self_reflect_all_agents()
                # GEN: final action
                print("Out of negotiation loop.")
                for agent_index_i, agent_i in enumerate(AGENT_LIST):
                    for attempt in range(1, MAX_LLM_RESPONSE_ATTEMPTS + 1):
                        if ep_len == 0:
                            end_action, end_action_content = agent_i.end_negotiation(epi = ep_len, proposal_records = proposal_epi_dict)
                        else:
                            end_action, end_action_content = agent_i.end_negotiation(epi = ep_len, proposal_records = proposal_epi_dict, last_reward = epi_rewards[agent_index_i])

                        if end_action is False and end_action_content is False:
                            raise_if_llm_retry_exhausted(
                                attempt,
                                f"CACC agent {agent_i.agent_id} final action parsing",
                                "final negotiation returned (False, False)",
                            )
                            continue

                        has_analysis = '<analysis>' in end_action_content and '</analysis>' in end_action_content
                        has_action = '<action>' in end_action_content and '</action>' in end_action_content
                        has_decision = '<final_decision>' in end_action_content and '</final_decision>' in end_action_content

                        # if has_analysis and has_action:
                        if has_action:
                            final_action = extract_from_label(end_action_content, 'action')
                            output_action = string_extract_number(final_action)
                            if len(output_action) == 1:
                                llm_act_acc[agent_index_i] = output_action[0]
                                break
                            elif len(output_action) == 2 and output_action[1] == 3.0:
                                llm_act_acc[agent_index_i] = output_action[0]
                                break
                        elif has_decision:
                            final_action = extract_from_label(end_action_content, 'final_decision')
                            output_action = string_extract_number(final_action)
                            if len(output_action) == 1:
                                llm_act_acc[agent_index_i] = output_action[0]
                                break
                            elif len(output_action) == 2 and output_action[1] == 3.0:
                                llm_act_acc[agent_index_i] = output_action[0]
                                break
                        raise_if_llm_retry_exhausted(
                            attempt,
                            f"CACC agent {agent_i.agent_id} final action parsing",
                            "missing a single numeric <action> or <final_decision>",
                        )
                    agent_i.append_self_message_to_file(message_content=end_action_content)
                    # SF record
                    end_action_content_SF = f"[{agent_i.agent_name}{agent_i.agent_id}'s final decision in Episode_{ep_len}]: \n {end_action_content}"
                    agent_i.append_self_message_to_SF(message_content=end_action_content_SF)

                proposal_record_dict[f'Episode{ep_len}'] = proposal_epi_dict

                llm_act = llm_act_acc.tolist()
                print(f'actions {llm_act}')
                # epi_rewards = reward_in_one_time_step(state_list=current_state, action_list=llm_act)
                # overall_episode_rewards[f"Episode{ep_len}"] = epi_rewards
                _, reward, done_flag, global_reward = env.llm_step(llm_act)
                epi_rewards = reward.tolist()
                overall_episode_rewards[f"Episode{ep_len}"] = epi_rewards

                episode += [(s.tolist(), llm_act, reward.tolist())]
                d = np.array(d)
                episode_ret = reward.sum()
                print(f'episode: {ep_len}, reward: {episode_ret}, reward_detail: {reward}')
                if episode_ret == -24:
                    print(f"Episode terminated due to reaching the minimum rewards in episode {ep_len}.")
                    sys.exit()
                # NOTE: llm make decision ---------------------------------------
                with open(file_path, "a") as file:  # Open in append mode
                    file.write(f"{episode_ret}\n")  # Write the new reward on a new line
                # -----------------------------------------
                ep_len += 1
