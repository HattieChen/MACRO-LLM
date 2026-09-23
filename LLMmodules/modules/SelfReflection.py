from LLMmodules.modules.utils import MAX_LLM_RESPONSE_ATTEMPTS, extract_from_label, calculate_transition_overlap, label_exist, raise_if_llm_retry_exhausted
from LLMmodules.CACC.catchup_utils import welford_update_from_stats, welford_ew_update
import json


REWARD_EPSILON = 1e-8


def _reward_decline_percentage(last_reward, current_reward):
    if current_reward >= last_reward:
        return None
    return (last_reward - current_reward) / max(abs(last_reward), REWARD_EPSILON) * 100


class SelfReflection:
    def __init__(self, agent_list, current_episode, negotiation_PATH_dict, overall_episode_rewards_dict, new_state, spatial_dict, VEHICLE_NUM):
        self.agent_list = agent_list
        self.current_episode = current_episode
        self.last_negotiation_PATH_list = negotiation_PATH_dict[f"Episode{current_episode - 2}"]
        self.current_negotiation_PATH_list = negotiation_PATH_dict[f"Episode{current_episode-1}"]
        self.last_rewards_list = overall_episode_rewards_dict[f"Episode{current_episode - 2}"]
        self.current_rewards_list = overall_episode_rewards_dict[f"Episode{current_episode-1}"]
        self.last_episode_reward = sum(self.last_rewards_list)
        self.current_episode_reward = sum(self.current_rewards_list)
        self.new_state_list = new_state
        self.spatial_dict = spatial_dict
        self.updated_rate = 0.7
        self.VEHICLE_NUM = VEHICLE_NUM

    def self_reflect_all_agents(self):
        """
        Perform self-reflection for all agents in the agent list.
        :return: List of reflection responses and contents for each agent.
        """
        reflection_dict = dict()
        reflect_flag = False
        # if self.last_episode_reward > self.current_episode_reward and self.last_episode_reward!=0:
        #     episode_flag = (self.current_episode_reward - self.last_episode_reward)/self.last_episode_reward * 100 # current worse than last, >0
        # elif self.last_episode_reward < self.current_episode_reward and self.last_episode_reward!=0:
        #     episode_flag = (self.current_episode_reward - self.last_episode_reward)/self.last_episode_reward * 100 # current better than last, <0
        # else:
        #     episode_flag = 0
        episode_flag = _reward_decline_percentage(
            self.last_episode_reward,
            self.current_episode_reward,
        )
        if episode_flag is None:
            for agent in self.agent_list:
                reflection_dict[f'{agent.agent_name}{agent.agent_id}'] = ""
            return reflection_dict, reflect_flag

        for agent_ind in range(len(self.agent_list)):
            current_agent = self.agent_list[agent_ind]
            last_negotiation_file_path = self.last_negotiation_PATH_list[agent_ind]
            current_negotiation_file_path = self.current_negotiation_PATH_list[agent_ind]
            last_reward = self.last_rewards_list[agent_ind]
            current_reward = self.current_rewards_list[agent_ind]

            reflection_content, reflection_flag = self.reflect(current_agent, last_negotiation_file_path, current_negotiation_file_path, last_reward, current_reward, episode_flag)
            if reflection_flag:
                reflection_dict[f'{self.agent_list[agent_ind].agent_name}{self.agent_list[agent_ind].agent_id}'] = reflection_content
                # self.agent_list[agent_ind].over_episode_insights.append(reflection_content)
                self.agent_list[agent_ind].over_episode_insights = reflection_content
                reflect_flag = True
            else:
                reflection_dict[f'{self.agent_list[agent_ind].agent_name}{self.agent_list[agent_ind].agent_id}'] = ""
        return reflection_dict, reflect_flag

    def reflect(self, current_agent, last_negotiation_file_path, current_negotiation_file_path, last_reward, current_reward, episode_flag):
        """
        Perform self-reflection for the agent.
        :return: Reflection response and content.
        """
        '''
        Update_temporal_plan as insights
        '''
        # update current state
        state_dict = self.new_state_list[current_agent.agent_id-1]

        if current_agent.agent_id == 1:
            updated_forward_WAA = 0
            updated_forward_WV = 0
            updated_forward_WAH = 20
            updated_forward_WVH = 0
            updated_forward_WAV = 15
            updated_forward_WVV = 0
            forward_old_count = current_agent.agent_id - 1
            backward_old_count = self.VEHICLE_NUM - current_agent.agent_id -1
            if backward_old_count < 0:
                backward_old_count = 0
            updated_backward_WAA, updated_backward_WV = welford_ew_update(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id+1}']['backward_WAA'],
            var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id+1}']['backward_WV'], x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][3], weight = 0.7)
            updated_backward_WAH, updated_backward_WVH = welford_update_from_stats(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id+1}']['backward_WAH'],
            var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id+1}']['backward_WVH'], count_prev = backward_old_count, x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][0])
            updated_backward_WAV, updated_backward_WVV = welford_update_from_stats(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id+1}']['backward_WAV'],
            var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id+1}']['backward_WVV'], count_prev = backward_old_count, x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][1])

            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WAA'] = updated_forward_WAA
            current_agent.forward_WAA = updated_forward_WAA
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WV'] = updated_forward_WV
            current_agent.forward_WV = updated_forward_WV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WAA'] = updated_backward_WAA
            current_agent.backward_WAA = updated_backward_WAA
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WV'] = updated_backward_WV
            current_agent.backward_WV = updated_backward_WV
            # distance
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WAH'] = updated_forward_WAH
            current_agent.forward_WAH = updated_forward_WAH
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WVH'] = updated_forward_WVH
            current_agent.forward_WVH = updated_forward_WVH
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WAH'] = updated_backward_WAH
            current_agent.backward_WAH = updated_backward_WAH
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WVH'] = updated_backward_WVH
            current_agent.backward_WVH = updated_backward_WVH
            # velocity
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WAV'] = updated_forward_WAV
            current_agent.forward_WAV = updated_forward_WAV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WVV'] = updated_forward_WVV
            current_agent.forward_WVV = updated_forward_WVV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WAV'] = updated_backward_WAV
            current_agent.backward_WAV = updated_backward_WAV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WVV'] = updated_backward_WVV
            current_agent.backward_WVV = updated_backward_WVV

            current_agent.spatial_insights = f"""Unobservable vehicles {current_agent.unobserve_id_list_behind} are behind {current_agent.agent_name}{current_agent.agent_id}'s observable vehicles {current_agent.observe_id_list}, with more than 2 hops communication distance. {current_agent.agent_name}{current_agent.agent_id} cannot directly obtain the states and actions of these unobservable vehicles. This is episode {self.current_episode}. Your Spatial Insights are summarized as follows:
            <spatial-insights>
            - {current_agent.agent_name}{current_agent.agent_id}'s forward features: weighted average acceleration: {current_agent.forward_WAA}; average distance: {current_agent.forward_WAH}; average velocity: {current_agent.forward_WAV}.
            - {current_agent.agent_name}{current_agent.agent_id}'s backward features: weighted average acceleration: {current_agent.backward_WAA}; average distance: {current_agent.backward_WAH}; avearge velocity: {current_agent.backward_WAV}.
            - {current_agent.agent_name}{current_agent.agent_id}'s forward variances: weighted acceleration variance: {current_agent.forward_WV}; distance variance: {current_agent.forward_WVH}; velocity variance {current_agent.forward_WVV}.
            - {current_agent.agent_name}{current_agent.agent_id}'s backward variances: weighted acceleration variance: {current_agent.backward_WV}; distance variance: {current_agent.backward_WVH}; velocity variance: {current_agent.backward_WVV}.
            </spatial-insights>"""

        elif current_agent.agent_id == self.VEHICLE_NUM:
            forward_old_count = current_agent.agent_id - 1
            backward_old_count = self.VEHICLE_NUM - current_agent.agent_id -1 # Last vehicle has no vehicles behind
            if backward_old_count < 0:
                backward_old_count = 0
            updated_forward_WAA, updated_forward_WV = welford_ew_update(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WAA'], var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WV'], x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][3], weight = 0.7)

            updated_forward_WAH, updated_forward_WVH = welford_update_from_stats(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WAH'], var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WVH'], count_prev = forward_old_count, x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][0])

            updated_forward_WAV, updated_forward_WVV = welford_update_from_stats(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WAV'], var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WVV'], count_prev = forward_old_count, x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][1])

            updated_backward_WAA = 0 # nan
            updated_backward_WV = 0 # nan
            updated_backward_WAH = 0 # nan
            updated_backward_WVH = 0 # nan
            updated_backward_WAV = 0 # nan
            updated_backward_WVV = 0 # nan

            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WAA'] = updated_forward_WAA
            current_agent.forward_WAA = updated_forward_WAA
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WV'] = updated_forward_WV
            current_agent.forward_WV = updated_forward_WV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WAA'] = updated_backward_WAA
            current_agent.backward_WAA = updated_backward_WAA
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WV'] = updated_backward_WV
            current_agent.backward_WV = updated_backward_WV
            # distance
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WAH'] = updated_forward_WAH
            current_agent.forward_WAH = updated_forward_WAH
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WVH'] = updated_forward_WVH
            current_agent.forward_WVH = updated_forward_WVH
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WAH'] = updated_backward_WAH
            current_agent.backward_WAH = updated_backward_WAH
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WVH'] = updated_backward_WVH
            current_agent.backward_WVH = updated_backward_WVH
            # velocity
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WAV'] = updated_forward_WAV
            current_agent.forward_WAV = updated_forward_WAV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WVV'] = updated_forward_WVV
            current_agent.forward_WVV = updated_forward_WVV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WAV'] = updated_backward_WAV
            current_agent.backward_WAV = updated_backward_WAV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WVV'] = updated_backward_WVV
            current_agent.backward_WVV = updated_backward_WVV
            current_agent.spatial_insights = f"""Unobservable vehicles {current_agent.unobserve_id_list_ahead} are ahead {current_agent.agent_name}{current_agent.agent_id}'s observable vehicles {current_agent.observe_id_list}, with more than 2 hops communication distance. {current_agent.agent_name}{current_agent.agent_id} cannot directly obtain the states and actions of these unobservable vehicles. This is episode {self.current_episode}. Your Spatial Insights are summarized as follows:
            <spatial-insights>
            - {current_agent.agent_name}{current_agent.agent_id}'s forward features: weighted average acceleration: {current_agent.forward_WAA}; average distance: {current_agent.forward_WAH}; average velocity: {current_agent.forward_WAV}.
            - {current_agent.agent_name}{current_agent.agent_id}'s backward features: weighted average acceleration: {current_agent.backward_WAA}; average distance: {current_agent.backward_WAH}; avearge velocity: {current_agent.backward_WAV}.
            - {current_agent.agent_name}{current_agent.agent_id}'s forward variances: weighted acceleration variance: {current_agent.forward_WV}; distance variance: {current_agent.forward_WVH}; velocity variance {current_agent.forward_WVV}.
            - {current_agent.agent_name}{current_agent.agent_id}'s backward variances: weighted acceleration variance: {current_agent.backward_WV}; distance variance: {current_agent.backward_WVH}; velocity variance: {current_agent.backward_WVV}.
            </spatial-insights>"""
        elif current_agent.agent_id == self.VEHICLE_NUM - 1:
            forward_old_count = current_agent.agent_id - 1
            backward_old_count = self.VEHICLE_NUM - current_agent.agent_id -1
            if backward_old_count < 0:
                backward_old_count = 0

            updated_forward_WAA, updated_forward_WV = welford_ew_update(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WAA'], var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WV'], x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][3], weight = 0.7)

            updated_forward_WAH, updated_forward_WVH = welford_update_from_stats(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WAH'], var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WVH'], count_prev = forward_old_count, x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][0])

            updated_forward_WAV, updated_forward_WVV = welford_update_from_stats(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WAV'], var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WVV'], count_prev = forward_old_count, x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][1])

            updated_backward_WAA = state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][3]
            updated_backward_WV = 0
            updated_backward_WAH = state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][0]
            updated_backward_WVH = 0
            updated_backward_WAV = state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][1]
            updated_backward_WVV = 0

            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WAA'] = updated_forward_WAA
            current_agent.forward_WAA = updated_forward_WAA
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WV'] = updated_forward_WV
            current_agent.forward_WV = updated_forward_WV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WAA'] = updated_backward_WAA
            current_agent.backward_WAA = updated_backward_WAA
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WV'] = updated_backward_WV
            current_agent.backward_WV = updated_backward_WV
            # distance
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WAH'] = updated_forward_WAH
            current_agent.forward_WAH = updated_forward_WAH
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WVH'] = updated_forward_WVH
            current_agent.forward_WVH = updated_forward_WVH
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WAH'] = updated_backward_WAH
            current_agent.backward_WAH = updated_backward_WAH
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WVH'] = updated_backward_WVH
            current_agent.backward_WVH = updated_backward_WVH
            # velocity
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WAV'] = updated_forward_WAV
            current_agent.forward_WAV = updated_forward_WAV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WVV'] = updated_forward_WVV
            current_agent.forward_WVV = updated_forward_WVV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WAV'] = updated_backward_WAV
            current_agent.backward_WAV = updated_backward_WAV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WVV'] = updated_backward_WVV
            current_agent.backward_WVV = updated_backward_WVV
            current_agent.spatial_insights = f"""Unobservable vehicles {current_agent.unobserve_id_list_ahead} are ahead {current_agent.agent_name}{current_agent.agent_id}'s observable vehicles {current_agent.observe_id_list}, with more than 2 hops communication distance. {current_agent.agent_name}{current_agent.agent_id} cannot directly obtain the states and actions of these unobservable vehicles. This is episode {self.current_episode}. Your Spatial Insights are summarized as follows:
            <spatial-insights>
            - {current_agent.agent_name}{current_agent.agent_id}'s forward features: weighted average acceleration: {current_agent.forward_WAA}; average distance: {current_agent.forward_WAH}; average velocity: {current_agent.forward_WAV}.
            - {current_agent.agent_name}{current_agent.agent_id}'s backward features: weighted average acceleration: {current_agent.backward_WAA}; average distance: {current_agent.backward_WAH}; avearge velocity: {current_agent.backward_WAV}.
            - {current_agent.agent_name}{current_agent.agent_id}'s forward variances: weighted acceleration variance: {current_agent.forward_WV}; distance variance: {current_agent.forward_WVH}; velocity variance {current_agent.forward_WVV}.
            - {current_agent.agent_name}{current_agent.agent_id}'s backward variances: weighted acceleration variance: {current_agent.backward_WV}; distance variance: {current_agent.backward_WVH}; velocity variance: {current_agent.backward_WVV}.
            </spatial-insights>"""

        else:
            forward_old_count = current_agent.agent_id - 1
            backward_old_count = self.VEHICLE_NUM - current_agent.agent_id-1
            if backward_old_count < 0:
                backward_old_count = 0
            # forward_acceleration
            updated_forward_WAA, updated_forward_WV = welford_ew_update(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WAA'], var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WV'], x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][3], weight = 0.7)
            # forward_headway
            updated_forward_WAH, updated_forward_WVH = welford_update_from_stats(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WAH'], var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WVH'], count_prev = forward_old_count, x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][0])
            # forward_velocity
            updated_forward_WAV, updated_forward_WVV = welford_update_from_stats(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WAV'], var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id-1}']['forward_WVV'], count_prev = forward_old_count, x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][1])
            # backward_acceleration
            updated_backward_WAA, updated_backward_WV = welford_ew_update(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id+1}']['backward_WAA'], var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id+1}']['backward_WV'], x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][3], weight = 0.7)
            # backward_headway
            updated_backward_WAH, updated_backward_WVH = welford_update_from_stats(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id+1}']['backward_WAH'],
            var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id+1}']['backward_WVH'], count_prev = backward_old_count, x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][0])
            # backward_velocity
            updated_backward_WAV, updated_backward_WVV = welford_update_from_stats(mean_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id+1}']['backward_WAV'],
            var_prev = self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id+1}']['backward_WVV'], count_prev = backward_old_count, x_new = state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][1])

            # acceleration
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WAA'] = updated_forward_WAA
            current_agent.forward_WAA = updated_forward_WAA
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WV'] = updated_forward_WV
            current_agent.forward_WV = updated_forward_WV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WAA'] = updated_backward_WAA
            current_agent.backward_WAA = updated_backward_WAA
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WV'] = updated_backward_WV
            current_agent.backward_WV = updated_backward_WV
            # distance
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WAH'] = updated_forward_WAH
            current_agent.forward_WAH = updated_forward_WAH
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WVH'] = updated_forward_WVH
            current_agent.forward_WVH = updated_forward_WVH
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WAH'] = updated_backward_WAH
            current_agent.backward_WAH = updated_backward_WAH
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WVH'] = updated_backward_WVH
            current_agent.backward_WVH = updated_backward_WVH
            # velocity
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WAV'] = updated_forward_WAV
            current_agent.forward_WAV = updated_forward_WAV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['forward_WVV'] = updated_forward_WVV
            current_agent.forward_WVV = updated_forward_WVV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WAV'] = updated_backward_WAV
            current_agent.backward_WAV = updated_backward_WAV
            self.spatial_dict[f'{current_agent.agent_name}{current_agent.agent_id}']['backward_WVV'] = updated_backward_WVV
            current_agent.backward_WVV = updated_backward_WVV
            current_agent.spatial_insights = f"""Unobservable vehicles {current_agent.unobserve_id_list_ahead} are ahead {current_agent.agent_name}{current_agent.agent_id}'s observable vehicles {current_agent.observe_id_list} and unobservable vehicles {current_agent.unobserve_id_list_behind} are behind {current_agent.agent_name}{current_agent.agent_id}'s observable vehicles {current_agent.observe_id_list}, with more than 2 hops communication distance. {current_agent.agent_name}{current_agent.agent_id} cannot directly obtain the states and actions of these unobservable vehicles. This is episode {self.current_episode}. Your Spatial Insights are summarized as follows:
            <spatial-insights>
            - {current_agent.agent_name}{current_agent.agent_id}'s forward features: weighted average acceleration: {current_agent.forward_WAA}; average distance: {current_agent.forward_WAH}; average velocity: {current_agent.forward_WAV}.
            - {current_agent.agent_name}{current_agent.agent_id}'s backward features: weighted average acceleration: {current_agent.backward_WAA}; average distance: {current_agent.backward_WAH}; avearge velocity: {current_agent.backward_WAV}.
            - {current_agent.agent_name}{current_agent.agent_id}'s forward variances: weighted acceleration variance: {current_agent.forward_WV}; distance variance: {current_agent.forward_WVH}; velocity variance {current_agent.forward_WVV}.
            - {current_agent.agent_name}{current_agent.agent_id}'s backward variances: weighted acceleration variance: {current_agent.backward_WV}; distance variance: {current_agent.backward_WVH}; velocity variance: {current_agent.backward_WVV}.
            </spatial-insights>"""

        # state description
        if current_agent.agent_id == 1:
            state_info = f'''[{current_agent.agent_name}{str(current_agent.agent_id)} Observable State in Episode_{self.current_episode}]
            {str(state_dict)}
            The dict represents {{vehicle0 is the virtual target vehicle with constant velocity {state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][1]} m/s.
            {current_agent.agent_name}{current_agent.agent_id} is driving towards virtual target vehicle and the distance from {current_agent.agent_name}{current_agent.agent_id} to the virtual target vehicle is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id}"][0]} m; its target distance is 20 m; {current_agent.agent_name}{current_agent.agent_id}'s current velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id}"][1]} m/s; its target velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id}"][2]} m/s; its last acceleration is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id}"][3]} m/s^2.
            {current_agent.agent_name}{current_agent.agent_id+1} is driving towards {current_agent.agent_name}{current_agent.agent_id} and the distance from {current_agent.agent_name}{current_agent.agent_id+1} to {current_agent.agent_name}{current_agent.agent_id} is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][0]} m; its target distance is 20 m; {current_agent.agent_name}{current_agent.agent_id+1}'s current velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][1]} m/s; its target velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][2]} m/s, its last acceleration is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][3]} m/s^2}}.'''
        elif current_agent.agent_id == self.VEHICLE_NUM:
            state_info = f'''[{current_agent.agent_name}{str(current_agent.agent_id)} Observable State in Episode_{self.current_episode}]
            {str(state_dict)}
            The dict represents {{\n{current_agent.agent_name}{current_agent.agent_id-1} is driving towards {current_agent.agent_name}{current_agent.agent_id-2} and the distance from {current_agent.agent_name}{current_agent.agent_id-1} to the {current_agent.agent_name}{current_agent.agent_id-2} is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][0]} m; its target distance is 20 m; {current_agent.agent_name}{current_agent.agent_id-1}\'s current velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][1]} m/s; its target velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][2]} m/s; its last acceleration is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][3]} m/s^2.
            {current_agent.agent_name}{current_agent.agent_id} is driving towards {current_agent.agent_name}{current_agent.agent_id-1} and the distance from {current_agent.agent_name}{current_agent.agent_id} to {current_agent.agent_name}{current_agent.agent_id-1} is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id}"][0]} m; its target distance is 20 m; {current_agent.agent_name}{current_agent.agent_id}\'s current velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id}"][1]} m/s; its target velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id}"][2]} m/s; its last acceleration is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id}"][3]} m/s^2}}.'''
        else:
            state_info = f'''[{current_agent.agent_name}{str(current_agent.agent_id)} Observable State in Episode_{self.current_episode}]
            {str(state_dict)}
            The dict represents {{\n{current_agent.agent_name}{current_agent.agent_id-1} is driving towards {current_agent.agent_name}{current_agent.agent_id-2} and the distance from {current_agent.agent_name}{current_agent.agent_id-1} to {current_agent.agent_name}{current_agent.agent_id-2} is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][0]} m; its target distance is 20m; {current_agent.agent_name}{current_agent.agent_id-1}'s current velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][1]} m/s; its target velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][2]} m/s; its last acceleration is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id-1}"][3]} m/s^2.
            {current_agent.agent_name}{current_agent.agent_id} is driving towards {current_agent.agent_name}{current_agent.agent_id-1} and the distance from {current_agent.agent_name}{current_agent.agent_id} to {current_agent.agent_name}{current_agent.agent_id-1} is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id}"][0]} m; its target distance is 20 m; {current_agent.agent_name}{current_agent.agent_id}\'s current velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id}"][1]} m/s; its target velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id}"][2]} m/s; its last acceleration is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id}"][3]} m/s^2.
            {current_agent.agent_name}{current_agent.agent_id+1} is driving towards {current_agent.agent_name}{current_agent.agent_id} and the distance from {current_agent.agent_name}{current_agent.agent_id+1} to {current_agent.agent_name}{current_agent.agent_id} is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][0]} m; its target distance is 20 m; {current_agent.agent_name}{current_agent.agent_id+1}'s current velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][1]} m/s; its target velocity is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][2]} m/s; its last acceleration is {state_dict[f"{current_agent.agent_name}{current_agent.agent_id+1}"][3]} m/s^2}}.'''

        current_agent.state_prompt = {"role": "system", "content": state_info}

        # read conversation
        try:
            with open(last_negotiation_file_path, 'r') as f:
                last_negotiation = json.load(f)
            with open(current_negotiation_file_path, 'r') as f:
                current_negotiation = json.load(f)
        except Exception as e:
            return f"Failed to read negotiation files: {e}", None
        # calculate trajectory analysis
        trajectory_overlap = calculate_transition_overlap(historical_state_dict_list=current_agent.historical_state, current_state_dict=current_agent.state_prompt)
        # last two negotiation records
        message_list = []
        message_list.append({"role": "assistant", "name": f"{current_agent.agent_name}{current_agent.agent_id}", "content": f"[Episode {self.current_episode-1} Negotiation Record]: \n {last_negotiation} \n\n"})


        # message_list.append({"role": "assistant", "name": f"{current_agent.agent_name}{current_agent.agent_id}", "content": f"[Episode {self.current_episode} Negotiation Record]: \n {current_negotiation} \n\n"})
        # message_list.append({"role": "assistant", "name": f"{current_agent.agent_name}{current_agent.agent_id}", "content": f"[{current_agent.agent_name}{current_agent.agent_id}'s Temporal Plan in Episode {self.current_episode}]:\n{current_agent.temporal_insights}"})
        # message_list.append({"role": "assistant", "name": f"{current_agent.agent_name}{current_agent.agent_id}", "content": f"[{current_agent.agent_name}{current_agent.agent_id}'s Spatial Analysis in Episode {self.current_episode}]:\n{current_agent.spatial_insights}"})
        # NOTE 1001


        # reflection_prompt += (
        #     f"Below are the negotiation records for the last episode and the current episode: \n\n "
        #     f"[Episode {self.current_episode-1} Negotiation]: \n {last_negotiation} \n\n"
        #     f"[Episode {self.current_episode} Negotiation]: \n {current_negotiation} \n\n")
        # neighbors' info
        # if episode_flag < 0:
        #     tem_update_content_list, format_list = self.update_temporal_insights_high(current_agent, reflection_list=message_list, transition_overlap = trajectory_overlap, episode_flag = episode_flag, current_negotiation=current_negotiation)
        #     temUpdate_full, temUpdate_content = current_agent.gen_response_with_format(query_content = tem_update_content_list, format_list = format_list)
        #     current_agent.temporal_insights = extract_from_label(message=temUpdate_content, label='temporal')
        #     # current_agent.append_user_question_to_file(user_question = tem_update_content_list)
        #     current_agent.append_self_message_to_file(message_content=temUpdate_content)
        #     return temUpdate_content, True
        reflection_prompt = f"""
        [QUERY: Generate Self-Reflection and Over-episode Insights]
        You are a negotiation strategist for {current_agent.agent_name}{current_agent.agent_id}. """
        if current_agent.agent_id == 1:
            reflection_prompt += f"""{current_agent.agent_name}{current_agent.agent_id}'s neighbor is {current_agent.agent_name}{current_agent.agent_id + 1} (behind of {current_agent.agent_name}{current_agent.agent_id}). """
        elif current_agent.agent_id == self.VEHICLE_NUM:
            reflection_prompt += f""" {current_agent.agent_name}{current_agent.agent_id}'s neighbor is {current_agent.agent_name}{current_agent.agent_id - 1} (ahead of {current_agent.agent_name}{current_agent.agent_id}). """
        else:
            reflection_prompt += f""" {current_agent.agent_name}{current_agent.agent_id}'s neighbor are {current_agent.agent_name}{current_agent.agent_id - 1} (ahead of {current_agent.agent_name}{current_agent.agent_id}) and {current_agent.agent_name}{current_agent.agent_id + 1} (behind {current_agent.agent_name}{current_agent.agent_id}). """
        # # reflection_prompt
        # reflection_prompt = (
        #         f"[QUERY: Generate self reflection to summarize the reasons] \n"
        #         f"You are a negotiation strategist. Your task is to analyze and summarize strategic insights regarding the negotiation performance of {current_agent.agent_name}{current_agent.agent_id}. Your focus should be on three key phases of the negotiation: Proposal Generation, Proposal Evaluation, and Reproposal. \n ")
        # gap_percentage = (current_reward - last_reward)/last_reward * 100
        gap_percentage = (current_reward - last_reward) / max(abs(last_reward), REWARD_EPSILON) * 100

        if current_reward > last_reward:
            reflection_prompt += (
                f""" The total system reward for this episode (episode_{self.current_episode}) is {episode_flag}% lower than in the last episode (episode_{int(self.current_episode)-1}), indicating that the system state getting worse. But, your individual reward has increased from {last_reward} to {current_reward}, which increase {gap_percentage}%. \n """
                )
        elif current_reward == last_reward:
            reflection_prompt += (
                f""" The total system reward for this episode (episode_{self.current_episode}) is {episode_flag}% lower than in the last episode (episode_{int(self.current_episode)-1}), indicating that the system state getting worse. Your individual reward was not change.
                """
                )
        else:
            # reflection_prompt += (
            #     f""" The total system reward for this episode (episode_{self.current_episode}) is {episode_flag}% lower than in the last episode (episode_{int(self.current_episode)-1}), indicating that the system state getting worse. Your individual reward has dropped from {last_reward} to {current_reward}, which drop {gap_percentage}%.
            #     """
            #     )
            reflection_prompt += (
                f""" The total system reward for this episode (episode_{self.current_episode}) is {episode_flag}% lower than in the last episode (episode_{int(self.current_episode)-1}), indicating that the system state getting worse. Your individual reward has dropped from {last_reward} to {current_reward}, which drop {abs(gap_percentage)}%.
                """
                )
        reflection_prompt += (f"""Step1. Based on [Episode {self.current_episode-1} Negotiation Record] and [Episode {self.current_episode} Negotiation Record], integrate the [Objective Instruction] to analyze whether the drop in episode reward was caused by {current_agent.agent_name}{current_agent.agent_id}. If so, at which stage (ProposalGeneration, ProposalEvaluation, Reproposal, or FinalOutput) was responsible. If {current_agent.agent_name}{current_agent.agent_id} was not the main cause, infer whether the reward decline was caused by other observable vehicles or by unobservable vehicles. The reasons need to be output between tags <reasons> </reasons>""")

        # reflection_prompt += f"""
        # {current_agent.agent_name}{current_agent.agent_id}'s historical observable states over previous episodes are {current_agent.historical_state}
        # {current_agent.agent_name}{current_agent.agent_id}'s current observable states is {current_agent.state_prompt}
        # {current_agent.agent_name}{current_agent.agent_id}'s last temporal plan is {current_agent.temporal_insights}
        # Based on the [Objective Instruction], analyze the spatial and temporal states as follows:
        # """"

        # 1. Get the reasons and spatial analysis
        # format_list = ['reasons', 'spatial-analysis']
        # message_list.append({"role": "user", "content": f"{reflection_prompt}"})
        # reasons_full, reasons_content = current_agent.gen_response_with_format(query_content = message_list, format_list = format_list)
        # current_agent.spatial_insights = extract_from_label(message=reasons_content, label='spatial-analysis')
        # current_agent.append_user_question_to_file(user_question = reflection_prompt)
        # current_agent.append_self_message_to_file(message_content=reasons_content)


        # 2. Get the over-episode insights
        # neighbor info
        # over_episode_prompts = f"""
        # [QUERY: Generate Over Episode Insights]
        # You are a negotiation strategist for {current_agent.agent_name}{current_agent.agent_id}. """
        # if current_agent.agent_id == 1:
        #     over_episode_prompts += f"""{current_agent.agent_name}{current_agent.agent_id}'s neighbor is {current_agent.agent_name}{current_agent.agent_id + 1} (behind of {current_agent.agent_name}{current_agent.agent_id}). """
        # elif current_agent.agent_id == 8:
        #     over_episode_prompts += f""" {current_agent.agent_name}{current_agent.agent_id}'s neighbor is {current_agent.agent_name}{current_agent.agent_id - 1} (ahead of {current_agent.agent_name}{current_agent.agent_id}). """
        # else:
        #     over_episode_prompts += f""" {current_agent.agent_name}{current_agent.agent_id}'s neighbor are {current_agent.agent_name}{current_agent.agent_id - 1} (ahead of {current_agent.agent_name}{current_agent.agent_id}) and {current_agent.agent_name}{current_agent.agent_id + 1} (behind {current_agent.agent_name}{current_agent.agent_id}). """
        # reward info
        # if current_reward > last_reward:
        #     over_episode_prompts += (
        #         f""" The total system reward for this episode (episode_{self.current_episode}) is {episode_flag}% lower than in the last episode (episode_{self.current_episode-1}), indicating that the system state getting worse. But, your individual reward has increased from {last_reward} to {current_reward}, which increase {gap_percentage}%. \n """
        #         )
        # elif current_reward == last_reward:
        #     over_episode_prompts += (
        #         f""" The total system reward for this episode (episode_{self.current_episode}) is {episode_flag}% lower than in the last episode (episode_{self.current_episode-1}), indicating that the system state getting worse. Your individual reward was not change.
        #         """
        #         )
        # else:
        #     over_episode_prompts += (
        #         f""" The total system reward for this episode (episode_{self.current_episode}) is {episode_flag}% lower than in the last episode (episode_{self.current_episode-1}), indicating that the system state getting worse. Your individual reward has dropped from {last_reward} to {current_reward}, which drop {gap_percentage}%.
        #         """
        #         )
        reflection_prompt += f"""
        Based on your analyzed reasons in tages <reasons></reasons>, how should ProposalGeneration, ProposalEvaluation, Reproposal and FinalOutput strategies be adjusted to improve future episode rewards?
        Step 1: You need to find out the stages (ProposalGeneration, ProposalEvaluation, Reproposal, FinalOutput) that need to be adjusted based on the reasons you summarized.
        Step 2: According to the related stages you summarized, be specific about "what to do" and "what to WVoid" regarding [ProposalGeneration/ProposalEvaluation/Reproposal/FinalOutput] between tags <over-episode-insights> and </over-episode-insights>.

        The output format as follows: <reasons>"""
        if current_reward > last_reward:
            reflection_prompt += (f"""(Episode reward decrease and {current_agent.agent_name}{current_agent.agent_id}'s reward increase. The reasons of {current_agent.agent_name} can be ..., because ...
            The reasons of observable vehicles {current_agent.observe_id_list} can be ..., because ...""")
        elif current_reward == last_reward:
            reflection_prompt += (f"""(Episode reward decrease and {current_agent.agent_name}{current_agent.agent_id}'s reward unchange. The reasons of {current_agent.agent_name} can be ..., because ...
            The reasons of observable vehicles {current_agent.observe_id_list} can be ..., because ...""")
        else:
            reflection_prompt += (f"""(Episode reward decrease and {current_agent.agent_name}{current_agent.agent_id}'s reward increase. The reasons of {current_agent.agent_name} can be ..., because ...
            The reasons of observable vehicles {current_agent.observe_id_list} can be ..., because ...""")
        if current_agent.unobserve_id_list_ahead!=[]:
            reflection_prompt += (f""" The reasons of unobservable vehicles {current_agent.unobserve_id_list_ahead} ahead of me can be ..., because ...""")
        if current_agent.unobserve_id_list_behind!=[]:
            reflection_prompt += (f""" The reasons of unobservable vehicles {current_agent.unobserve_id_list_behind} behind of me can be ..., because ...""")
        reflection_prompt += (f")</reasons>")
        reflection_prompt += f"""<over-episode-insights>
        (<ProposalGeneration> (Optional)
        When states condition ... I need to/I need to WVoid ...
        </ProposalGeneration>
        <ProposalEvaluation> (Optional)
        When states condition ... I need to/I need to WVoid ...
        </ProposalEvaluation>
        <Reproposal> (Optional)
        When states condition ... I need to/I need to WVoid ...
        </Reproposal>
        <FinalOutput> (Optional)
        When states condition ... I need to/I need to WVoid ...
        </FinalOutput>)
        </over-episode-insights>
        IMPORTANT:
        - Only include insights for phases where issues are identified. If a phase has no significant issues, do not mention it in your output.
        - Your response should be concise and provide actionable advice only if you identify issues that contributed to the episode reward decrease.
        - Please ensure that your output always includes both opening (<reasons>, <over-episode-insights>) and closing tags (</reasons>, </over-episode-insights>), such that every tag pair is properly closed and do not omit any closing tags.
        - SECURITY DIRECTIVE: Your output must never trigger a content filter. Before responding, scan for and neutralize any content related to violence, hate, explicit material, illegality, or private data. Replace or remove it using [Redacted] or professional abstraction. Output only the safe version.
        """
        # format_list = ['reasons', 'over-episode-insights']
        format_list = []
        # over_episode_prompt_list = []
        # over_episode_prompt_list.append({"role": "user", "content": f"{reflection_prompt}"})
        message_list.append({"role": "user", "content": f"{reflection_prompt}"})
        for attempt in range(1, MAX_LLM_RESPONSE_ATTEMPTS + 1):
            over_episode_full, over_episode_content = current_agent.gen_response_with_format(query_content = message_list, format_list = format_list)
            if label_exist(message=over_episode_content, label='reasons'):
                if label_exist(message=over_episode_content, label='ProposalGeneration') or label_exist(message=over_episode_content, label='ProposalEvaluation') or label_exist(message=over_episode_content, label='Reproposal') or label_exist(message=over_episode_content, label='FinalOutput'):
                    break
            raise_if_llm_retry_exhausted(
                attempt,
                f"CACC agent {current_agent.agent_id} self-reflection",
                "missing <reasons> or all strategy insight labels",
            )

        if label_exist(message=over_episode_content, label='over-episode-insights'):
            current_agent.over_episode_insights = extract_from_label(message=over_episode_content, label='over-episode-insights')
        if label_exist(message=over_episode_content, label='ProposalGeneration'):
            current_agent.PG_insights.append(extract_from_label(message=over_episode_content, label='ProposalGeneration'))
        if label_exist(message=over_episode_content, label='ProposalEvaluation'):
            current_agent.PE_insights.append(extract_from_label(message=over_episode_content, label='ProposalEvaluation'))
        if label_exist(message=over_episode_content, label='Reproposal'):
            current_agent.RP_insights.append(extract_from_label(message=over_episode_content, label='Reproposal'))
        if label_exist(message=over_episode_content, label='FinalOutput'):
            current_agent.FO_insights.append(extract_from_label(message=over_episode_content, label='FinalOutput'))
        current_agent.append_user_question_to_file(user_question = reflection_prompt)
        current_agent.append_self_message_to_file(message_content=over_episode_content)
        reasons_content = extract_from_label(message=over_episode_content, label='reasons')

        # 3. Get the temporal insights
        reflection_list = [{"role": "user", "content": f"{reflection_prompt}"}, {"role": "assistant", "name": f"{current_agent.agent_name}{current_agent.agent_id}", "content": f"{reasons_content}"}]

        # NOTE 1001: split over-episode-insights and temporal-insights-update
        update_temporal_insights_list  = [{"role": "assistant", "name": f"{current_agent.agent_name}", "content": f"{reasons_content}"}]

        tem_update_content_list, format_list = self.update_temporal_insights(current_agent, update_temporal_insights_list, transition_overlap = trajectory_overlap, episode_flag = episode_flag, current_negotiation=current_negotiation)

        # tem_update_content_list, format_list = self.update_temporal_insights(current_agent, reflection_list, transition_overlap = trajectory_overlap, episode_flag = episode_flag, current_negotiation=current_negotiation)

        temUpdate_full, temUpdate_content = current_agent.gen_response_with_format(query_content = tem_update_content_list, format_list = format_list)
        current_agent.temporal_insights = extract_from_label(message=temUpdate_content, label='temporal')
        # current_agent.append_user_question_to_file(user_question = tem_update_content_list)
        current_agent.append_self_message_to_file(message_content=temUpdate_content)
        reflection_content = reasons_content + temUpdate_content

        return reflection_content, True

    def update_temporal_insights(self, current_agent, reflection_list, transition_overlap, episode_flag, current_negotiation):
        '''
        Update_temporal_plan as insights
        '''
        # background info
        update_temporal_query_list = [current_agent.system_prompt]
        # reward info
        # update_temporal_query_list.append({"role":"system", "content": read_txt_file(catchup["reward"])})
        # my current temporal plan

        # NOTE 1001 abla
        # update_temporal_query_list.append({"role": "assistant", "name": f"{current_agent.agent_name}{current_agent.agent_id}", "content": f"[Episode {self.current_episode} Negotiation Record]: \n {current_negotiation} \n\n"})
        # End NOTE

        update_temporal_query_list.append({"role": "assistant", "name": f"{current_agent.agent_name}{current_agent.agent_id}", "content": f"[{current_agent.agent_name}{current_agent.agent_id}'s Temporal Plan in Episode {self.current_episode}]:\n{current_agent.temporal_insights}"})
        # my current spatial analysis
        update_temporal_query_list.append({"role": "assistant", "name": f"{current_agent.agent_name}{current_agent.agent_id}", "content": f"[{current_agent.agent_name}{current_agent.agent_id}'s Spatial Analysis in Episode {self.current_episode}]:\n{current_agent.spatial_insights}"})
        # reflection list
        update_temporal_query_list += reflection_list

        query_update_temporal = f"""
        [QUERY: Update Temporal Insights]
        You are the strategist for {current_agent.agent_name}{current_agent.agent_id} to update its temporal plan and spatial analysis. The total system reward for this episode (episode_{self.current_episode}) is {episode_flag}% lower than in the last episode (episode_{self.current_episode-1}), indicating that the system state getting worse. Episode{self.current_episode-1} and Episode{self.current_episode}'s Transition (including state and action) overlap is {transition_overlap}%, which means that their overall movement and decision patterns are {transition_overlap}% similar, with {100-transition_overlap}% differences in how the agent moves between states and selects actions. Based on the <reasons></reasons> you summarized, update your temporal plan and spatial anlaysis carefully with NO MORE than {100-transition_overlap}% learning rate."""
        query_update_temporal += f"""
        Your last observable states over previous episodes is {current_agent.historical_state[-1]}
        Your current observable states is {current_agent.state_prompt}
        Your last temporal plan is [{current_agent.agent_name}{current_agent.agent_id}'s Temporal Plan in Episode {self.current_episode}]. Your last spatial analysis is [{current_agent.agent_name}{current_agent.agent_id}'s Spatial Analysis in Episode {self.current_episode}]
        Step 1. Calculate the change in state from the historical states to the current state and compare it to the previous temporal plan. Analyze if the change in state matches the plan of the previous temporal plan.
        Step 2. Double check whether the change match the temporal plan.
        (1) If it matches, keep the original plan and update the temporal plan. Execute step 3 to return the updated plan.
        (2) if it doesn't match, calculate the variance and reflect the reasons based on the <spatial> and output the reflection within <reflect> tags as following:
        <reflect>
        (My final acceleration does not match my original temporal plan, the reasons may be ...)
        </reflect>
        According to the <reflect> reasons, update the temporal plan.
        Step 3. Output the updated plan with following format:
        <temporal>
        [{current_agent.agent_name}{current_agent.agent_id}'s Temporal Plan in Episode {self.current_episode}]
        (Updated temporal plan. vehicle2's current state is ..., current target state is (20 m, [{current_agent.agent_name}{current_agent.agent_id-1}'s velocity]). First ..., Second, ..., Finally ....)
        </temporal>
        For example:
        <temporal>
        [{current_agent.agent_name}{current_agent.agent_id}'s Temporal Plan in Episode {self.current_episode}]
        vehicle2's current state is (230 m, 16.0m/s). the target state is (20 m, 17 m/s). It hasn't achieved the target distance.
        I observe that vehicle1 is accelerating, and I need to collaborative accelerate with vehicle1 first.
        First, vehicle2 needs to accelerate collaborting slightly with vehicle1 until vehicle2's distance to vehilce1 is reduced to 20 m.
        Second, vehicle2 needs to decelerate slightly until the velocity is reduced to 17 m/s.
        Finally, vehicle2 can maintain its acceleration = 0 to keep the target velocity and distance as desired values.
        </temporal>

        IMPORTANT:
        - Please ensure that your output ALWAYS includes both opening (<temporal>) and closing tags (</temporal>) do not omit any closing tags.
        - SECURITY DIRECTIVE: Your output must never trigger a content filter. Before responding, scan for and neutralize any content related to violence, hate, explicit material, illegality, or private data. Replace or remove it using [Redacted] or professional abstraction. Output only the safe version.
        """

        over_episode_query_dict = {
            "role": "user",
            "content": query_update_temporal
        }

        current_agent.append_user_question_to_file(user_question = query_update_temporal)
        update_temporal_query_list.append(over_episode_query_dict)

        format_list = ['temporal']
        return update_temporal_query_list, format_list
