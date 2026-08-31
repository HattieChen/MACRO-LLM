from LLMmodules.modules.utils import calculate_proposal_overlap, extract_from_label


CPP_CONSENSUS_DELTA = 0.1


def _has_cpp_local_consensus(agent_e, detailed_proposal_diff):
    agent_key = f"{agent_e.agent_name}{agent_e.agent_id}"
    return all(
        difference < CPP_CONSENSUS_DELTA
        for neighbor_differences in detailed_proposal_diff[agent_key].values()
        for difference in neighbor_differences.values()
    )


class Negotiation:
    def __init__(self, agents_list, comm_matrix=None, max_rounds=10):
        """
        Initialize the negotiation process among multiple agents.
        :param agents: List of agents
        :param max_rounds: Maximum number of negotiation rounds before termination
        """
        self.agents = agents_list
        self.max_rounds = max_rounds
        self.current_round = 0
        self.agreement = None
        self.comm_matrix = comm_matrix

    def evaluate_proposal(self, agent_e, received_proposal_list, received_output_list):
        """
        Evaluate whether the proposed agent accepts the given offer.
        :param agent: The agent evaluating the offer.
        :return: Boolean indicating acceptance.
        """
        detailed_proposal_diff, proposal_diff_percentage, proposal_mean = calculate_proposal_overlap(agent_e, received_output_list)
        is_consensus = _has_cpp_local_consensus(agent_e, detailed_proposal_diff)
        evaluate_prompt = f"""[Current Spatial Insights]
        {agent_e.spatial_insights}"""
        # if proposal_mean == 0:
        if is_consensus:
            evaluate_prompt += f"""[QUERY: Proposal Evaluation]
            You are an analyst for cooperative adaptive cruise control focusing on {agent_e.agent_name}{agent_e.agent_id} in a platoon. In this round, your and your neighbors' proposals are same, output as following format:
            <deal>True</deal>
            <action>({agent_e.agent_name}{agent_e.agent_id}'s proposed acceleration value)</action>
            IMPORTANT:
            - Please ensure that your output always includes both opening (<deal>, <action>) and closing tags (</deal>, </action>), such that every tag pair is properly closed and do not omit any closing tags.
            """
        else:
            if agent_e.agent_id == 1:
                evaluate_prompt += f"""[QUERY: Proposal Evaluation]
                You are an analyst for cooperative adaptive cruise control task, focusing on {agent_e.agent_name}{agent_e.agent_id} in a platoon. Your observable neighbor is {agent_e.agent_name}{agent_e.agent_id+1} (behind of {agent_e.agent_name}{agent_e.agent_id}). In the last negotiation round, consensus was not reached. Your Task is to analyze the disagreement and summarize your reasoning in the following steps:
                Step 0: Analyze whether the strategy proposed by {agent_e.agent_name}{agent_e.agent_id} conflicts with the strategy proposed by its observable neighbor {agent_e.agent_name}{agent_e.agent_id+1}:
                - Rule 1: If the two strategies are contradictory (e.g., one chooses ACCELERATION while the other chooses DECELERATION), suggest the observable neighbor's weight be set to 0.
                - Rule 2: If the strategies are consistent (both choose ACCELERATION or both choose DECELERATION), then suggest the observable neighbor's weight in the range [0, 0.5].
                - Rule 3: If the strategies do not conflict (at least one party chooses MAINTAINING), then suggest the observable neighbor's weight in the range [0, 0.3].
                Step 1: According to [Current Spatial Insights], analyze whether the strategy proposed by {agent_e.agent_name}{agent_e.agent_id} conflicts with the strategy proposed by unobservable vehicles {agent_e.unobserve_id_list_behind} behind of them:
                - Rule 1: If the two strategies are contradictory (e.g., one chooses ACCELERATION while the other chooses DECELERATION), suggest the observable neighbor's weight be set to 0.
                - Rule 2: If the strategies are consistent (both choose ACCELERATION or both choose DECELERATION), then suggest the observable neighbor's weight in the range [0, 0.5].
                - Rule 3: If the strategies do not conflict (at least one party chooses MAINTAINING), then suggest the observable neighbor's weight in the range [0, 0.3].

                Output as following format:
                <deal>False</deal>
                <reasons>
                (Observable neighboring vehicles:
                - {agent_e.agent_name}{agent_e.agent_id+1} (behind): choose [collaboration strategy], and proposes [acceleration]. It's strategy is match/conflict/not conflict with my strategy. The reason is ... I suggest it's weight in the range ...
                Unobservable vehicles {agent_e.unobserve_id_list_behind} behind of vehicles {agent_e.observe_id_list}: It's strategy is match/conflict/not conflict with my strategy. The reason is ...I suggest it's weight in the range ...)
                </reasons>
                """
            elif agent_e.agent_id == 8:
                evaluate_prompt += f"""[QUERY: Proposal Evaluation]
                You are an analyst for cooperative adaptive cruise control task, focusing on {agent_e.agent_name}{agent_e.agent_id} in a platoon. Your observable neighbor is {agent_e.agent_name}{agent_e.agent_id-1} (ahead of {agent_e.agent_name}{agent_e.agent_id}). In the last negotiation round, consensus was not reached. Your Task is to analyze the disagreement and summarize your reasoning in the following steps:
                Step 0: Analyze whether the strategy proposed by {agent_e.agent_name}{agent_e.agent_id} conflicts with the strategy proposed by its observable neighbor {agent_e.agent_name}{agent_e.agent_id-1}:
                - Rule 1: If the two strategies are contradictory (e.g., one chooses ACCELERATION while the other chooses DECELERATION), suggest the observable neighbor's weight be set to 0.
                - Rule 2: If the strategies are consistent (both choose ACCELERATION or both choose DECELERATION), then suggest the observable neighbor's weight in the range [0, 0.5].
                - Rule 3: If the strategies do not conflict (at least one party chooses MAINTAINING), then suggest the observable neighbor's weight in the range [0, 0.3].
                Step 1: According to [Current Spatial Insights], analyze whether the strategy proposed by {agent_e.agent_name}{agent_e.agent_id} conflicts with the strategy proposed by unobservable vehicles {agent_e.unobserve_id_list_ahead} ahead of them:
                - Rule 1: If the two strategies are contradictory (e.g., one chooses ACCELERATION while the other chooses DECELERATION), suggest the observable neighbor's weight be set to 0.
                - Rule 2: If the strategies are consistent (both choose ACCELERATION or both choose DECELERATION), then suggest the observable neighbor's weight in the range [0, 0.5].
                - Rule 3: If the strategies do not conflict (at least one party chooses MAINTAINING), then suggest the observable neighbor's weight in the range [0, 0.3].

                Output as following format:
                <deal>False</deal>
                <reasons>
                (Observable neighboring vehicles:
                - {agent_e.agent_name}{agent_e.agent_id-1} (ahead): choose [collaboration strategy], and proposes [acceleration]. It's strategy is match/conflict/not conflict with my strategy. The reason is ... I suggest it's weight in the range ...
                Unobservable vehicles {agent_e.unobserve_id_list_ahead} ahead of vehicles {agent_e.observe_id_list}: It's strategy is match/conflict/not conflict with my strategy. The reason is ... I suggest it's weight in the range ...)
                </reasons>
                """
            else:
                evaluate_prompt += f"""[QUERY: Proposal Evaluation]
                You are an analyst for cooperative adaptive cruise control task, focusing on {agent_e.agent_name}{agent_e.agent_id} in a platoon. Your observable neighbor is {agent_e.agent_name}{agent_e.agent_id-1} (ahead of {agent_e.agent_name}{agent_e.agent_id}) and {agent_e.agent_name}{agent_e.agent_id+1} (behind of {agent_e.agent_name}{agent_e.agent_id}). In the last negotiation round, consensus was not reached. Your Task is to analyze the disagreement and summarize your reasoning in the following steps:
                Step 0: Analyze whether the strategy proposed by {agent_e.agent_name}{agent_e.agent_id} conflicts with the strategy proposed by its observable neighbors {agent_e.agent_name}{agent_e.agent_id-1} and {agent_e.agent_name}{agent_e.agent_id+1}:
                - Rule 1: If the two strategies are contradictory (e.g., one chooses ACCELERATION while the other chooses DECELERATION), suggest the observable neighbor's weight be set to 0.
                - Rule 2: If the strategies are consistent (both choose ACCELERATION or both choose DECELERATION), then suggest the observable neighbor's weight in the range [0, 0.5].
                - Rule 3: If the strategies do not conflict (at least one party chooses MAINTAINING), then suggest the observable neighbor's weight in the range [0, 0.3].
                Step 1:According to [Current Spatial Insights], analyze whether the strategy proposed by {agent_e.agent_name}{agent_e.agent_id} conflicts with the strategy proposed by"""
                if agent_e.unobserve_id_list_ahead != [] and agent_e.unobserve_id_list_behind != []:
                    evaluate_prompt+= f" unobservable vehicles {agent_e.unobserve_id_list_ahead} ahead of them and unobservable vehicles {agent_e.unobserve_id_list_behind} behind of them in this round."
                elif agent_e.unobserve_id_list_ahead == [] and agent_e.unobserve_id_list_behind != []:
                    evaluate_prompt+= f"unobservable vehicles {agent_e.unobserve_id_list_behind} behind of them in this round."
                elif agent_e.unobserve_id_list_ahead != [] and agent_e.unobserve_id_list_behind == []:
                    evaluate_prompt+= f"unobservable vehicles {agent_e.unobserve_id_list_ahead} ahead of them in this round."
                evaluate_prompt += f"""
                - Rule 1: If the two strategies are contradictory (e.g., one chooses ACCELERATION while the other chooses DECELERATION), suggest the observable neighbor's weight be set to 0.
                - Rule 2: If the strategies are consistent (both choose ACCELERATION or both choose DECELERATION), then suggest the observable neighbor's weight in the range [0, 0.5].
                - Rule 3: If the strategies do not conflict (at least one party chooses MAINTAINING), then suggest the observable neighbor's weight in the range [0, 0.3].
                Output as following format:
                <deal>False</deal>
                <reasons>
                Observable vehicles:
                - {agent_e.agent_name}{agent_e.agent_id-1} (ahead): choose [collaboration strategy], and proposes [acceleration]. It's strategy is match/conflict/not conflict with my strategy. The reason is ... I suggest it's weight in the range ...
                - {agent_e.agent_name}{agent_e.agent_id+1} (behind): choose [collaboration strategy], and proposes [acceleration]. It's strategy is match/conflict/not conflict with my strategy. The reason is ... I suggest it's weight in the range ...
                """
                if agent_e.unobserve_id_list_ahead != []:
                    evaluate_prompt+= f"\nUnobservable vehicles {agent_e.unobserve_id_list_ahead} ahead of vehicles {agent_e.observe_id_list}: It's strategy is match/conflict/not conflict with my strategy. The reason is ... I suggest it's weight in the range ...) "
                if agent_e.unobserve_id_list_behind != []:
                    evaluate_prompt+= f"\nUnobservable vehicles {agent_e.unobserve_id_list_behind} behind of vehicles {agent_e.observe_id_list}: It's strategy is match/conflict/not conflict with my strategy. The reason is ... I suggest it's weight in the range ...)"
                evaluate_prompt += f"]</reasons>"
            evaluate_prompt += f"""IMPORTANT:
            - Please ensure that your output always includes both opening (<deal>, <reasons>) and closing tags (</deal>, </reasons>), such that every tag pair is properly closed and do not omit any closing tags.
            - If the aceleration value > 0, the strategy is ACCELERATION; If the aceleration value < 0, the strategy is DECELERATION; If the aceleration value ≈ 0, the strategy is MAINTAINING."""
        # Current Temporal Plan: {agent_e.temporal_insights}
        # Current Spatial Analysis: {agent_e.spatial_insights}"""
        if '<ProposalEvaluation>' in agent_e.over_episode_insights and '</ProposalEvaluation>' in agent_e.over_episode_insights:
            evaluate_prompt += f"""
                {extract_from_label(agent_e.over_episode_insights, 'ProposalEvaluation')}
            """
        # record evaluation query
        agent_e.append_user_question_to_file(user_question = evaluate_prompt)
        evaluate_prompt_dict = {
            "role": "user",
            "content": evaluate_prompt
        }
        # received_proposal_list = [{"role": "assistant", "name": f"{agent_e.agent_name}{agent_e.agent_id}", "content": f"My current [Temporal Plan]: {agent_e.temporal_insights}"}, {"role": "assistant", "name": f"{agent_e.agent_name}{agent_e.agent_id}", "content": f"My current [Spatial Analysis]: {agent_e.spatial_insights}"}]
        received_proposal_list.append(evaluate_prompt_dict)
        return received_proposal_list
        # response, response_content = agent_e.gen_response()
        # return response, response_content
