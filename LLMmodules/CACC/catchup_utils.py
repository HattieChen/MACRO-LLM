import numpy as np
from typing import Tuple, Optional
import math

# def update_mean(old_mean, old_count, new_sample):
#     """
#     """
#     delta = new_sample - old_mean
#     new_mean = old_mean + delta / new_count
#     return new_mean

def welford_update_from_stats(mean_prev: float,
                              var_prev: float,
                              count_prev: int,
                              x_new: float,
                              *,
                              input_ddof: int = 1,
                              output_ddof: Optional[int] = None
                              ) -> Tuple[float, float, int]:
    if output_ddof is None:
        output_ddof = input_ddof

    if count_prev < 0:
        raise ValueError("count_prev 必须为非负整数")

    n_new = count_prev + 1

    if count_prev == 0:
        mean_new = float(x_new)
        M2_new = 0.0
    else:
        # input_ddof=1 -> M2_prev = var_prev * (count_prev - 1)
        # input_ddof=0 -> M2_prev = var_prev * count_prev
        M2_prev = var_prev * (count_prev - input_ddof)

        delta = x_new - mean_prev
        mean_new = mean_prev + delta / n_new
        delta2 = x_new - mean_new
        M2_new = M2_prev + delta * delta2

    denom = n_new - output_ddof
    var_new = (M2_new / denom) if denom > 0 else math.nan

    return mean_new, var_new

def welford_ew_update(mean_prev: Optional[float],
                      var_prev: Optional[float],
                      x_new: float,
                      weight: float,
                      *,
                      clamp_nonneg: bool = True
                      ) -> Tuple[float, float]:
    if not (0 < weight <= 1):
        raise ValueError("weight 必须在 (0, 1] 内")

    if mean_prev is None or var_prev is None:
        return float(x_new), 0.0

    delta = x_new - mean_prev
    mean_new = mean_prev + weight * delta
    var_new = (1.0 - weight) * (var_prev + weight * delta * delta)

    if clamp_nonneg and var_new < 0:
        var_new = 0.0

    return mean_new, var_new

'''
0730 ref_acc: error
'''
def ref_acc(
    ahead_velocity: float,
    my_distance: float,
    my_velocity: float,
    my_last_acceleration: float,
    ahead_acceleration: float,
    sample_rate: float = 0.5,
    v_des: float = 15.0,
    d_target: float = 20.0,
    d_min: float = 10.0,
    a_min: float = -2.5,
    a_max: float =  2.5,
    k_d: float = 0.20,
    k_delta: float = 1.00,
    k_s: float = 0.10,
    k_ff: float = 1.00,
    e_th: float = 5.0,     # [m]
    dv_th: float = 2.0,    # [m/s]
    jerk_per_sec: float = 5.0
) -> float:
    T = sample_rate
    v_f = ahead_velocity
    d   = my_distance
    v   = my_velocity
    a_prev = my_last_acceleration
    a_f_prev = ahead_acceleration

    alpha = 1.0 - abs(d - d_target) / max(e_th, 1e-6) - abs(v - v_f) / max(dv_th, 1e-6)
    alpha = max(0.0, min(1.0, alpha))

    a_base = (
        - k_d    * (d - d_target)
        - k_delta* (v - v_f)
        - alpha  * k_s * (v - v_des)
        + k_ff   * (a_f_prev)
    )

    da_max = jerk_per_sec * T
    a_low  = max(a_min, a_prev - da_max)
    a_high = min(a_max, a_prev + da_max)

    a_f_min = max(a_min, a_f_prev - da_max)
    a_safe_max_1 = a_f_min + (2.0 / (T*T)) * (d - d_min + (v_f - v) * T)
    a_ref = min(a_base, a_safe_max_1)
    a_ref = min(max(a_ref, a_low), a_high)

    v1  = v  + a_ref * T
    v1f = v_f + a_f_min * T
    d1  = d  + (v_f - v) * T + 0.5 * (a_f_min - a_ref) * (T*T)

    a_f2_min = max(a_min, a_f_min - da_max)

    a_safe_max_2 = a_f2_min + (2.0 / (T*T)) * (d1 - d_min + (v1f - v1) * T)

    min_feasible_next = max(a_min, a_ref - da_max)
    if min_feasible_next > a_safe_max_2:
        a_ref = min(a_ref, a_safe_max_2 + da_max)
        a_ref = min(max(a_ref, a_low), a_high)

    opt_acc = min(max(a_ref, a_low), a_high)
    return float(opt_acc)

def optimal_acc(ahead_velocity, my_headway, my_velocity, flag = 2, v_flag = 0, dt = 1, task_flag = 1):
    # task_flag = 1: catchup; task_flag = 2: slowdown
    dt_square = dt * dt
    headway_gap = abs(my_headway - 20)
    if int(v_flag) == 1:
        velocity_gap = abs(my_velocity - ahead_velocity)
    else:
        velocity_gap = abs(my_velocity - 15)
    headway_optimal_acc = (2 * ahead_velocity)/dt + (2 * my_headway)/dt_square - 40/dt_square - (2 * my_velocity)/dt

    if int(v_flag) == 1:
        velocity_optimal_acc = (ahead_velocity - my_velocity)/dt
    else:  # 0: use final velocity as optimal
        velocity_optimal_acc = (15 - my_velocity)/dt # with reward as target
    velocity_support_acc = (15 - my_velocity)/dt

    if task_flag == 1:
        # updated 3 rules optimal for CU task
        if headway_gap > 6:
            w_headway = 1
            w_velocity = 0
            w_velocity_2 = 0
        elif headway_gap<=6 and headway_gap > 3:
            w_headway = 0.2
            w_velocity = 0.6
            w_velocity_2 = 0.2
        elif headway_gap <=3 and headway_gap > 1:
            w_headway = 0.1
            w_velocity = 0.6
            w_velocity_2 = 0.3
        else:
            w_headway = 0.1
            w_velocity = 0.4
            w_velocity_2 = 0.5

        # if headway_gap >8:
        #     w_headway = 1
        #     w_velocity = 0
        # elif headway_gap<=8 and headway_gap > 4:
        #     w_headway = 0.3
        #     w_velocity = 0.7
        # elif headway_gap <=4 and headway_gap > 2:
        #     w_headway = 0.2
        #     w_velocity = 0.8
        # else:
        #     w_headway = 0.1
        #     w_velocity = 0.9

        if abs(velocity_optimal_acc) < 0.0001:
            velocity_optimal_acc = 0
        if abs(headway_optimal_acc) < 0.0001:
            headway_optimal_acc = 0
        opt_acc = w_headway * headway_optimal_acc + w_velocity * velocity_optimal_acc + w_velocity_2 * velocity_support_acc

    # if headway_gap > 6:
    #     w_headway = 1
    #     w_velocity = 0
    #     w_velocity_2 = 0
    # elif headway_gap<=6 and headway_gap > 3:
    #     w_headway = 0.2
    #     w_velocity = 0.6
    #     w_velocity_2 = 0.2
    # elif headway_gap <=3 and headway_gap > 1:
    #     if velocity_gap > 2:
    #         w_headway = 0.1
    #         w_velocity = 0.8
    #         w_velocity_2 = 0.1
    #     else:
    #         w_headway = 0.1
    #         w_velocity = 0.9
    #         w_velocity_2 = 0
    # else:
    #     w_headway = 0.1
    #     w_velocity = 0.9
    #     w_velocity_2 = 0

    if task_flag == 2:
        # updated 2 rule optimal for SD task
        if headway_gap >8:
            w_headway = 1
            w_velocity = 0
        elif headway_gap<=8 and headway_gap > 4:
            w_headway = 0.3
            w_velocity = 0.7
        elif headway_gap <=4 and headway_gap > 2:
            w_headway = 0.2
            w_velocity = 0.8
        else:
            w_headway = 0.1
            w_velocity = 0.9
        if abs(velocity_optimal_acc) < 0.0001:
            velocity_optimal_acc = 0
        if abs(headway_optimal_acc) < 0.0001:
            headway_optimal_acc = 0
        opt_acc = w_headway * headway_optimal_acc + w_velocity * velocity_optimal_acc

    return opt_acc, w_headway, headway_optimal_acc, w_velocity, velocity_optimal_acc

def vh_acc(my_velocity, vh, dt = 0.5):
    acceleration = (vh-my_velocity)/dt
    return acceleration

def observe_current_state(episode, state_list, agent_list, observable_range = 1, env_flag = 'catchup'):
    '''
    Transfer state list to a detailed states
    '''
    observe_state = list()
    for agent_id_i, agent_i in enumerate(agent_list):
        current_state_description = dict()
        agent_i_state = dict() #!
        if agent_i.agent_id == 1 and env_flag == 'catchup':
            virtual_leader_state = (0, 15, 15, 0)  # Include leader velocity.
            current_state_description['vehicle0'] = virtual_leader_state
        elif agent_i.agent_id == 1 and env_flag == 'slowdown':
            velocity_list = np.linspace(25.204, 15, 30).tolist()
            if episode < len(velocity_list):
                virtual_leader_state = (0, velocity_list[episode], 15, (15 - 25.204)/(30 - 1))  # Include leader velocity.
            else:
                virtual_leader_state = (0, 15, 15, 0)  # Include leader velocity.
            current_state_description['vehicle0'] = virtual_leader_state


        for agent_id_j, agent_j in enumerate(agent_list):
            if abs(agent_id_i - agent_id_j) <= observable_range:
                v_lead = state_list[agent_id_j][0]
                v_speed = state_list[agent_id_j][1]
                v_star = state_list[agent_id_j][2]
                v_vh = state_list[agent_id_j][3]
                h_state = state_list[agent_id_j][4]
                h_star = state_list[agent_id_j][5]
                u_state = state_list[agent_id_j][6]
                agent_i_state = (h_state, v_speed, v_lead, u_state, v_vh)  # Include leader velocity.
                current_state_description[f"{agent_j.agent_name}{agent_j.agent_id}"] = agent_i_state
        observe_state.append(current_state_description)
    # observe_state[0] = {
    # 'vehicle_0': (0, 15, 0),
    # 'vehicle_1': observe_state[0]['vehicle_1'],
    # 'vehicle_2': observe_state[0]['vehicle_2']}
    return observe_state

def observe_catchup_state(state_list, agent_list, observable_range = 1):
    '''
    Transfer state list to a detailed states
    '''
    observe_state = list()
    for agent_id_i, agent_i in enumerate(agent_list):
        agent_i_state = dict() #!
        for agent_id_j, agent_j in enumerate(agent_list):
            if abs(agent_id_i - agent_id_j) <= observable_range:
                v_lead = state_list[agent_id_j][0]
                v_speed = state_list[agent_id_j][1]
                v_star = state_list[agent_id_j][2]
                v_vh = state_list[agent_id_j][3]
                h_state = state_list[agent_id_j][4]
                h_star = state_list[agent_id_j][5]
                u_state = state_list[agent_id_j][6]
                # agent_i_state[f"Vehicle {agent_id_j}"] = f'Headway: {h_state} meters; Velocity: {v_speed} meters per seconds; Acceleration: {u_state} meters per second squared.'
                agent_i_state[f"Vehicle_{agent_id_j}"] = f'Headway: {h_state} meters; Velocity: {v_speed} meters per seconds; Acceleration: {u_state} meters per second squared.'
        agent_i_observe_state = f'<state>{agent_i_state}</state>'
        observe_state.append(agent_i_observe_state)
    return observe_state

def gen_reward_prompt(dt):
    dt_square = dt * dt
    prompt = f"""
    [Function: Next_State Computation]
    Given the my current state (distance, velocity), ahead vehicle state (distance_ahead, velocity_ahead), my action (acceleration), ahead vehicle action (proposed_current_ahead_acceleration, that is, you do not know the current_ahead_acceleration, you can propose an optimal current_ahead_acceleration for vehicle_ahead). The control interval is {dt} second.
    next_velocity = velocity + acceleration × {dt}
    ego_displacement = velocity × {dt} + 0.5 × acceleration × {dt}²
    ahead_displacement = velocity_ahead × {dt} + 0.5 × current_ahead_acceleration × {dt}²
    next_distance = distance + (ahead_displacement - ego_displacement)
    return next_velocity, next_distance

    [Function: Reward_score Calculation]
    Given the my current state (distance, velocity), ahead vehicle state (distance_ahead, velocity_ahead), my action (acceleration), ahead vehicle action (acceleration_ahead). The control interval is {dt} second.
    Calling function [Function: Next_State Computation] to calculate the next_velocity and next_distance.
    distance_reward = - (next_distance - 20)^2
    velocity_reward = - (next_velocity - ahead_velocity)^2
    acceleration_reward = - 0.1 * acceleration^2
    if next_distance <= 10 meters: return reward_score = -3000,
    else: return reward_score = distance_reward + velocity_reward + acceleration_reward

    [Equation: optimal action according to distance]
    distance_optimal_acceleration = (2 * distance)/{dt_square} + ahead_acceleration + (2 * velocity)/{dt} - 40/{dt_square} - (2*velocity)/{dt}.
    [Equation: optimal action according to velocity]
    velocity_optimal_acceleration = (ahead_velocity - velocity)/{dt} + ahead_acceleration
    [Equation: acceleration_ahead for vehilce ahead of me]
    ahead_optimal_acceleration = 40/{dt_square} - (2 * distance)/{dt} - (2 * ahead_velocity)/{dt_square} + my_acceleration + (2 * velocity)/{dt}
    """
    # distance_optimal_acceleration = (2 * distance)/{dt_square} + ahead_acceleration + (2 * velocity)/{dt} - 40/{dt_square} - (2*velocity)/{dt}.
    # [Equation: optimal action according to velocity]
    # velocity_optimal_acceleration = (ahead_velocity - velocity)/{dt} + ahead_acceleration
    return prompt

'''
def reward_in_one_round(
    # n,  # seconds
    my_headway,
    rear_headway,
    my_velocity,
    front_velocity,
    rear_velocity,
    my_acceleration,
    front_acceleration_next,
    rear_acceleration_next,
    my_acceleration_next
) -> float:
    n = 1  # seconds, 1s
    my_displacement = (
        my_velocity * n +
        0.5 * my_acceleration_next * n**2
    )

    front_displacement = (
        front_velocity * n +
        0.5 * front_acceleration_next * n**2
    )

    rear_displacement = (
        rear_velocity * n +
        0.5 * rear_acceleration_next * n**2
    )

    new_my_headway = my_headway + (front_displacement - my_displacement)
    new_rear_headway = rear_headway + (my_displacement - rear_displacement)

    headway_cost = (
        (new_my_headway - 20)**2 +
        (new_rear_headway - 20)**2
    )

    new_my_velocity = my_velocity + my_acceleration_next * n
    velocity_cost = (new_my_velocity - 15)**2

    acceleration_cost = (my_acceleration_next - my_acceleration)**2

    total_reward = - headway_cost - velocity_cost - acceleration_cost
    return total_reward
'''

def reward_in_one_round(E_Observable_State, Promising_proposal) -> float:
    """
    Observable_State: list of observable vehicles, the formate is E<episode_index>_state = {}
    E_proposal: promising proposal
    """
    n = 1  # seconds, 1s
    if len(E_Observable_State) == 3 & len(Promising_proposal) == 3:
        sorted_observable_state = [E_Observable_State[k] for k in sorted(E_Observable_State, key=lambda x: int(x.split("_")[1]))]
        sorted_promising_action = [Promising_proposal[k] for k in sorted(Promising_proposal, key=lambda x: int(x.split("_")[1]))]
        my_headway = sorted_observable_state[1][0] # my headway
        my_velocity = sorted_observable_state[1][1] # my velocity
        my_acceleration = sorted_observable_state[1][2] # my acceleration
        my_acceleration_next = sorted_promising_action[1]

        rear_headway = sorted_observable_state[2][0] # behind headway
        rear_velocity = sorted_observable_state[2][1] # behind velocity
        rear_acceleration = sorted_observable_state[2][2] # behind acceleration
        rear_acceleration_next = sorted_promising_action[2]

        front_headway = sorted_observable_state[0][0]
        front_velocity = sorted_observable_state[0][1]
        front_acceleration = sorted_observable_state[0][2]
        front_acceleration_next = sorted_promising_action[0]

        my_displacement = (
            my_velocity * n +
            0.5 * my_acceleration_next * n**2
        )
        front_displacement = (
            front_velocity * n +
            0.5 * front_acceleration_next * n**2
        )
        rear_displacement = (
            rear_velocity * n +
            0.5 * rear_acceleration_next * n**2
        )
        new_my_headway = my_headway + (front_displacement - my_displacement)
        new_rear_headway = rear_headway + (my_displacement - rear_displacement)
        headway_cost = (
            (new_my_headway - 20)**2 +
            (new_rear_headway - 20)**2
        )
        new_my_velocity = my_velocity + my_acceleration_next * n
        velocity_cost = (new_my_velocity - 15)**2
        acceleration_cost = (my_acceleration_next - my_acceleration)**2

    elif len(E_Observable_State) == 2 & len(Promising_proposal) == 2:
        if list(E_Observable_State())[0] == "vehicle_1" and list(E_Observable_State())[1] == "vehicle_2":
            my_headway = sorted_observable_state[0][0] # my headway
            my_velocity = sorted_observable_state[0][1] # my velocity
            my_acceleration = sorted_observable_state[0][2] # my acceleration
            my_acceleration_next = sorted_promising_action[0]

            rear_headway = sorted_observable_state[1][0] # behind headway
            rear_velocity = sorted_observable_state[1][1] # behind velocity
            rear_acceleration = sorted_observable_state[1][2] # behind acceleration
            rear_acceleration_next = sorted_promising_action[1]

            front_headway = 0
            front_velocity = 15
            front_acceleration = 0
            front_acceleration_next = 0

            my_displacement = (
                my_velocity * n +
                0.5 * my_acceleration_next * n**2
            )
            front_displacement = (
                front_velocity * n +
                0.5 * front_acceleration_next * n**2
            )
            rear_displacement = (
                rear_velocity * n +
                0.5 * rear_acceleration_next * n**2
            )
            new_my_headway = my_headway + (front_displacement - my_displacement)
            new_rear_headway = rear_headway + (my_displacement - rear_displacement)
            headway_cost = (
                (new_my_headway - 20)**2 +
                (new_rear_headway - 20)**2
            )
            new_my_velocity = my_velocity + my_acceleration_next * n
            velocity_cost = (new_my_velocity - 15)**2
            acceleration_cost = (my_acceleration_next - my_acceleration)**2
        elif list(E_Observable_State())[0] == "vehicle_7" and list(E_Observable_State())[1] == "vehicle_8":
            my_headway = sorted_observable_state[1][0] # my headway
            my_velocity = sorted_observable_state[1][1] # my velocity
            my_acceleration = sorted_observable_state[1][2] # my acceleration
            my_acceleration_next = sorted_promising_action[1]

            front_headway = sorted_observable_state[0][0]
            front_velocity = sorted_observable_state[0][1]
            front_acceleration = sorted_observable_state[0][2]
            front_acceleration_next = sorted_promising_action[0]

            my_displacement = (
                my_velocity * n +
                0.5 * my_acceleration_next * n**2
            )
            front_displacement = (
                front_velocity * n +
                0.5 * front_acceleration_next * n**2
            )
            rear_displacement = (
                rear_velocity * n +
                0.5 * rear_acceleration_next * n**2
            )
            new_my_headway = my_headway + (front_displacement - my_displacement)
            new_rear_headway = rear_headway + (my_displacement - rear_displacement)
            headway_cost = (
                (new_my_headway - 20)**2 +
                (new_rear_headway - 20)**2
            )
            new_my_velocity = my_velocity + my_acceleration_next * n
            velocity_cost = (new_my_velocity - 15)**2
            acceleration_cost = (my_acceleration_next - my_acceleration)**2
    total_reward = - headway_cost - velocity_cost - acceleration_cost
    return total_reward

def structed_received_proposals(proposal_record_dict, comm_matrix, agent_list):
    """
    Get the proposals of observable agents.
    """
    observable_proposal = dict()
    key_list = list(proposal_record_dict.keys())
    for agent in agent_list:
        observable_proposal[agent.agent_id] = ''
    for proposal_index, proposal_dict_key in enumerate(key_list):
        proposal = proposal_record_dict[proposal_dict_key]
        for agent in agent_list:
            if comm_matrix[proposal_index, agent.agent_id-1] == 1:
                observable_proposal[agent.agent_id] += proposal
    return observable_proposal

def reward_in_one_time_step(state_list, action_list):
    """
    state_list: list of state, the formate is [v_lead, v_speed, v_star, v_vh, h_state, h_star, u_state]
    action_list: list of actions for each agent
    """
    rewards = []
    for agent_id, state in enumerate(state_list):
        if agent_id == 0:
            acc_ahead = 0
        else:
            acc_ahead = action_list[agent_id-1]
        v_lead = state[0]
        v_speed = state[1]
        v_star = state[2]

        h_state = state[4]
        h_star = state[5]
        u_state = state[6]
        action = action_list[agent_id]

        # Step 1: Compute next velocity and position
        next_velocity = v_speed + action
        average_velocity = 0.5 * (v_speed + next_velocity)
        next_position = average_velocity  # assuming 1 second time step
        next_position_ahead = v_lead + acc_ahead + 0.5 * v_lead  # est. lead vehicle moves (simplified)
        next_headway = h_state + (next_position_ahead - next_position)

        # Step 2: Collision check
        if next_headway <= 10:
            reward = -300
        else:
            # Step 3: Compute reward components
            headway_error = next_headway - h_star
            velocity_error = next_velocity - v_star
            acceleration_penalty = action

            headway_reward = - (headway_error ** 2)
            velocity_reward = - (velocity_error ** 2)
            acceleration_reward = -0.1 * (acceleration_penalty ** 2)
            reward = headway_reward + velocity_reward + acceleration_reward

        rewards.append(reward)
    return rewards

if __name__ == "__main__":
    # ahead_velocity = 18
    # my_headway = 22
    # my_velocity = 14
    # opt_acc, w_headway, headway_optimal_acc, w_velocity, velocity_optimal_acc = optimal_acc(ahead_velocity, my_headway, my_velocity, flag = 2, v_flag = 1)
    pass
