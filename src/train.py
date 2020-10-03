from collections import deque
import numpy as np
import torch
from tqdm import tqdm


def train_multiagent(agent_1, agent_2, env, num_agents, n_episodes=10000, max_t=200):
    previous_score = 0.1
    scores_average_window = 100
    episode_scores = []  # list containing scores from each episode
    average_scores = []  # Average scores of the last 100 episodes
    brain_name = env.brain_names[0]
    goal_score = 1.0

    episode_loop = tqdm(range(1, n_episodes + 1), desc="Episode 0 | Avg Score: None", leave=False)  # First text to be shown on tqdm

    for _ in episode_loop:
        # reset the unity environment
        env_info = env.reset(train_mode=True)[brain_name]

        # get initial state of the unity environment
        states = env_info.vector_observations
        states = np.reshape(states, (1, 48))

        # reset the agent for the new episode
        agent_1.reset()
        agent_2.reset()

        agent_scores = np.zeros(num_agents)

        while True:
            # determine actions for the unity agents from current sate
            actions_1 = agent_1.act(states)
            actions_2 = agent_2.act(states)

            # send the actions to the unity agents in the environment and receive resultant environment information
            actions = np.concatenate((actions_1, actions_2), axis=0)
            actions = np.reshape(actions, (1, 4))
            env_info = env.step(actions)[brain_name]

            next_states = env_info.vector_observations  # get the next states for each unity agent in the environment
            next_states = np.reshape(next_states, (1, 48))
            rewards = env_info.rewards  # get the rewards for each unity agent in the environment
            dones = env_info.local_done  # see if episode has finished for each unity agent in the environment

            # Send (S, A, R, S') info to the training agent for replay buffer (memory) and network updates
            agent_1.step(states, actions_1, rewards[0], next_states, dones[0])
            agent_2.step(states, actions_2, rewards[1], next_states, dones[1])

            # set new states to current states for determining next actions
            states = next_states
            # print(states)
            # Update episode score for each unity agent
            agent_scores += rewards

            # If any unity agent indicates that the episode is done,
            # then exit episode loop, to begin new episode
            if np.any(dones):
                break

        episode_scores.append(np.max(agent_scores))
        average_score = np.mean(episode_scores[episode_loop.n - min(episode_loop.n, scores_average_window):episode_loop.n + 1])
        average_scores.append(average_score)

        episode_loop.set_description_str('Episode {} | Avg Score: {:.2f}'.format(episode_loop.n, average_score))
        if agent_scores.mean() > previous_score:
            previous_score = agent_scores.mean()
            episode_loop.set_description_str('Achieved in {} episode | Avg Score: {:.2f}'.format(episode_loop.n, average_score))
            torch.save(agent_1.actor_local.state_dict(), 'ag_1_checkpoint_actor.pth')
            torch.save(agent_1.critic_local.state_dict(), 'ag_1_checkpoint_critic.pth')
            torch.save(agent_2.actor_local.state_dict(), 'ag_2_checkpoint_actor.pth')
            torch.save(agent_2.critic_local.state_dict(), 'ag_2_checkpoint_critic.pth')
        if np.mean(average_scores) >= goal_score:
            episode_loop.set_description_str('Done! Achieved in {} episode | Avg Score: {:.2f}'.format(episode_loop.n, average_score))
            torch.save(agent_1.actor_local.state_dict(), 'ag_1_checkpoint_actor.pth')
            torch.save(agent_1.critic_local.state_dict(), 'ag_1_checkpoint_critic.pth')
            torch.save(agent_2.actor_local.state_dict(), 'ag_2_checkpoint_actor.pth')
            torch.save(agent_2.critic_local.state_dict(), 'ag_2_checkpoint_critic.pth')
            break
    return episode_scores, average_scores