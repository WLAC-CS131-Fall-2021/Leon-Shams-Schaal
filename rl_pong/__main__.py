from datetime import datetime

import numpy as np
import tensorflow as tf
from pong import PARENT_DIR, Pong
from rl import Agent, ReplayMemory


def main():
    game = Pong([1920, 1080])

    agent = Agent(
        replay_memory=ReplayMemory(
            max_size=1_000_000, state_shape=game.state.shape, n_step=3
        ),
        minibatch_size=32,
        action_space=3,
        min_rp_mem_size=100,
        target_model_update_frequency=1000,
        max_no_op_steps=30,
    )

    # Create TensorBoard logger
    write_to_tensorboard = False
    name = "128x256x256"

    if write_to_tensorboard:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        agent_log_dir = f"{PARENT_DIR}/logs/{name}/{current_time}"
        summary_writer = tf.summary.create_file_writer(agent_log_dir)

        print(
            "Writing to TensorBoard. Launch TensorBoard with the command `tensorboard --logdir logs`."
        )

    episodes = 100_000
    episode_rewards = []

    while game.running and game.num_games < episodes:
        # Check if the user closed the window or clicked pause
        # game.user_interaction()

        # Agent move
        action = agent.select_action(game.state)
        state, reward, done = game.step(game.user_interaction())

        if state is not None:  # Don't train if the screen is paused
            # Train the agent
            agent.train(state, action, reward, done)
            episode_rewards.append(reward)

            # Log at the end of  each episode
            if done:
                total_rewards = np.sum(episode_rewards).round(4)
                reward_density = np.mean(episode_rewards).round(4)
                difficulty = round(game.calculate_difficulty(), 4)
                episode_rewards = []

                print(
                    f"Episode: {game.num_games}, Step: {agent.rp_mem.size}, Total Rewards: {total_rewards}, Reward Density: {reward_density}, Difficulty: {difficulty}"
                )

                if write_to_tensorboard:
                    with summary_writer.as_default():
                        tf.summary.scalar(
                            "Total Rewards", total_rewards, step=game.num_games
                        )
                        tf.summary.scalar(
                            "Reward Density", reward_density, step=game.num_games
                        )
                        tf.summary.scalar("Difficulty", difficulty, step=game.num_games)

    game.finish()


if __name__ == "__main__":
    main()
