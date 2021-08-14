"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: This is a general training template.
"""

import sys
import os
import shutil
import argparse
sys.path.append('/content/rocket-lander/') #Colab dir
#sys.path.append('D://Coding//rocket-lander') #local dir
from environments.rocketlander import get_state_sample
from .utils import Utils
import numpy as np
import pandas as pd
from constants import *


def train(env, agent, FLAGS):
    print("Fuel Cost = 0, Max Steps = 500, Episode Training = %d, RANDOM FORCE = 20000, RANDOM X_FORCE = 0.2*RANDOM FORCE" %(FLAGS.num_episodes))

    obs_size = env.observation_space.shape[0]

    util = Utils()
    state_samples = get_state_sample(samples=5000, normal_state=True)
    util.create_normalizer(state_sample=state_samples)
    rew = []
    eps = []
    if FLAGS.save_state_act == True:
        xpos, ypos, xvel, yvel, lander_angle, angular_vel, rem_fuel, lander_mass = ([] for _ in range(8))
        fE, fS, pSi = ([] for _ in range(3))

    for episode in range(1, FLAGS.num_episodes + 1):
        old_state = None
        done = False
        total_reward = 0

        state = env.reset()
        state = util.normalize(state)
        max_steps = 500

        left_or_right_barge_movement = np.random.randint(0, 2)
        epsilon = 0.05

        for t in range(max_steps): # env.spec.max_episode_steps
            if FLAGS.show or episode % FLAGS.render_episodes == 0:
                env.refresh(render=True)

            old_state = state
            # infer an action
            action = agent.get_action(np.reshape(state, (1, obs_size)), not FLAGS.test)
            
            if FLAGS.save_state_act == True:
                current_state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=True)
                
                xpos.append(current_state[0]-current_state[12]) #xpos_rocket - xpos_landingPad
                ypos.append(current_state[1]-current_state[13]) #ypos_rocket - ypos_landingPad
                xvel.append(current_state[2]) #xdot
                yvel.append(current_state[3]) #ydot
                lander_angle.append(current_state[4]) #theta
                angular_vel.append(current_state[5]) #theta_dot
                rem_fuel.append(current_state[6]) #initial fuel = 0.2 * initial_mass
                lander_mass.append(current_state[7]) #initial_mass = 25.222
                
                fE.append(action[0][0])
                fS.append(action[0][1])
                pSi.append(action[0][2])

            # take it
            state, reward, done, _ = env.step(action[0])
            state = util.normalize(state)
            total_reward += reward

            if state[LEFT_GROUND_CONTACT] == 0 and state[RIGHT_GROUND_CONTACT] == 0:
                env.move_barge_randomly(epsilon, left_or_right_barge_movement)
                env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
                env.apply_random_y_disturbance(epsilon=0.005)

            if not FLAGS.test:
                # update q vals
                agent.update(old_state, action[0], np.array(reward), state, done)

            if done:
                break

        agent.log_data(total_reward, episode)

        if episode % 50 == 0 and not FLAGS.test:
            print('Saved model at episode', episode)
            agent.save_model(episode)
        rew.append(total_reward)
        eps.append(episode)
        print("Episode:\t{0}\tReward:\t{1}".format(episode, total_reward))

    if FLAGS.save_state_act == True:
        reward_data=pd.DataFrame(list(zip(rew,eps)),columns=['reward','episode'])
        state_data=pd.DataFrame(list(zip(xpos,ypos,xvel,yvel,lander_angle,angular_vel,rem_fuel,lander_mass)),\
            columns=['x_pos','y_pos','x_vel','y_vel','lateral_angle','angular_velocity','remaining_fuel','lander_mass'])
        action_data=pd.DataFrame(list(zip(fE,fS,pSi)),columns=['Fe','Fs','Psi'])
        with pd.ExcelWriter(f"DDPG_{total_reward}_model1.xlsx") as writer:
            state_data.to_excel(writer, sheet_name="state")
            action_data.to_excel(writer, sheet_name="action")
    
    if FLAGS.save_reward == True:
        reward_data=pd.DataFrame(list(zip(rew,eps)),columns=['reward','episode'])
        with pd.ExcelWriter(f"DDPG_rewardlogs_{total_reward}_{len(eps)}.xlsx") as writer:
            reward_data.to_excel(writer, sheet_name="epreward")
    
# Left here from the original code repo reference
def set_up():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_episodes',
        type=int,
        default=1000,
        help='How many episodes to train for'
    )

    parser.add_argument(
        '--show',
        default=False,
        action='store_true',
        help='At what point to render the cart environment'
    )

    parser.add_argument(
        '--wipe_logs',
        default=False,
        action='store_true',
        help='Wipe logs or not'
    )

    parser.add_argument(
        '--log_dir',
        default='logs',
        help='Where to store logs'
    )

    parser.add_argument(
        '--retrain',
        default=False,
        action='store_true',
        help='Whether to start training from scratch again or not'
    )

    parser.add_argument(
        '--test',
        default=False,
        action='store_true',
        help='Test more or no (true = no training updates)'
    )
    
    parser.add_argument(
        '--save_state_act',
        default=False,
        action='store_false',
        help='save state and action logs or no (true = save logs)'
    )
    
    parser.add_argument(
        '--save_reward',
        default=False,
        action='store_false',
        help='save reward or no (true = save reward logs)'
    )
    
    parser.add_argument(
        '--render_episodes',
        type=int,
        default=10,
        help='When to render the simulation'
    )

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.wipe_logs and os.path.exists(os.getcwd() + '/' + FLAGS.log_dir):
        shutil.rmtree(os.getcwd() + '/' + FLAGS.log_dir)

    return FLAGS
