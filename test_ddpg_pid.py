import numpy as np
from numpy.core.numeric import False_
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from control_and_ai.DDPG.ddpg import DDPG
from control_and_ai.DDPG.utils import Utils
from control_and_ai.DDPG.exploration import OUPolicy

from control_and_ai.pid import PID_Benchmark

from constants import *
from constants import DEGTORAD
from environments.rocketlander import RocketLander, get_state_sample

action_bounds = [1, 1, 15*DEGTORAD]

eps = []
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))

simulation_settings = {'Side Engines': True,
                       'Clouds': True,
                       'Vectorized Nozzle': True,
                       'Graph': False,
                       'Render': False,
                       'Starting Y-Pos Constant': 1,
                       'Initial Force': 'random',
                       'Rows': 1,
                       'Columns': 2,
                       'Episodes': 500}

env = RocketLander(simulation_settings)

#Set both line below to False if you want to contniue training from a saved checkpoint
RETRAIN = False#Restore weights if False
TEST = True #Test the model

NUM_EPISODES = 1
SAVE_REWARD = False
SAVE_TO_EXCEL = False #Export states & actions logs as .xlsx

NAME = "test" #Model name

PID_SWITCH_HEIGHT = 15.72 #The height of the switch

#SIMULATE_WIND = True
# x_force = 0 # x-axis wind force in Newton
# y_force = 0 # y-axis wind force in Newton

# DDPG Model directory
model_dir = 'control_and_ai\DDPG\model_normal_state'

# Initialize DDPG Agent
agent = DDPG(
    action_bounds,
    eps,
    env.observation_space.shape[0], #for first model
    actor_learning_rate=0.001,
    critic_learning_rate=0.01,
    retrain=RETRAIN,
    log_dir="./logs",
    model_dir=model_dir,
    batch_size=100,
    gamma=0.99)

#Initialize PID algorithm
pid = PID_Benchmark()

obs_size = env.observation_space.shape[0]

util = Utils()
state_samples = get_state_sample(samples=5000, normal_state=True)
util.create_normalizer(state_sample=state_samples)

for episode in range(1, NUM_EPISODES + 1):
    old_state = None
    done = False
    total_reward = 0

    state = env.reset()
    state = util.normalize(state)
    max_steps = 500

    left_or_right_barge_movement = np.random.randint(0, 2)
    
    pid_switch = False #flag to denormalize state when switching to PID

    for t in range(max_steps): # env.spec.max_episode_steps
        old_state = state
        #Get current state for logging
        current_state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=True)
        xpos = current_state[0]-current_state[12]
        #Calculate rocket y-position relative to the screen bottom edge (ypos + ~0.7)
        ypos = current_state[1] - current_state[13] 
        
        # Use DDPG agent when the rocket is above the PID_SWITCH_HEIGHT
        if ypos > PID_SWITCH_HEIGHT:
            # infer an action
            action = agent.get_action(np.reshape(state, (1, obs_size)), not TEST)
            # Pass the action to the env and step through the simulation (1 step)
            state, reward, done, _ = env.step(action[0])
            state = util.normalize(state)
            print("X:", round(xpos, 3), "Y:", round(ypos, 3), "Controller: DDPG")
        
        # Use PID when the rocket is at or below the PID_SWITCH_HEIGHT    
        else:
            if pid_switch == False:
                print("Denormalizing State")
                state = util.denormalize(state)
                pid_switch = True
                
            # pass the state to the algorithm, get the actions
            action = pid.pid_algorithm(state) 
            # Pass the action to the env and step through the simulation (1 step).
            # Refer to Simulation Update in constants.py
            state, reward, done, _ = env.step(action)
            print("X:", round(xpos, 3), "Y:", round(ypos, 3), "Controller: PID")
        
        total_reward += reward
        
        # Refresh render
        env.refresh(render=True)

        # if SIMULATE_WIND:
        #     if state[LEFT_GROUND_CONTACT] == 0 and state[RIGHT_GROUND_CONTACT] == 0:
        #         env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement, x_force=x_force)
        #         env.apply_random_y_disturbance(epsilon=0.005, y_force=y_force)
        
        if done:
            print("Rocket landed")
            break
        
    env.close()