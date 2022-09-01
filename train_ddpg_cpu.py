from numpy.core.numeric import False_
import tensorflow as tf
from control_and_ai.DDPG.ddpg import DDPG
from control_and_ai.DDPG.train import set_up
from control_and_ai.DDPG.test import test
from control_and_ai.DDPG.train_third_model_normalized import train as train_third_model_normalized
from control_and_ai.DDPG.train_second_model import train as train_second_model
from control_and_ai.DDPG.train_third_model_unnormalized import train as train_third_model_unnormalized
from control_and_ai.DDPG.train import train as train_first_model

from constants import DEGTORAD
from control_and_ai.DDPG.exploration import OUPolicy
from main_simulation import RocketLander

# with tf.device('/cpu:0'):
FLAGS = set_up()

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
#env = wrappers.Monitor(env, '/tmp/contlunarlander', force=True, write_upon_reset=True)

FLAGS.retrain = True #Restore weights if False
FLAGS.test = False #Test the model
FLAGS.num_episodes = 1
FLAGS.save_state_act = False #Export State and Action of a single episode as .xlsx (only work if num_episodes = 1)
FLAGS.save_reward = True #Export reward log as .xlsx
FLAGS.render_episodes = 1 #When to render

model_dir = 'F:/OneDrive - UNIVERSITAS INDONESIA/Semester 8/Skripsi/rocket-lander/control_and_ai/DDPG/model1' #local dir

with tf.device('/cpu:0'):
    agent = DDPG(
        action_bounds,
        eps,
        env.observation_space.shape[0],
        #16,
        actor_learning_rate=0.0001,
        critic_learning_rate=0.001,
        retrain=FLAGS.retrain,
        log_dir=FLAGS.log_dir,
        model_dir=model_dir)
        #batch_size=1000,
        #gamma=0.999)

    #test(env, agent, simulation_settings)
    train_first_model(env, agent, FLAGS)
    #train_second_model(env, agent, FLAGS)
    #train_third_model_normalized(env, agent, FLAGS)
    #train_third_model_unnormalized(env, agent, FLAGS)

