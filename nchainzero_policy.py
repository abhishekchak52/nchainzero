# implementing policy gradient using keras

import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import trange

import keras.layers as layers
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.initializers import glorot_uniform

def get_policy_model(env, hidden_layer_neurons, lr):
    num_actions = env.action_space.n
    inp = layers.Input(shape=[1], name='input_x')
    adv = layers.Input(shape=[1], name="advantages")
    x = layers.Dense(hidden_layer_neurons,
                    activation='relu',
                    use_bias=False,
                    kernel_initializer=glorot_uniform(seed=42),
                     name="dense_1")(inp)
    out = layers.Dense(num_actions,
                    activation='softmax',
                    kernel_initializer=glorot_uniform(seed=42),
                    use_bias=False,
                    name="out")(x)
    def custom_loss(y_true, y_pred):
        log_lik = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
        return K.mean(log_lik * adv, keepdims=True)

    model_train = Model(inputs=[inp, adv], outputs=out)
    model_train.compile(loss=custom_loss, optimizer=Adam(lr))
    model_predict = Model(inputs=[inp], outputs=out)
    return model_train, model_predict    

def discount_rewards(r, gamma=0.99):
    prior = 0
    out = []
    for val in r:
        new_val = val + prior * gamma
        out.append(new_val)
        prior = new_val
    return np.array(out[::-1])



def score_model(model, num_tests, render=False):
    scores = []    
    for _ in range(num_tests):
        observation = env.reset()
        reward_sum = 0
        while True:
            if render:
                env.render()
            state = np.reshape(observation, [1, dimen])
            predict = model.predict([state])[0]
            action = np.argmax(predict)
            observation, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                break
        scores.append(reward_sum)
    env.close()
    return np.mean(scores)

def policy_gradient_nn(env,num_games=100):
    model_train, model_predict = get_policy_model(env, hidden_layer_neurons, lr)
    reward = 0
    reward_sum = 0
    num_actions = env.action_space.n

    # Placeholders for our observations, outputs and rewards
    states = np.empty(0).reshape(0,dimen)
    actions = np.empty(0).reshape(0,1)
    rewards = np.empty(0).reshape(0,1)
    discounted_rewards = np.empty(0).reshape(0,1)

    # Setting up our environment
    observation = env.reset()
    num_episode = 0

    losses = []
    scores_list = []

    for num_game in trange(num_games):

        done  = False
        while not done:
            # Append the observations to our batch
            state = np.reshape(observation, [1, dimen])
            
            predict = model_predict.predict([state])[0]
            action = np.random.choice(range(num_actions),p=predict)
            
            # Append the observations and outputs for learning
            states = np.vstack([states, state])
            actions = np.vstack([actions, action])
            
            # Determine the oucome of our action
            observation, reward, done, _ = env.step(action)
            reward_sum += reward
            rewards = np.vstack([rewards, reward])
        
        if done:
            # Determine standardized rewards
            discounted_rewards_episode = discount_rewards(rewards, gamma)       
            discounted_rewards = np.vstack([discounted_rewards, discounted_rewards_episode])
            
            rewards = np.empty(0).reshape(0,1)
            if (num_game + 1) % batch_size == 0:
                discounted_rewards -= discounted_rewards.mean()
                discounted_rewards /= discounted_rewards.std()
                discounted_rewards = discounted_rewards.squeeze()
                actions = actions.squeeze().astype(int)
            
                actions_train = np.zeros([len(actions), num_actions])
                actions_train[np.arange(len(actions)), actions] = 1
                
                loss = model_train.train_on_batch([states, discounted_rewards], actions_train)
                losses.append(loss)

                # Clear out game variables
                states = np.empty(0).reshape(0,dimen)
                actions = np.empty(0).reshape(0,1)
                discounted_rewards = np.empty(0).reshape(0,1)


            # Print periodically
            if (num_game + 1) % print_every == 0:
                # Print status
                score = score_model(model_predict,10)
                # print("Average reward for training episode {}: {:0.2f} Test Score: {:0.2f} Loss: {:0.6f} ".format(
                #     (num_episode + 1), reward_sum/print_every, 
                #     score,
                #     np.mean(losses[-print_every:])))
                scores_list.append(score)
                if score >= goal:
                    print("Solved in {} episodes!".format(num_episode))
                    break
                reward_sum = 0
            observation = env.reset()
            
    # print(scores_list)

if __name__=='__main__':
    env = gym.make('NChain-v0')
    env.reset()

    hidden_layer_neurons = 10
    gamma = .99
    dimen = 1
    print_every = 10
    batch_size = 5
    render = False
    lr = 1e-2
    goal = 5000

    policy_gradient_nn(env)