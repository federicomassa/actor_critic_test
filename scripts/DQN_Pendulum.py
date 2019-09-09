import gym
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import InputLayer, Dense, Dropout
from keras.utils import plot_model
from keras import optimizers
import matplotlib.pyplot as plt
import collections

def discretize(state):
    state_discrete = state
    state_discrete[0] = round(state[0],1)
    state_discrete[1] = round(state[1],1)
    state_discrete[2] = round(state[2],2)
    return state_discrete

Transition = collections.namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        samplesIndex = np.random.choice(self.capacity, batch_size, False)
        samples = [self.memory[i] for i in samplesIndex]
        states = np.array([x[0] for x in samples])
        actions = np.array([x[1] for x in samples])
        next_states = np.array([x[2] for x in samples])
        rewards = np.array([x[3] for x in samples])

        return states, actions, next_states, rewards



    def __len__(self):
        return len(self.memory)

capacity = 2000
memory = ReplayMemory(capacity)
mini_batch_size = 128
env = gym.make("Pendulum-v0")
s = env.reset()

Nu = 3
controls = np.linspace(-2.0,2.0,Nu)

# Every how many cicles update the target network
TARGET_UPDATE = 10

targetNN = Sequential()
# targetNN.add(InputLayer(batch_input_shape=(100, 3)))
targetNN.add(Dense(24, activation='relu',input_shape=(3,)))
targetNN.add(Dropout(0.5))
targetNN.add(Dense(24, activation='relu'))
targetNN.add(Dropout(0.5))
# targetNN.add(Dense(256, activation='relu'))
# targetNN.add(Dropout(0.5))
# targetNN.add(Dense(128, activation='relu'))
# targetNN.add(Dropout(0.5))
targetNN.add(Dense(Nu, activation='linear'))
learning_rate = 0.001
targetNN.compile(loss='mse', metrics=['mae'], optimizer=optimizers.Adam(lr=learning_rate))

print(targetNN.summary())

policyNN = targetNN

#plot_targetNN(targetNN, to_file='targetNN.png', show_shapes='True')

num_epochs = 5000

plt.figure(1)
plt.subplot(211)
plt.axis([0,num_epochs,-100,0])
plt.subplot(212)
plt.axis([0,num_epochs,-1000,0])
plt.ion()
plt.show()

policyNN = load_model('pendulum_test')
targetNN = load_model('pendulum_test')

for epoch in range(num_epochs):
    print("Epoch: ", epoch, "/", num_epochs)

    #Reset memory
    replayMemory = []

    total_reward = 0
    best_reward = -100000
    y = 0.99
    eps = 0.5
    decay = 0.98
    s = env.reset()

    worse_loss = -1E100
    best_loss = 1E100
    # s_vec = np.empty([100,3])
    # t_vec = np.empty([100,3])
    for k in range(200):

        #if epoch == 99:
        # if epoch == num_epochs - 1:
        #if epoch % 10 == 0:
        #    env.render()

        eps *= decay

        if np.random.random() < eps:
            a = np.random.randint(0,Nu)
        else:
            #print("WTF!!!!")
            a = np.argmax(policyNN.predict(s.reshape(-1,3)))
            # print("PREDICT: ", targetNN.predict(s.reshape(-1,3)))
            # print("A: ", a)

        u = controls[a]
        # if a == 0:
        #     u = -2.0
        # elif a == 1:
        #     u = 0.0 #-100*s[2] - 100*s[1]
        # elif a == 2:
        #     u = 2.0

        new_s, _, done, _ = env.step([u])



        # print("original vs discretized")
        # print(new_s)
        # new_s[0] = round(new_s[0],1)
        # new_s[1] = round(new_s[1],1)
        # new_s[2] = round(new_s[0],2)
        #new_s = discretize(new_s)
        #s = discretize(s)

        # From discretized states
        reward = -10*(np.arctan2(new_s[1], new_s[0])**2 +0.1*new_s[2]**2 + 0.001*u**2) 

        if reward > best_reward:
            best_reward = reward

        # Store into replay memory
        memory.push(s, a, new_s, reward)

        # print(new_s)
        # print("r: ", r, "Qmax(s,a): ", np.max(targetNN.predict(new_s.reshape(-1,3))))
        #print("s: ", s.reshape(-1,3),"a: ", a,  "Q: ", targetNN.predict(s.reshape(-1,3)))

        # if np.abs(np.arctan2(new_s[1], new_s[0])) < 0.3:
        #     target = 0.0
        # elif np.arctan2(new_s[1], new_s[0]) > (np.pi - 0.3) or np.arctan2(new_s[1], new_s[0]) < (-np.pi + 0.3):
        #     target = -1.0
        # else:
        #     #target = -1.0/(np.pi**2)*(np.arctan2(new_s[1], new_s[0]))**2 + y*np.max(targetNN.predict(new_s.reshape(-1,3))) # r+y*\max_a'{Q(s',a')}
        
        # target = reward + y*np.max(targetNN.predict(new_s.reshape(-1,3))) # r+y*\max_a'{Q(s',a')}
        total_reward += reward

        # target_vec = targetNN.predict(s.reshape(-1,3))[0] # Q(s,*)

        # # Clamping
        # clampMin = -1.0
        # clampMax = 1.0
        # if target - target_vec[a] < clampMin:
        #     target = target_vec[a] + clampMin
        # elif target - target[a] > clampMax:
        #     target = target[a] + clampMax
        
        # target_vec[a] = target # Q(s,a)
        
        

        # print(target_vec)
        # hist = targetNN.fit(s.reshape(-1,3), target_vec.reshape(-1,Nu), batch_size = 100,epochs=1, verbose=0)
        #hist = targetNN.fit(s.reshape(-1,3), target_vec.reshape(-1,Nu), epochs=1, verbose=0)

        #
        # #print(hist.history)
        # if hist.history['loss'][0] > worse_loss:
        #     worse_loss = hist.history['loss'][0]
        # if hist.history['loss'][0] < best_loss:
        #     best_loss = hist.history['loss'][0]

        # if k==0:
        #     s_vec = np.array(s)
        #     t_vec = np.array(target_vec)
        # else:
        #     s_vec = np.vstack([s_vec,s])
        #     t_vec = np.vstack([t_vec,target_vec])

        s = new_s
        # s = new_s_continuous

    # if epoch == 0:
    #     s_vec_blindato = s_vec
    #     memoria = t_vec
    # else:
    #     memoria = memoria + 0.1*(t_vec - memoria)
    #if (total_reward > -100000):


    #hist = targetNN.fit(s_vec, t_vec, batch_size = 10,epochs=1, verbose=1)
    # print(targetNN.predict(s_vec_blindato.reshape(-1,3)))
    #print(hist.history)
    # if hist.history['loss'][0] < best_loss:
        # best_loss = hist.history['loss'][0]
    #print("Best loss: ", best_loss,"Worse loss: ", worse_loss, " Total reward: ", total_reward)
    # print("Total reward: ", total_reward)

    if len(memory) < capacity:
            continue

    states, actions, new_states, rewards = memory.sample(mini_batch_size)
        
    t_vec = []
    for iter in range(mini_batch_size):
        target_vec = []
        target = reward + y*np.max(targetNN.predict(new_states[iter].reshape(-1,3))) # r+y*\max_a'{Q(s',a')}
        target_vec = targetNN.predict(new_states[iter].reshape(-1,3))[0]
        #print(target_vec)
        target_vec[actions[iter]] = target

        if len(t_vec) == 0:
            t_vec = np.array(target_vec)
        else:
            t_vec = np.vstack([t_vec, target_vec])
    
    policyNN.fit(states, t_vec, batch_size=mini_batch_size)

    #Update TargetNN
    if (epoch % TARGET_UPDATE == 0):
        print("Updating TargetNN")
        targetNN = policyNN
        print("------- Target NN Weights ------")
        print(targetNN.get_weights())
        print("------- Policy NN Weights ------")
        print(policyNN.get_weights())
        print("----- Batch states -----")
        print(states)

    plt.subplot(211)
    plt.plot(epoch, best_reward, 'go')
    plt.subplot(212)
    plt.plot(epoch,total_reward,'ko')
    plt.draw()
    plt.pause(0.001)



policyNN.save('pendulum_test')
