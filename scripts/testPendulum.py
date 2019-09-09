import gym
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import InputLayer, Dense, Dropout
from keras.utils import plot_model
from keras import optimizers
import matplotlib.pyplot as plt

def discretize(state):
    state_discrete = state
    state_discrete[0] = round(state[0],1)
    state_discrete[1] = round(state[1],1)
    state_discrete[2] = round(state[2],2)
    return state_discrete

env = gym.make("Pendulum-v0")
s = env.reset()
print(s)

Nu = 21
controls = np.linspace(-2.0,2.0,Nu)

model = Sequential()
# model.add(InputLayer(batch_input_shape=(100, 3)))
model.add(Dense(16, activation='relu',input_shape=(3,)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(Nu, activation='linear'))
learning_rate = 0.1
model.compile(loss='mse', metrics=['mae'], optimizer=optimizers.Adam(lr=learning_rate))

plot_model(model, to_file='model.png', show_shapes='True')

num_epochs = 100

plt.figure(1)
plt.subplot(211)
plt.axis([0,num_epochs,-1,10000])
plt.subplot(212)
# plt.axis([0,num_epochs,-1,10000])
plt.ion()
plt.show()

# model = load_model('pendulum_test')

for epoch in range(num_epochs):
    print("Epoch: ", epoch, "/", num_epochs)

    total_reward = 0
    y = 0.95
    eps = 0.5
    decay = 0.99
    s = env.reset()

    worse_loss = -1E100
    best_loss = 1E100
    # s_vec = np.empty([100,3])
    # t_vec = np.empty([100,3])
    for k in range(100):

        #if epoch == 99:
        # if epoch%10 == 0:
        #     env.render()

        eps *= decay

        if np.random.random() < eps:
            a = np.random.randint(0,Nu)
        else:
            a = np.argmax(model.predict(s.reshape(-1,3)))
            # print("PREDICT: ", model.predict(s.reshape(-1,3)))
            # print("A: ", a)

        u = controls[a]
        # if a == 0:
        #     u = -2.0
        # elif a == 1:
        #     u = 0.0 #-100*s[2] - 100*s[1]
        # elif a == 2:
        #     u = 2.0

        new_s, r, done, _ = env.step([u])
        # print("original vs discretized")
        # print(new_s)
        # new_s[0] = round(new_s[0],1)
        # new_s[1] = round(new_s[1],1)
        # new_s[2] = round(new_s[0],2)
        new_s = discretize(new_s)
        s = discretize(s)
        # print(new_s)
        total_reward += r
        # print("r: ", r, "Qmax(s,a): ", np.max(model.predict(new_s.reshape(-1,3))))
        target = r + y*np.max(model.predict(new_s.reshape(-1,3))) # r+y*\max_a'{Q(s',a')}
        target_vec = model.predict(s.reshape(-1,3))[0] # Q(s,*)
        target_vec[a] = target # Q(s,a)
        # print(target_vec)
        # hist = model.fit(s.reshape(-1,3), target_vec.reshape(-1,Nu), batch_size = 100,epochs=1, verbose=0)
        hist = model.fit(s.reshape(-1,3), target_vec.reshape(-1,Nu), epochs=1, verbose=0)

        #
        # #print(hist.history)
        if hist.history['loss'][0] > worse_loss:
            worse_loss = hist.history['loss'][0]
        if hist.history['loss'][0] < best_loss:
            best_loss = hist.history['loss'][0]

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
    # hist = model.fit(s_vec, t_vec, batch_size = 10,epochs=1, verbose=1)
    # print(model.predict(s_vec_blindato.reshape(-1,3)))
    #print(hist.history)
    # if hist.history['loss'][0] < best_loss:
        # best_loss = hist.history['loss'][0]
    print("Best loss: ", best_loss,"Worse loss: ", worse_loss, " Total reward: ", total_reward)
    # print("Total reward: ", total_reward)
    plt.subplot(211)
    plt.plot(epoch, best_loss, 'go')
    plt.plot(epoch, worse_loss, 'ro')
    plt.subplot(212)
    plt.plot(epoch,total_reward,'ko')
    plt.draw()
    plt.pause(0.001)



model.save('pendulum_test')
