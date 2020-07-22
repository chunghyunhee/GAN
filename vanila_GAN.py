# for Valnila with Fully Connected layer 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# plot visualization excetion
matplotlib_is_available = True
try:
    from matplotlib import pyplot as plt
except ImportError:
    print("will skip plotting; matplotlib is not available")
    matplotlib_is_available = False

# Data params
data_mean = 4
data_stddev = 1.25

# only one of these to define what data is actually sent to the D
(name, preprocess, d_input_Func) = ("Only 4 momemts", lambda data : get_moments(data), lambda x : 4)
print("using data [%s]"%(name))

# data : target data and Genertor input data
# target data
def get_distribution_sample(mu,sigma):
    return lambda n : torch.Tensor(np.random.normal(mu, sigma, (1,n))) # Gaussian
# noise data
def get_generator_input_sampler():
    return lamda m,n : torch.rand(m,n) # uni-dist data into Generator

# Models : Generator model and Discriminator model
# D와 G는 같은 과정의 Fully connected layer로 연결
class Generator(nn.Module):
    # 생성자에 해당하는 __init__부분
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    # 순전파
    def foward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hiden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    # 순전파
    def foward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.f(self.map3(X))

# Detector에서 가장 큰 error값을 저장하기 위해 쓰는 함수
# train하면서 loss값을 출력하기 위한 함수
def extract(v):
    return v.data.storage().tolist()

# train하면서 각 분포를 나타내면서 출력하기 위한 함수
def stats(d):
    return [np.mean(d), np.std(d)]

def get_moments(d):
    # return the first 4 momets of the data provided
    mean = torch.mean(d)
    diffs = d-mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs/ std # d값의 정규화
    skews= torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
    final = torch.cat((mean.shape(1, ), std.reshape(1,), skews.reshape(1,), kurtoses.reshape(1,)))

def decorate_with_diffs(data, expoenets, remove_raw_data = True):
    mean = torch.mean(data.data, 1, keepdum = True)
    mean_broadcast = torch.mul(data.data, 1, keepdim = True)
    diffs = torch.pow(data - Variable(mean_broadcast), expoenets)
    # 설정한 remove data에 대한 설정
    if remove_raw_data:
        return torch.car([diffs], 1)
    else :
        return torch.cat([data, diffs], 1)

# train with loss function
def train():
    # for model parameters
    g_input_size = 1 # random noise dim coming into Generator, per output vector
    g_hidden_size = 5 # generative complexity
    g_output_size = 1 # size of generated output vector
    d_input_size = 500 # minibatch_size - cardinality of dist
    d_hidden_size = 10 # discriminator complexity
    d_output_size = 1 # single dimention for real or not
    minibatch_size = d_input_size

    # hyperparameter for train process
    d_learning_rate = 1e-3
    g_learning_rate = 1e-3
    sgd_momentum = 0.9

    num_epoches = 5000
    print_interval = 100
    d_steps = 20
    g_steps = 20

    dfe, dre, ge = 0, 0, 0
    d_real_data, d_fake_data, g_fake_data = None, None, None

    discriminator_activation_function = torch.sigmoid
    generator_activation_function = torch.tanch

    # for target data
    d_sampler = get_distribution_sample(data_mean, data_stddev)
    # for fake data(noise data)
    gi_sampler = get_generator_input_sampler()

    # Generator layer
    G = Generator(input_size = g_input_size,
                  hidden_size = g_hidden_size,
                  output_size = g_output_size,
                  f = generator_activation_function)

    D = Discriminator(input_size = d_input_Func(d_input_suze),
                      hidden_size = d_hidden_size,
                      output_size = d_output_size,
                      f = discriminator_activation_function)

    # loss Binary cross entropy
    criterion = nn.BCELoss() # binary class entropy

    # gradient descent method for optimization
    d_optimizer = optim.SGD(D.parameters(), lr = d_learning_rate, momentum = sgd_momentum)
    g_optimizer = optim.SGD(G.parameters(), lr = g_learning_rate, momentum = sgd_momentum)

    # epoch and optimization
    for epoch in range(num_epoches):
        for d_index in range(d_steps):
            # 1. train D on real+fake
            D.zero_grad()

            # 1-1. Train D on real
            d_real_data = Variable (d_sampler(d_input_size))
            d_real_dicision = D(preprocess(d_real_data))
            d_real_error = criterion(d_real_dicision, Variable(torch.ones([1,1])))
            d_real_error.backward()
            # 1-2. train D on fake
            d_gen_input = Variable(gi_smapler(minibatch_size, g_input_size)) #새로운 fake값 생성
            d_fake_data = G(d_get_input).detach()
            d_fake_decision = D(preprocess(d_fake_data.t()))
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1,1])))
            d_fake_error.backward()
            d_optimizer.step()

            dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

        # for train Forger(Generator)
        for g_index in range(g_steps):
            G.zero_grad()

            # 이미지 자체를 Generator의 input으로 사용한다,
            gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            g_fake_data = G(gen_input)
            df_fake_descition = D(preprocess(g_fake_data()))
            g_error = criterion(df_fake_descition, Variable(torch.ones([1,1])))

            g_error.backward()
            g_optimizer.step()
            ge = extract(g_error)[0]

        # for print
        if epoch % print_interval == 0
            print("Epoch %s: D (%s real_err, %s fake_err) G (%s err); Real Dist (%s),  Fake Dist (%s) " %
                  (epoch, dre, dfe, ge, stats(extract(d_real_data)), stats(extract(d_fake_data))))

    # visulization
    if matplotlib_is_available:
        print("plotting the generated dist")
        values = extract(g_fake_data) # error값
        print(" Values : %s" % (str(values)))
        plt.hist(values, bins = 30)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title("histogram of Generated dist")
        plt.grid(True)
        plt.show()

# run the train method
train()