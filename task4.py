import numpy as np
import csv
import functools

''' Python '''

# read csv
def load_csv(filepath, test=False):
    with open(filepath) as f:
        reader = csv.reader(f)
        if not test:
            rows = [[int(x[0]), int(x[1]), float(x[2])] for x in reader]
        else:
            rows = [[int(x[0]), int(x[1]), int(x[2])] for x in reader]
    return rows


# sum up max and min value of users and items in data
def data_statistics(data):
    max_u = 0
    max_i = 0
    min_u = 10000
    min_i = 10000
    for triple in data:
        u, i = triple[0], triple[1]
        if u >= max_u:
            max_u = u
        if i >= max_i:
            max_i = i
        if u <= min_u:
            min_u = u
        if i <= min_i:
            min_i = i
    return max_u + 1, max_i + 1  # indices of users and items start from 0,therefore +1 to use in matrix


# calculate MAE
def MAE(r, predict_ratings):
    return np.mean(np.absolute(np.array((r) - np.array(predict_ratings))))


# same as final_predic_ratings, but return the MAE
def evalueMAE(testset, lfm):
    testset = np.array(testset)
    u = testset[:, 0].astype(int)
    i = testset[:, 1].astype(int)
    r = testset[:, 2]
    predict_ratings = lfm.forward(u, i)
    predict_ratings = predict_ratings.reshape(-1)
    # round up the ratings
    data = np.array(predict_ratings)
    for i in range(len(data)):
        if data[i] < 1.5:
            data[i] = 1.0
        elif data[i] >= 1.5 and data[i] < 2.5:
            data[i] = 2.0
        elif data[i] >= 2.5 and data[i] < 3.5:
            data[i] = 3.0
        elif data[i] >= 3.5 and data[i] < 4.5:
            data[i] = 4.0
        elif data[i] >= 4.5:
            data[i] = 5.0
    predict_ratings = data
    return MAE(r, predict_ratings)


# same as evalueMAE, return prediction array (can modify)
def final_predic_ratings(testset, lfm):
    testset = np.array(testset)
    u = testset[:, 0].astype(int)
    i = testset[:, 1].astype(int)
    #r = testset[:, 2]
    predict_ratings = lfm.forward(u, i)
    predict_ratings = predict_ratings.reshape(-1)
    data = np.array(predict_ratings)
    for i in range(len(data)):
        if data[i] < 1.5:
            data[i] = 1.0
        elif data[i] >= 1.5 and data[i] < 2.5:
            data[i] = 2.0
        elif data[i] >= 2.5 and data[i] < 3.5:
            data[i] = 3.0
        elif data[i] >= 3.5 and data[i] < 4.5:
            data[i] = 4.0
        elif data[i] >= 4.5:
            data[i] = 5.0
    predict_ratings = data
    return predict_ratings

''' Latent Factor Model class'''
class LFM():
    def __init__(self, max_u, max_i, dim):
        self.p = np.random.uniform(size=(max_u, dim))  # randomly generate user matrix, demension is max_u * dim
        self.q = np.random.uniform(size=(max_i, dim))  # randomly generate item matrix, demension is max_i * dim
        self.bu = np.random.uniform(size=(max_u, 1))  # randomly generate user's bias term
        self.bi = np.random.uniform(size=(max_i, 1))  # randomly generate item's bias term

    # forward propagation
    def forward(self, u, i):
        return np.sum(self.p[u] * self.q[i], axis=1, keepdims=True) + self.bu[u] + self.bi[i]  # predict ratings
    # back propagation, iterate model parameters according to gradient descend method
    def backward(self, r, predict_ratings, u, i, lr, lamda):  # r = actual rating
        # loss function should be the square of difference of actual and predict ratings
        # because of gradient descend, we need to compute the partial derivative of loss function
        # for convenience, compute the difference directly
        loss = r - predict_ratings
        # update gradient
        self.p[u] += lr * (loss * self.q[i] - lamda * self.p[u])
        self.q[i] += lr * (loss * self.p[u] - lamda * self.q[i])
        self.bu[u] += lr * (loss * lamda * self.bu[u])
        self.bi[i] += lr * (loss * lamda * self.bi[i])


''' Data iterator class '''
class DataIter():
    def __init__(self, dataset):
        self.dataset = np.array(dataset)
    # each iteration return batch_size data
    def iter(self, batch_size):
        for _ in range(len(self.dataset) // batch_size):
            np.random.shuffle(self.dataset)  # shuffle data
            yield self.dataset[:batch_size]  # output, batch_size as unit


''' With the following parameters in final train, the MAE on training data got stuck at 0.584~0.586 '''
def final_train(Train_data, Test_data, max_u, max_i):
    epoch = 1000            # number of iterations
    batchSize = 1024000     # batchSize, 1024000 will take more time per epoch (6mins), but higher accuracy
    lr = 0.03               # learning rate, 0.03 will make the MAE decrease faster through epoches
    lamda = 0.1             # lamda = 0.1
    factor_dim = 64         # factor_dim = 64, p matrix is max_u * factor_dim, q matrix is (max_i * factor_dim).T

    lfm = LFM(max_u, max_i, factor_dim)  # construct LFM model
    dataIter = DataIter(Train_data)  # construct a data iterator

    for e in range(epoch):  # start the epoches
        for batch in (dataIter.iter(batchSize)):  # loop through all batches from data iterator
            u = batch[:, 0].astype(int)
            i = batch[:, 1].astype(int)
            r = batch[:, 2].reshape(-1, 1)

            predict_ratings = lfm.forward(u, i)
            lfm.backward(r, predict_ratings, u, i, lr, lamda)
        # print the epoch no. and the MAE on training data
        print('Train_Data:epoch {} | MAE {:.4f}'.format(e, evalueMAE(Train_data, lfm)))
        if e == epoch - 1:  # output the final prediction at the last epoch
            data = np.array(final_predic_ratings(Test_data, lfm))
            results = []
            for i in range(len(Test_data)):
                results.append([Test_data[i][0], Test_data[i][1], data[i], Test_data[i][2]])
            f = open('results-final.csv', 'w', newline='')
            writer = csv.writer(f)
            for i in results:
                writer.writerow(i)
            f.close()
            print('successfully predict！')
        elif e == 100:  # output the prediction after 100 epochs
            data = np.array(final_predic_ratings(Test_data, lfm))
            results = []
            for i in range(len(Test_data)):
                results.append([Test_data[i][0], Test_data[i][1], data[i], Test_data[i][2]])
            f = open('results-100.csv', 'w', newline='')
            writer = csv.writer(f)
            for i in results:
                writer.writerow(i)
            f.close()
            print('successfully predict！')
        elif e == 300:  # output the prediction after 300 epochs
            data = np.array(final_predic_ratings(Test_data, lfm))
            results = []
            for i in range(len(Test_data)):
                results.append([Test_data[i][0], Test_data[i][1], data[i], Test_data[i][2]])
            f = open('results-300.csv', 'w', newline='')
            writer = csv.writer(f)
            for i in results:
                writer.writerow(i)
            f.close()
            print('successfully predict！')
        elif e == 600:  # output the prediction after 600 epochs
            data = np.array(final_predic_ratings(Test_data, lfm))
            results = []
            for i in range(len(Test_data)):
                results.append([Test_data[i][0], Test_data[i][1], data[i], Test_data[i][2]])
            f = open('results-600.csv', 'w', newline='')
            writer = csv.writer(f)
            for i in results:
                writer.writerow(i)
            f.close()
            print('successfully predict！')


''' Main function to start the whole task '''
if __name__ == '__main__':
    ##### data processing
    Train_data = load_csv('comp3208_20m_train_withratings.csv')
    Test_data = load_csv('comp3208_20m_test_withoutratings.csv', test=True)

    train_max_u, train_max_i = data_statistics(Train_data)
    test_max_u, test_max_i = data_statistics(Test_data)

    max_u = train_max_u if train_max_u >= test_max_u else test_max_u
    max_i = train_max_i if train_max_i >= test_max_i else test_max_i

    print('user max id: {}, item max id: {}'.format(max_u - 1, max_i - 1))

    final_train(Train_data, Test_data, max_u, max_i)


'''The following code is unused for submission, 
it's used for self-validating the MAE before a formal submission'''
# split the provided data into training set 9 : validation set 1
def SplitTrainData(Train_data):
    data_num = Train_data.shape[0]
    np.random.shuffle(Train_data)
    trainset = list(Train_data[:data_num // 10 * 9])
    testset = list(Train_data[data_num // 10 * 9:])

    # re-order the trainset and testset according to uesrid and itemid (raising order)
    def cmp(x, y):
        if x[0] != y[0]:
            return x[0] - y[0]
        else:
            return x[1] - y[1]

    trainset.sort(key = functools.cmp_to_key(cmp))
    testset.sort(key = functools.cmp_to_key(cmp))
    trainset = [[int(u), int(i), float(r)] for u, i, r in trainset]
    testset = [[int(u), int(i), float(r)] for u, i, r in testset]

    f = open('trainset.csv', 'w', newline='')
    writer = csv.writer(f)
    for i in trainset:
        writer.writerow(i)
    f.close()

    f = open('testset.csv', 'w', newline='')
    writer = csv.writer(f)
    for i in testset:
        writer.writerow(i)
    f.close()
    return trainset, testset
