import numpy as np
import csv
import functools

#读取csv文件
def load_csv(filepath,test=False):
    with open(filepath) as f:
        reader = csv.reader(f)
        if not test:
           rows = [[int(x[0]), int(x[1]), float(x[2])] for x in reader]
        else:
            rows = [[int(x[0]), int(x[1]), int(x[2])] for x in reader]

    return rows


#将训练集分为9:1的训练集和验证集
def SplitTrainData(Train_data):
    data_num=Train_data.shape[0]
    np.random.shuffle(Train_data)
    trainset=list(Train_data[:data_num//10*9])         #81513行
    testset=list(Train_data[data_num//10*9:])          #9057行

    #将无序的trainset和testset重新按照uesrid和itemid升序排列
    def cmp(x, y):
        if x[0] != y[0]:
            return x[0] - y[0]
        else:
            return x[1] - y[1]
    trainset.sort(key = functools.cmp_to_key(cmp))
    testset.sort(key = functools.cmp_to_key(cmp))
    trainset=[[int(u),int(i),float(r)] for u, i, r in trainset]
    testset = [[int(u), int(i), float(r)] for u, i, r in testset]

    f = open('trainset.csv', 'w',newline='')
    writer = csv.writer(f)
    for i in trainset:
        writer.writerow(i)
    f.close()

    f = open('testset.csv', 'w',newline='')
    writer = csv.writer(f)
    for i in testset:
        writer.writerow(i)
    f.close()
    return trainset,testset

#统计数据集中user和item的最大最小值
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

        if u<=min_u:
            min_u=u

        if i<=min_i:
            min_i=i
    return max_u + 1, max_i + 1   #因为用户和物品矩阵索引是0开头，+1后可以直接将max_u作为矩阵索引

#计算MAE值
def MAE(r, predict_ratings):
    return np.mean(np.absolute(np.array((r) - np.array(predict_ratings))))

# 该部分代码与下面final_predic_ratings方法一样，只是最后返回值为计算MAE的函数，可删减
def evalueMAE(testset, lfm):
    testset = np.array(testset)
    u = testset[:,0].astype(int)
    i = testset[:,1].astype(int)
    r = testset[:,2]
    predict_ratings = lfm.forward(u, i)
    predict_ratings = predict_ratings.reshape(-1)
    #看看规范为0.5倍数之后，MAE怎么样。结果是MAE变小了，这样处理有好处。
    data=np.array(predict_ratings)
    for i in range(len(data)):
        if data[i] < 1.25:
            data[i] = 1.0
        elif data[i] >= 1.25 and data[i] < 1.75:
            data[i] = 1.5
        elif data[i] >= 1.75 and data[i] < 2.25:
            data[i] = 2.0
        elif data[i] >= 2.25 and data[i] < 2.75:
            data[i] = 2.5
        elif data[i] >= 2.75 and data[i] < 3.25:
            data[i] = 3.0
        elif data[i] >= 3.25 and data[i] < 3.75:
            data[i] = 3.5
        elif data[i] >= 3.75 and data[i] < 4.25:
            data[i] = 4.0
        elif data[i] >= 4.25 and data[i] < 4.75:
            data[i] =4.5
        elif data[i] >= 4.75:
            data[i] = 5.0
    predict_ratings=data
    return MAE(r, predict_ratings)

#该部分代码与上面evalueMAE方法一样，返回值是预测值的数组，可自行修改
def final_predic_ratings(testset, lfm):
    testset = np.array(testset)
    u = testset[:,0].astype(int)
    i = testset[:,1].astype(int)
    r = testset[:,2]
    predict_ratings = lfm.forward(u, i)
    predict_ratings = predict_ratings.reshape(-1)
    #看看规范为0.5倍数之后，MAE怎么样。结果是MAE变小了，这样处理有好处。
    data=np.array(predict_ratings)
    for i in range(len(data)):
        if data[i] < 1.25:
            data[i] = 1.0
        elif data[i] >= 1.25 and data[i] < 1.75:
            data[i] = 1.5
        elif data[i] >= 1.75 and data[i] < 2.25:
            data[i] = 2.0
        elif data[i] >= 2.25 and data[i] < 2.75:
            data[i] = 2.5
        elif data[i] >= 2.75 and data[i] < 3.25:
            data[i] = 3.0
        elif data[i] >= 3.25 and data[i] < 3.75:
            data[i] = 3.5
        elif data[i] >= 3.75 and data[i] < 4.25:
            data[i] = 4.0
        elif data[i] >= 4.25 and data[i] < 4.75:
            data[i] =4.5
        elif data[i] >= 4.75:
            data[i] = 5.0
    predict_ratings=data
    return predict_ratings

class LFM():
    def __init__(self, max_u, max_i, dim):
        self.p = np.random.uniform(size = (max_u, dim))   #  随机生成最初的用户矩阵，维度是max_u * dim
        self.q = np.random.uniform(size = (max_i, dim))   #  随机生成最初的物品矩阵，维度是max_i * dim
        self.bu = np.random.uniform(size = (max_u, 1))    #  随机生成用户的偏置项
        self.bi = np.random.uniform(size = (max_i, 1))    #  随机生成用户的偏置项

    #前项传播
    def forward(self, u, i):
        return np.sum(self.p[u] * self.q[i], axis=1, keepdims=True) + self.bu[u] + self.bi[i]     #预测评分

    #反向传播, 根据梯度下降的方法迭代模型参数
    def backward(self, r, predict_ratings, u, i, lr, lamda): #r 真实评分   pretict——ratings预测评分
        loss = r - predict_ratings        #loss函数本应该是真实评分与预测评分的差的平方，由于梯度下降需要对loss求偏导，为方便，直接算差
        #梯度更新
        self.p[u] += lr * (loss * self.q[i] - lamda * self.p[u])
        self.q[i] += lr * (loss * self.p[u] - lamda * self.q[i])
        self.bu[u] += lr * (loss * lamda * self.bu[u])
        self.bi[i] += lr * (loss * lamda * self.bi[i])

# 批数据迭代器
class DataIter():
    def __init__(self, dataset):
        self.dataset = np.array(dataset)

    #每次迭代返回batch_size个数据
    def iter(self, batch_size):
        for _ in range(len(self.dataset)//batch_size):
            np.random.shuffle(self.dataset)    #  打乱数据
            yield self.dataset[:batch_size]    #  以batch_size为单位输出


def train(trainset,testset, max_u, max_i):
    epoch = 20           #  epoch = 20  训练迭代次数为20次
    batchSize = 1024     #  batchSize = 1024 每一批数量为1024
    lr = 0.01            #  lr = 0.01 学习率为0.01
    lamda = 0.1          #  lamda = 0.1 正则项系数为0.1
    factor_dim = 64      #  factor_dim = 64 隐因子数量为64   p矩阵为max_u * factor_dim    q矩阵为(max_i * factor_dim).T

    #初始化LFM模型
    lfm = LFM(max_u, max_i, factor_dim)
    #初始化批数量提出数据的迭代器
    dataIter = DataIter(trainset)

    for e in range(epoch):
        for batch in (dataIter.iter(batchSize)):
            u = batch[:,0].astype(int)
            i = batch[:,1].astype(int)
            r = batch[:,2].reshape(-1, 1)         #变换形状是为了方便广播计算

            #获取预测分数
            predict_ratings = lfm.forward(u, i)
            #梯度下降迭代
            lfm.backward(r, predict_ratings, u, i, lr, lamda)

        print('trainset:epoch {} | MAE {:.4f}'.format(e, evalueMAE(trainset, lfm)))
        print('testset: epoch {} | MAE {:.4f}'.format(e, evalueMAE(testset, lfm)))

def final_train(Train_data, Test_data, max_u, max_i):
    epoch = 20           #  epoch = 20  训练迭代次数为20次
    batchSize = 1024     #  batchSize = 1024 每一批数量为1024
    lr = 0.01            #  lr = 0.01 学习率为0.01
    lamda = 0.1          #  lamda = 0.1 正则项系数为0.1
    factor_dim = 64      #  factor_dim = 64 隐因子数量为64   p矩阵为max_u * factor_dim    q矩阵为(max_i * factor_dim).T

    #初始化LFM模型
    lfm = LFM(max_u, max_i, factor_dim)
    #初始化批数量提出数据的迭代器
    dataIter = DataIter(Train_data)

    for e in range(epoch):
        for batch in (dataIter.iter(batchSize)):
            u = batch[:,0].astype(int)
            i = batch[:,1].astype(int)
            r = batch[:,2].reshape(-1, 1)         #变换形状是为了方便广播计算

            #获取预测分数
            predict_ratings = lfm.forward(u, i)
            #梯度下降迭代
            lfm.backward(r, predict_ratings, u, i, lr, lamda)

        print('Train_Data:epoch {} | MAE {:.4f}'.format(e, evalueMAE(Train_data, lfm)))
        if e == epoch - 1:
            # print('Test_Data:epoch {} | MAE {:.4f}'.format(e, evalueMAE(Test_data, lfm)))
            data = np.array(final_predic_ratings(Test_data, lfm))
            results = []
            for i in range(len(Test_data)):
                results.append([Test_data[i][0], Test_data[i][1], data[i], Test_data[i][2]])

            f = open('results-2.csv', 'w', newline='')
            writer = csv.writer(f)
            for i in results:
                writer.writerow(i)
            f.close()
            print('successfully predict！')



if __name__ == '__main__':
    #####数据处理
    Train_data = load_csv('comp3208_100k_train_withratings.csv')
    Test_data = load_csv('comp3208_100k_test_withoutratings.csv',test=True)
    #将原本的训练集按9：1重新划分训练集和测试集，只需要调用一次，下次用直接从文件里读取
    trainset, testset=SplitTrainData(np.array(Train_data))
    trainset=load_csv('trainset.csv')
    testset=load_csv('testset.csv')

    train_max_u, train_max_i = data_statistics(Train_data)
    test_max_u, test_max_i = data_statistics(Test_data)

    max_u = train_max_u if train_max_u >= test_max_u else test_max_u
    max_i = train_max_i if train_max_i >= test_max_i else test_max_i
    print('user max id: {}, item max id: {}'.format(max_u - 1, max_i - 1 ))  #之前加1是为了方便作为索引，现在减1还原用户真实数量

    #将(原测试集划分的测试集和验证集进行训练)，得到测试集的MAE和验证集的MAE
    train(trainset, testset, max_u, max_i)

    #最终预测
    final_train(Train_data, Test_data, max_u, max_i)
