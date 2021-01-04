import numpy as np
from sklearn.model_selection import train_test_split  # 训练和测试集分隔
import matplotlib.pyplot as plt
import csv
from sklearn.neural_network import MLPRegressor
import pandas as pd
import random

# 载入数据
global pd_data1
global pd_data2
pd_data1 = pd.read_csv(r'C:\Users\DELL\PycharmProjects\research\Car_Auto_Anwser\data\问题/all1.csv', encoding='utf_8_sig')#原始数表
pd_data2 = pd.read_csv(r'C:\Users\DELL\PycharmProjects\research\Car_Auto_Anwser\data\问题/all2.csv', encoding='utf_8_sig')

def bp_net():
    X = pd_data2.loc[:, ('lenaN', 'entiaN', 'readaaN', 'hotN', 'duraaN', 'sim')]
    y = pd_data1.iloc[:, 6].divide(10)

    # 数据切分，测试集占0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=532)

    #构建模型，隐藏层有50个神经元，训练1000周期
    mlp = MLPRegressor(hidden_layer_sizes=50, max_iter=1000)
    mlp.fit(X_train,y_train)

    # calculate RMSE
    y_pred = mlp.predict(X_test)
    n = len(y_pred)
    sum_mean = 0
    for i in range(n):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / n)  # 测试级的数量
    print("RMSE by hand:", sum_erro)
    # draw_error(n,y_pred,y_test,sum_erro)

def draw_roc(n,y_pred,y_test):
    # 做ROC曲线
    plt.figure(figsize=(15, 5))
    plt.plot(range(n), y_pred, lw=0.5, color='red', label="Predictiveness scores")
    plt.plot(range(n), y_test, lw=0.5, color='blue', label="Marked scores")
    # 添加细节
    plt.title("Predict Error", size=10, color='black')
    plt.xlabel('Index', size=10)
    plt.ylabel('Scores', size=10)
    # plt.axis('tight')
    plt.xlim(-1, n)
    plt.ylim(0, 1)
    # 添加图例
    plt.legend(loc=0)
    plt.show()

def draw_error(n,y_pred,y_test,sum_erro):
    x = list(map(lambda x: abs(x[0] - x[1]), zip(y_pred, y_test)))
    # err = y_test - y_pred
    # x = list(map(abs, err))

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    fig,ax1 = plt.subplots()
    ax1.grid(axis='x', linestyle='-.', alpha=0.6)
    ax2 = ax1.twinx()
    ax2.grid(axis='both', linestyle='-.', alpha=0.6)
    B = [i/20.0 for i in range(0,16,1)]
    # 绘制直方图
    ax1.hist(x=x,color = 'teal',edgecolor = 'silver',bins=B,density=False,rwidth=5)
    ax2.hist(x=x,bins=B,density=True,color='lightgreen',cumulative=True,rwidth=5,histtype='step')
    # 添加x轴和y轴标签
    ax1.set_xlabel('误差')
    ax1.set_ylabel('数量')
    ax2.set_ylabel('累计占比')
    # 添加标题
    plt.suptitle('BP神经网络预测误差分布情况')
    plt.title('RMSE=' + str(sum_erro))
    # 显示图形
    plt.savefig(r'C:\Users\DELL\BP_Net.png', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    bp_net()
