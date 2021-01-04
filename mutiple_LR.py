from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

global pd_data1
global pd_data2
pd_data1 = pd.read_csv(r'C:\Users\DELL\PycharmProjects\research\Car_Auto_Anwser\data\问题/all1.csv', encoding='utf_8_sig')#原始数表
pd_data2 = pd.read_csv(r'C:\Users\DELL\PycharmProjects\research\Car_Auto_Anwser\data\问题/all2.csv', encoding='utf_8_sig')

def get_all():
    data = []
    for count in range(1,50):
        try:
            with open(fr'C:\Users\DELL\PycharmProjects\research\Car_Auto_Anwser\data\问题\{count}.csv',encoding='utf_8_sig') as csvfile:
                reader = csv.reader(csvfile)
                da = []
                count = 0
                for row in reader:
                    count += 1
                    if count != 1 and count <= 102:
                        da.append([row[i] for i in range(3, 9)])
            data += da
        except:
            pass

    name = ['lenaN', 'entiaN', 'readaaN', 'hotN', 'duraaN', '']
    pd_data = pd.DataFrame(columns=name,data=data)
    pd_data.to_csv(r'C:\Users\DELL\PycharmProjects\research\Car_Auto_Anwser\data\问题\all1.csv')

def build_lr():
    X = pd_data2.iloc[:, [3,6,7,8]]
    y = pd_data1.iloc[:, 6].divide(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=532)  # 选择20%为测试集
    print('训练集测试及参数:')
    print('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n,  y_test.shape={}'.format(X_train.shape,y_train.shape, X_test.shape,y_test.shape))
    linreg = LinearRegression()
    # 训练
    model = linreg.fit(X_train, y_train)
    print('模型参数:')
    print(model)
    # 训练后模型截距
    print('模型截距:')
    print(linreg.intercept_)
    # 训练后模型权重（特征个数无变化）
    print('参数权重:')
    print(linreg.coef_)

    # calculate RMSE
    y_pred = linreg.predict(X_test)
    n = len(y_pred)
    sum_mean = 0
    for i in range(n):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / n)  # 测试级的数量
    print("RMSE by hand:", sum_erro)
    # err = y_test - y_pred
    # x = list(map(abs, err))
    # y_test = y_test.tolist()
    # y_pred = y_pred.tolist()
    # # 1. 创建文件对象
    # f = open('预测结果.csv', 'w',newline='')
    # # 2. 基于文件对象构建 csv写入对象
    # csv_writer = csv.writer(f)
    # # 3. 构建列表头
    # csv_writer.writerow(["原值", "预测值", "误差"])
    # # 4. 写入csv文件内容
    # for i in range(n):
    #     csv_writer.writerow([y_test[i], y_pred[i], x[i]])
    # y_test_6 = []
    # y_pred_6 = []
    # for i in range(n):
    #     if y_test.values[i]>=0.6:
    #         y_test_6.append(y_test.values[i])
    #         y_pred_6.append(y_pred[i])
    # n_6 = len(y_test_6)
    draw_error(n,y_pred,y_test,sum_erro)

def draw_roc(n,y_pred,y_test):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # 做ROC曲线
    plt.figure(figsize=(15, 5))
    plt.plot(range(n), y_pred, lw=0.5, color='red', label="Predictiveness scores")
    plt.plot(range(n), y_test, lw=0.5, color='blue', label="Marked scores")
    # 添加细节
    plt.title("Predict Result by Multiple Regression", size=10, color='black')
    plt.xlabel('Test Set Number', size=10)
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
    plt.suptitle('多元回归预测误差分布情况')
    plt.title('RMSE='+str(sum_erro))
    # 显示图形
    plt.savefig(r'C:\Users\DELL\mlr.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    build_lr()