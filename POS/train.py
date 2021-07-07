import os
import torch
from parameteres import Parameter as pm
from model import BiLstmCRF
from w2v import Word2vec_my
from data_processor import DataProcessor
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from random import shuffle
from sklearn import metrics

# fake data
'''
x = np.array(np.random.randint(0,pm.vocab_size,pm.max_seqLength*pm.batch_size)).reshape((pm.max_seqLength,pm.batch_size))
x = torch.tensor(x,dtype=torch.long)
y = np.array(np.random.randint(1,pm.num_tags,pm.max_seqLength*pm.batch_size)).reshape((pm.max_seqLength,pm.batch_size))
y = torch.tensor(y,dtype=torch.long)
mask = torch.tensor(np.ones_like(x),dtype=torch.uint8)
print(x.shape)
print(y.shape)
'''


def main():
    device = torch.device('cuda:0')
    W2V = Word2vec_my(pm)
    embeddings_matrix, word2idx = W2V.create_embed_w2i()
    embeddings_matrix = np.array(embeddings_matrix)
    print('##############################begin load data#################################')
    # 加载数据
    dp = DataProcessor(pm)
    X, Y, MASK = dp.get_data(word2idx)  # 返回 idh
    indices = [i for i in range(len(X))]
    shuffle(indices)
    X = np.array(X)[indices,:]
    Y = np.array(Y)[indices,:]
    MASK = np.array(MASK)[indices, :]
    # 非最终模型 划分训练集和验证集 用于调节参数 当选定参数时，将 final 转为 True
    indicate = int(len(X) * 0.9)
    if not pm.final:
        # 先存储测试集实际长度以便后用
        real_length = np.array(MASK, dtype=bool)
        real_length_test = real_length[indicate:]
        #先转化类型 在划分，不用每个都在转化类型
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    MASK = torch.tensor(MASK, dtype=torch.uint8)
    #划分 当pm.final 为 True 时 Y_test 为空
    if not pm.final:
        X_train,Y_train,MASK_train = X[:indicate],Y[:indicate],MASK[:indicate]
        X_test,Y_test,MASK_test  = X[indicate:],Y[indicate:],MASK[indicate:]
    else :
        X_train,Y_train,MASK_train = X,Y,MASK
    #训练数据类型转化
    train_dataset = TensorDataset(X_train,Y_train,MASK_train)
    train_loader = DataLoader(train_dataset, batch_size=pm.batch_size, shuffle=True)  # [batchsize,maxseq]

    print('##############################finish load data#################################')

    # 实例化模型
    model = BiLstmCRF(embeddings_matrix).to(device)  # 采用W2V
    opt = torch.optim.Adam(model.parameters(), lr=pm.lr, weight_decay=pm.w_d)
    ##如果已经训练好，可加载最好的模型继续训练
    if pm.is_trained:
        model.load_state_dict(torch.load(pm.best_model))
    # 保存模型路径
    save_dir = pm.model_save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ## 加载校正器
    corrector = dp.creat_corrector()
    ## 开始训练
    print('##############################begin train#################################')
    for epoch in range(pm.num_epoch):
        model.train()
        for step,(X, Y, MASK) in enumerate(train_loader):
            #####################################################
            # 模型训练i
            X = X.to(device)
            Y = Y.to(device)
            MASK = MASK.to(device)
            #[batchsize,length] -> [length,batchsize]
            loss,pred_ = model(X.t(),Y.t(),MASK.t())
            ####################################################
            # 矫正器 对 pred 进行校正
            #dp.do_correct(X.numpy(),pred,word2idx,corrector)
            ###################################################
            # 测评在训练集的准确率
            labels = Y.data.cpu().numpy()
            pred = np.zeros_like(labels)
            for i, j in enumerate(pred_):
                pred[i][0:len(j)] = j
            real_length = np.array(MASK.data.cpu().numpy(),dtype=bool)
            acc_step = np.sum(labels[real_length] == pred[real_length])/np.sum(real_length)
            if not step % 100 :
                print('epoch:{} step:{} loss:{} acc:{}'.format(epoch, step, loss.item(), acc_step))
            opt.zero_grad()
            loss.backward()
            opt.step()
        ###################################################################
        #每个 epoch 后对模型在测试集上的评估
        if not pm.final:
            model.eval()
            with torch.no_grad():
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)
                MASK_test = MASK_test.to(device)
                loss_valid, p_valid_ = model(X_test.t(), Y_test.t(), MASK_test.t())
                # 矫正器 对 pred 进行校正
                #dp.do_correct(X.numpy(), p_valid, word2idx, corrector)
                y_test = Y_test.data.cpu().numpy()
                p_valid = np.zeros_like(y_test)
                for i, j in enumerate(p_valid_):
                    p_valid[i][0:len(j)] = j
                acc_valid = np.sum(p_valid[real_length_test] == y_test[real_length_test])/np.sum(real_length_test)
            print('epoch:{} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>valid_acc:{}'.format(epoch, acc_valid))
        ####################################################################
        #每10个epoch 保存一次模型
        if not (epoch+1) % 2:
            # 保存模型
            save_model_name = '{}/model_e{}_lr{}.pkl'.format(pm.model_save_path, epoch+1, pm.lr)
            print(save_model_name)
            torch.save(model.state_dict(), save_model_name)


if __name__ == '__main__':
    # 加载词典，embedding参数
    main()


