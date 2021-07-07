import torch
from parameteres import Parameter as pm
from model import BiLstmCRF
from w2v import Word2vec_my
from data_processor import DataProcessor
import numpy as np

def main():
    W2V = Word2vec_my(pm)
    embeddings_matrix, word2idx = W2V.create_embed_w2i()
    embeddings_matrix = np.array(embeddings_matrix)
    print('##############################begin load data#################################')
    # 加载数据
    dp = DataProcessor(pm)
    test_X = dp.get_test_data(pm.is_Ind)  #
    X,MASK = dp.token2id_for_test(test_X,word2idx)
    print(X[0],len(X[0]))
    print('##############################finish load data#################################')
    X = torch.tensor(X, dtype=torch.long)
    MASK = torch.tensor(MASK, dtype=torch.uint8)
    # 实例化模型
    model = BiLstmCRF(embeddings_matrix)  # 采用W2V
    ##如果已经训练好，可加载最好的模型继续训练
    if pm.is_trained:
        model.load_state_dict(torch.load('./checkpoints/model_e30_lr0.01.pkl'))
    else:
        print('ERRO:No trained model!')
        exit()
    ## 加载校正器
    corrector = dp.creat_corrector()
    ## 开始测试
    model.eval()
    with torch.no_grad():
        pred = model.decode(X.t(),MASK.t())
    dp.to_test_file(test_X,pred)
    # X,pred 为真实长度
    # 矫正器 对 pred 进行校正
    #dp.do_correct(X.numpy(),pred,word2idx,corrector)
    ###################################################


if __name__ == '__main__':
    # 加载词典，embedding参数
    main()


