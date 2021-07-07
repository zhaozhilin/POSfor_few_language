import gensim
from gensim.models import word2vec
import numpy as np
from parameteres import Parameter as pm
from data_processor import DataProcessor

class Word2vec_my():
    def __init__(self, pm):
        self.size = pm.embedding_dim
        self.windows = pm.windows
        self.model_path = pm.w2v_path          # w2v模型保存文件
        self.raw_data = pm.raw_data

    def generate_text(self, X):
        '''
        生成标准格式的文本，一行一句话，写入到txt文件中。
        :param path_corpus: 要保存的路径
        :param X: lsit:[[],[],……]
        :return:
        '''
        with open(self.raw_data, 'w', encoding='utf-8') as f:
            for line in X:
                i = 1
                for word in line:
                    f.write(str(word))
                    if i < len(line):
                        f.write(" ")
                    i += 1
                f.write("\n")

    def word2vec_train(self):
        '''
        将标准格式的文本加载成Word2Vec需要的格式
        训练词向量模型
        保存模型
        :param path_corpus:
        :param path_model:
        :return:
        '''
        sentences = word2vec.Text8Corpus(self.raw_data)
        model = word2vec.Word2Vec(sentences, size=self.size, hs=1, min_count=1, window=self.windows)
        model.save(self.model_path)
        print('w2v成功保存！')

    def create_embed_w2i(self):
        '''
        加载已训练的词向量模型
        将PAD和UNKOWN添加到word2idx中
        初始化embedding权重矩阵
        再利用词向量模型的参数对embedding权重矩阵 重新赋值  （注意：将Unkown随机赋值）
        :param path: 词向量模型保存的地址
        :return: embeddings_matrix, word2idx
        '''
        model = gensim.models.Word2Vec.load(self.model_path)
        word2idx = {'_PAD':0}  # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
        vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
        # 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
        embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 2, model.vector_size))
        print('Found %s word vectors.' % (len(model.wv.vocab.items())+2))

        for i in range(len(vocab_list)):
            word = vocab_list[i][0]
            word2idx[word] = i + 1
            embeddings_matrix[i + 1] = vocab_list[i][1]
        word2idx['UNK'] = len(model.wv.vocab.items())+1
        embeddings_matrix[len(model.wv.vocab.items())+1] = np.random.randn(model.vector_size)
        return embeddings_matrix, word2idx





if __name__ == '__main__':
    DS = DataProcessor(pm)
    X,_,=DS.get_train_data()  #flag 默认为true 即默认处理India data
    model = Word2vec_my(pm)
    model.generate_text(X)
    model.word2vec_train()
    embeddings_matrix, word2idx = model.create_embed_w2i()  # 主函数只需调用此句
    print(len(word2idx))
    print(word2idx)

