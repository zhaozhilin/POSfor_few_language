class DataProcessor:
    '''
       1\读取数据文件，转化为X，Y，mask
       2\构造embeeding
    '''

    def __init__(self,pm):
        self.train_file =pm.train_filename
        self.test_filename = pm.test_filename
        self.outfile = pm.out_filename
        self.maxlength = pm.max_seqLength
        self.token2id_file = './wordembded/vocab_id.txt'   #词典？
        '''
        self.state_list = {'PAD':0,'CD':1,'DT':2,'FW':3,'ID':4,'IN':5,'JJ':6,'JJS':7,
                           'MD':8,'NN':9,'NNP':10,'OD':11,'P':12,'PO':13,'PRD':14,'PRF':15,'PRI':16,
                           'PRL':17,'PRP':18,'RB':19,'SC':20,'SP':21,'SYM':22,'UH':23,'VB':24,'VO':25,'WH':26,'X':27,'Z':28,'CC':29} #从 0/1?开始？
        '''
        self.state_list = {'PAD':0, 'N':1, 'PRA':2, 'TTL':3, 'PVA':4, 'V':5, 'DAN':6, 'ADJ':7, 'DBQ':8, 'ADV':9, 'DAQ':10,
               'PRS':11, 'IAC':12, 'DMN':13, 'IBQ':14, 'NTR':15, 'IAQ':16, 'REL':17, 'CLF':18, 'COJ':19, 'INT':20,
               'PRE':21, 'FIX':22, 'PRN':23, 'NEG':24, 'CNM':25, 'PUNCT':26, 'ONM':27}
    def _get_Ind_train_data(self):
        '''
        功能：从文件中读取数据，训练集，则返回 X,Y
         X   list 类型，[[],[],......]  shape = (句子数，maxseqlent),即 X[i] 为一个用'PAD' 补全的句子  # 注意词的补全标记与词嵌入保持一致
         Y   list 类型，[[],[],......]  shape = (句子数，maxseqlent),即 Y[i] 为一个用'PAD' 补全的句子的标签，与X[i]对相应
        :return: X,Y
        '''
        filename = self.train_file
        X = []    #[['我们','去','吃饭'], [,,],[,,]....   ]
        Y = []    #[['NN','V','CC'],  .....  ]
        #MASK = []  #[[1,   1,  1 ]]
        words = []
        tags = []
        #mask = []
        with open(filename,'r',encoding='utf-8') as F:
            for line in F.readlines():
                if line.strip().split('\t') == ['']:
                    # X,Y append
                    # words, tags 清空
                    assert len(words) == len(tags)
                    X.append(words)
                    Y.append(tags)
                    #MASK.append(mask)
                    words = []#不能用clear(),会出错
                    tags = []
                    #mask = []
                else:
                    word, tag = line.strip().split('\t')
                    words.append(word)
                    tags.append(tag)
                    #mask.append(1)
        assert len(X) == len(Y)
        return X,Y
    def _get_Lao_train_data(self):
        '''
        功能：从文件中读取数据，训练集，则返回 X,Y
         X   list 类型，[[],[],......]  shape = (句子数，maxseqlent),即 X[i] 为一个用'PAD' 补全的句子  # 注意词的补全标记与词嵌入保持一致
         Y   list 类型，[[],[],......]  shape = (句子数，maxseqlent),即 Y[i] 为一个用'PAD' 补全的句子的标签，与X[i]对相应
        :return: X,Y
        '''
        filename = self.train_file
        X = []    #[['我们','去','吃饭'], [,,],[,,]....   ]
        Y = []    #[['NN','V','CC'],  .....  ]
        #MASK = []  #[[1,   1,  1 ]]
        words = []
        tags = []
        #mask = []
        with open(filename,'r',encoding='utf-8') as F:
            for line in F.readlines():
                line = line.split('\t')
                sentence = line[1].strip().split(' ')
                for s in sentence:
                    if '//' in s:
                        word = '/'
                        tag = 'PUNCT'
                    elif '///' in s:
                        word = '//'
                        tag = 'PUNCT'
                    else:
                        word, tag = s.split('/')
                    words.append(word)
                    tags.append(tag)
                    #mask.append(1)
                X.append(words)
                Y.append(tags)
                #MASK.append(mask)
                words = []
                tags = []
                #mask = []
        return X,Y
    def get_train_data(self,flag = False):
        if flag:
            return self._get_Ind_train_data()
        else:
            return self._get_Lao_train_data()
    def _get_Ind_test_data(self):
       filename = self.test_filename
       X = []  # [['我们','去','吃饭'], [,,],[,,]....   ]
       words = []
       with open(filename, 'r', encoding='utf-8') as F:
           for line in F.readlines():
               if 'id:' in line:
                   continue
               if line.strip() == '':
                   X.append(words)
                   words = []
               else:
                   word = line.strip()
                   words.append(word)
       return X

    def _get_Lao_test_data(self):
       filename = self.test_filename
       X = []  # [['我们','去','吃饭'], [,,],[,,]....   ]
       words = []
       with open(filename, 'r', encoding='utf-8') as F:
           for line in F.readlines():
               words = line.strip().replace('	',' ').split(' ')
               X.append(words[1:])
       return X

    def get_test_data(self,flag = True):
        if flag:
            X = self._get_Ind_test_data()
        else:
            X = self._get_Lao_test_data()
        return X

    def token2id(self,X,Y,word2idx):  #词到词向量   训练集使用函数
        X_id =[]
        Y_id =[]
        MASK = []
        for x,y in zip(X,Y):
            x_id = []
            y_id = []
            mask = []
            for token,label in zip(x,y):
                if word2idx.get(token):
                    x_id.append(word2idx[token])
                else:
                    x_id.append(word2idx['UNK'])
                y_id.append(self.state_list[label])
                mask.append(1)
            lenth = len(x)
            if  lenth < self.maxlength:
                x_id.extend([0]*(self.maxlength - lenth))
                y_id.extend([0]*(self.maxlength - lenth))
                mask.extend([0]*(self.maxlength - lenth))
            assert len(x_id) == len(y_id)
            assert len(x_id) == len(mask)
            assert len(x_id) == self.maxlength
            X_id.append(x_id)
            Y_id.append(y_id)
            MASK.append(mask)
        assert len(X_id) == len(Y_id)
        assert len(X_id) == len(MASK)
        return X_id,Y_id,MASK
    def token2id_for_test(self,X,word2idx):
        X_id = []
        MASK = []
        for x in X:
            x_id = []
            mask = []
            for token in x:
                if word2idx.get(token):
                    x_id.append(word2idx[token])
                else:
                    x_id.append(word2idx['UNK'])
                mask.append(1)
            lenth = len(x)
            if lenth < self.maxlength:
                x_id.extend([0] * (self.maxlength - lenth))
                mask.extend([0] * (self.maxlength - lenth))
            assert len(x_id) == len(mask)
            X_id.append(x_id)
            MASK.append(mask)
        assert len(X_id) == len(MASK)
        #assert len(X_id) == self.maxlength
        return X_id,MASK
    def id2label(self,Y_id):
        '''
        测试模型时使用，将label转化为tag
        :return:
        '''
        id2label_dict = reverse_dict(self.state_list)
        Y = []
        for y_id in Y_id:
            Y.append([id2label_dict[id] for id in y_id if id]) #当id 为0 取消  #看下crf decode 解码形式#######################################
        return Y
    def get_data(self,word2idx,is_train =True,is_Ind =True):
        if is_train:
            X,Y = self.get_train_data()
            return self.token2id(X,Y,word2idx)
        #else:  #
            #X = self.get_test_data(is_Ind)
            #return self.token2id_for_test(X,word2idx)
    def creat_corrector(self):
        word_tag_dict = dict()  # 词  词性标注
        corrector = dict()#dict  词：label
        with open(self.train_file, 'r', encoding='UTF-8') as f:
            for line in f.read().splitlines():
                if len(line) == 0:
                    continue
                wordandtag = line.split("\t")
                word = wordandtag[0]
                tag = wordandtag[1]
                # word_tag_dict
                if word not in word_tag_dict:
                    word_tag_dict[word] = [tag]
                else:
                    if tag in word_tag_dict[word]:
                        continue
                    word_tag_dict[word].append(tag)
        # 单种词性
        for word in word_tag_dict:
            if len(word_tag_dict[word]) == 1:
                corrector[word] = word_tag_dict[word]
        return corrector
    #进行校正 时间复杂度O(样本数*序列长度)   # 争取进一步化简
    def do_correct(self,X,pred,word2idx,corrector):
        idx2word = reverse_dict(word2idx)
        for i in range(len(X)):
            for j in range(len(X[i])):
                if  X[i][j]:
                    word = idx2word[X[i][j]]
                    if corrector.get(word):
                        pred[i][j] = self.state_list[corrector[word][0]]
                else: # 当 x[i][j] 为 0 时，说明到达paddding 进行下一个样本
                    break
    def to_test_file(self,X,Pred):  #[[词，。。。。]]  [labelid,]
        id2state = reverse_dict(self.state_list)
        assert len(X)==len(Pred)
        with open(self.outfile,'w',encoding='utf-8') as f:
            for i,x in enumerate(X):
                assert len(x) == len(Pred[i])
                for word,label_id in zip(x,Pred[i]):
                    f.write('%s\t%s\n'%(word,id2state[label_id]))
                f.write('\n')








def reverse_dict(adict):
    new_dict = {}
    for key, value in adict.items():
        new_dict[value] = key
    return new_dict
