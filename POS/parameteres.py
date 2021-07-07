class Parameter:
    # model parameters
    num_tags = 30   #标记标签的种类数
    embedding_dim = 150  # 嵌入层维度 = 词向量维度
    hidden_dim = 50 # [maxseq,batchsize]  -> bilstm -> [maxseq,batchsize,hiddendim] ->linear-> [maxseq,batchsize,num_tags]
    max_seqLength = 39 # 统计最长的句子 作为最大长度
    vocab_size = 11979   # Ind_ size: 20326 包含pad _unknow    loatain
    # train parameters
    is_trained = False
    batch_size = 128
    num_epoch = 30
    lr = 0.01
    w_d = 0.01
    final = False
    # file path
    is_Ind = False
    #train_filename = './corpus/Ind_train.txt'
    #train_filename = './corpus/Ind_train.txt'
    train_filename = './corpus/Lao_train.txt'
    #test_filename = './corpus/Ind_test.txt'
    test_filename = './corpus/Lao_test.txt'
    #out_filename = 'Ind_out.txt'
    out_filename = 'Lao_out2.txt'
    model_save_path = './checkpoints/'
    best_model = './checkpoints/model_e18_lr0.01.pkl'
    # word2vec
    windows = 5
    #w2v_path = 'W2V/model_India_w2v.m'
    w2v_path = 'W2V/model_Lao_w2v.m'
    #文件格式： 每行为sentence
    #raw_data = 'W2V/raw_India.txt'
    raw_data = 'W2V/raw_lao.txt'
