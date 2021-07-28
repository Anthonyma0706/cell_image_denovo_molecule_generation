
class CFG:
    # data_dir = '/gxr/fanyang/Molecular-translation/'
    # train_dir = data_dir + 'train/'
    # test_dir = data_dir + 'test/'
    # train_csv_path = data_dir + 'train_labels.csv'
    # test_csv_path = data_dir + 'sample_submission.csv'

    #train_folds_path = './commercial_data/commercial_data_smiles_folds.csv'
    #train_folds_path = './cil_data/df_info_meta_all_v2_sliced.csv'   
    train_folds_path = './split_data/trainset_by_cluster.csv'    
    tokenizer_path = './commercial_data/tokenizer_smiles_new.pth'

    
    device = 'cuda'             ### set gpu or cpu mode
    debug = True              ### debug flag for checking your modify code
    
    gpus = 1 #2 #3            ### gpu numbers
    precision = 16             ### training precision 8, 16,32, etc
    ### total batch size  = 1 for Multinomial Sampling Testset Prediction
    batch_size = 1 #96       ### total batch size  96 for training
    epochs = 12            ### total training epochs
    encoder_lr = 1e-4      ### encode learning rate default 1e-4
    decoder_lr = 1e-4      ### decode learning rate default 4e-4
    min_lr = 1e-6              ### min learning rate
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
#     max_grad_norm =1000         ### 5
    num_workers = 8         ### number workers
    print_freq = 100            ### print log frequency
    
#     decoder_mode = 'lstm'         ## lstm, transformer
#     encoder_name = 'resnet34'
#     image_size = 224            ### input size in training
#     attention_dim=512
#     embed_dim=512
#     encoder_dim = 512
#     decoder_dim = 512
#     max_length = 275 # 275

# for image data configurations:
    
    decoder_mode = 'transformer'
    encoder_name = 'tnt_s_patch16_224'  ##tf_efficientnet_b0_ns  tnt_s_patch16_224
    
    image_channels = 1 #3 # 5
    image_size =  224 #512 #224 #512 #224            ### input size in training
    image_dim = 384
    text_dim = 384
    decoder_dim = 384
    ff_dim = 512
    num_layer = 3
    num_head = 8
    max_length = 97 # 95 + 2 #227 #102 # smiles_length <= 100
    
    dropout = 0.5

    seed = 42
    n_fold = 5
    trn_fold = [0] #[0,1,2,3,4] #[0] # [0,1,2,3,4]
    #fold=0
    train=True
    
    scheduler = 'CosineAnnealingWarmRestarts' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']

#     # ReduceLROnPlateau
#     factor=0.2 # ReduceLROnPlateau
#     patience=4 # ReduceLROnPlateau
#     eps=1e-6 # ReduceLROnPlateau
    
    ## CosineAnnealingLR
    # T_max=4 # CosineAnnealingLR

    ## CosineAnnealingWarmRestarts
    T_0 = 10
    T_mult = 1
