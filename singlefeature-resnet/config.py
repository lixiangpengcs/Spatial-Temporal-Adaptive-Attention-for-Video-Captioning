from jobman import DD

# the dir where there should be a subdir named 'youtube2text_iccv15'
RAB_DATASET_BASE_PATH = '/mnt/data/guozhao/predatas/MSVD/'
RAB_FEATURE_BASE_PATH = '/mnt/data/guozhao/features/MSVD/ResNet_152/'
# the dir where all the experiment data is dumped.
RAB_EXP_PATH = '/home/lixiangpeng/workspace/videoCaption/result/MSVD/resnet/'

config = DD({
    'model': 'attention',
    'random_seed': 1234,
    # ERASE everything under save_model_path
    'erase_history': True,
    'attention': DD({
        'reload_': False,
        'save_model_dir': RAB_EXP_PATH + 'save_dir/',
        'from_dir': '',
        'dataset': 'youtube2text', 
        'video_feature': 'googlenet',
        'dim_word':512, # 468, # 474
        'ctx_dim':-1,# auto set 
        'dim':512, # lstm dim # 536
        'n_layers_out':1, # for predicting next word        
        'n_layers_init':0, 
        'encoder_dim': 300,
        'prev2out':True, 
        'ctx2out':True, 
        'patience':20,
        'max_epochs':500, 
        'decay_c':1e-4,
        'alpha_entropy_r': 0.,
        'alpha_c':0.70602,
        'lrate':0.0001,
        'selector':True,
        'n_words':20000, 
        'maxlen':30, # max length of the descprition
        'optimizer':'adadelta',
        'clip_c': 10.,
        'batch_size': 64, # for trees use 25
        'valid_batch_size':200,
        # in the unit of minibatches
        'dispFreq':10,
        'validFreq':2000,
        'saveFreq':-1, # this is disabled, now use sampleFreq instead
        'sampleFreq':100,
        # blue, meteor, or both
        'metric': 'everything', # set to perplexity on DVS
        'use_dropout':True,
        'K':28, # 26 when compare
        'OutOf':None, # used to be 240, for motionfeature use 26
        'verbose': True,
        'debug': False,
        }),
    })
