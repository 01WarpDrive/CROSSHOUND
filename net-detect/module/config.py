##### FastText #####
EMBEDDING_EPOCH = 200
EMBEDDING_SIZE = 256


##### VAE #####
EPOCH = 500

DATASET_FILE_MAP = {
    'optc_day23-flow' : {
        'train': 'train_conn_23_0-12.json',
        'test': 'test_conn_23_15-19.json',
        'ecarbro': 'ecarbro_23red_0201.json',
        'FASTTEXT_PATH': './models/FastText.model',
        'VAE_PATH': './models/VAE.model'
    },
    'optc_day24-flow' : {
        'train': 'train_conn_23_0-12.json',
        'test': 'test_conn_24_14-21.json',
        'ecarbro': 'ecarbro_24red_0501.json',
        'FASTTEXT_PATH': './models/FastText.model',
        'VAE_PATH': './models/VAE.model'
    },
    'optc_day25-flow' : {
        'train': 'train_conn_23_0-12.json',
        'test': 'test_conn_25_13-18.json',
        'ecarbro': 'ecarbro_25red_0051.json',
        'FASTTEXT_PATH': './models/FastText.model',
        'VAE_PATH': './models/VAE.model'
    },
    'lanl-flow': {
        'train': 'train.json',
        'test': 'test.json',
        'FASTTEXT_PATH': './models/FastText_lanl_flow.model',
        'VAE_PATH': './models/VAE_lanl_flow.model'
    }
}