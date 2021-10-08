import os
from pdb import set_trace as stop


def get_args(parser,eval=False): 
    """Gets the arguments for the parser, sets the arguments and tasks, and sets the environment.
    The method initializes the training and validation datasets, saves the models, and creates directories for the models.
    :param parser: Contains the parser variable, has parse_args
    :param eval: Always set to false
    """
    parser.add_argument("-tr", "--train_dataset", type=str, default=None, help="train dataset for train bert")
    parser.add_argument("-va", "--valid_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-te", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="output/bert.model")
    parser.add_argument("-d", "--data_root", type=str, default='./dataset/data/', help="label file for ss task")
    parser.add_argument("--activation", type=str, default='gelu',choices=['relu', 'gelu'])
    parser.add_argument("--task", type=str, default='secondary', help="")
    parser.add_argument('--tasks', nargs='+',default='')
    parser.add_argument("-hs", "--hidden", type=int, default=768, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=12, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-do", "--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("-s", "--seq_len", type=int, default=1024, help="maximum sequence len")
    parser.add_argument("-s2", "--seq_len2", type=int, default=-1, help="maximum sequence len for testing")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="number of batch_size")
    parser.add_argument("--max_batches", type=int, default=-1, help="")
    parser.add_argument("--test_batch_size", type=int, default=-1, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=1, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--dont_overwrite", action='store_true')
    parser.add_argument("--freeze_bert", action='store_true')
    parser.add_argument("--esm", action='store_true')
    parser.add_argument("--reset_weights", action='store_true')
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0, help="weight_decay of adam")
    parser.add_argument("--optimizer", type=str, default='adam',choices=['sgd', 'adam'])
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="warmup_steps")
    parser.add_argument("--name", type=str, default='', help="")
    parser.add_argument("-save_root", "--save_root", type=str, default='./results/', help="")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--test_lm', action='store_true')
    parser.add_argument("--grad_ac_steps", type=int, default=1, help="")
    parser.add_argument("--fold", type=int, default=1, help="")
    parser.add_argument('--use_evo', action='store_true')
    parser.add_argument("--saved_bert", type=str, default='', help="")
    parser.add_argument("--saved_model", type=str, default='', help="")
    parser.add_argument("--emb_type", type=str, default='conv',choices=['lookup','conv','continuous','both','pair'])

    args = parser.parse_args()
    
    args.save_best = True

    if args.debug:
        # args.train_dataset = os.path.join(args.data_root,args.test_dataset)
        args.train_dataset = os.path.join(args.data_root,args.valid_dataset)
        args.valid_dataset = None
        args.test_dataset =  None
        args.batch_size = 4
        args.cuda_devices = [0]
    else:
        args.train_dataset = os.path.join(args.data_root,args.train_dataset) if args.train_dataset is not None else None
        args.valid_dataset = os.path.join(args.data_root,args.valid_dataset) if args.valid_dataset is not None else None
        args.test_dataset = os.path.join(args.data_root,args.test_dataset) if args.test_dataset is not None else None

    args.vocab_path = os.path.join(args.data_root,args.vocab_path)

    if args.seq_len2 == -1:
        args.seq_len2 = args.seq_len

    args.model_name = os.path.join(args.save_root,args.task)
    args.model_name += '.bert'
    args.model_name += '.bsz_'+str(int(args.batch_size*args.grad_ac_steps))
    args.model_name += '.layers_'+str(args.layers)
    args.model_name += '.size_'+str(args.hidden)
    args.model_name += '.heads_'+str(args.attn_heads)
    args.model_name += '.drop_'+("%.2f" % args.dropout).split('.')[1]
    args.model_name += '.lr_'+str(args.lr)#.split('.')[1]

    if args.test_batch_size != -1:
        args.batch_size = args.test_batch_size
    
    if args.use_evo:
        args.model_name += '.use_evo'
        args.emb_type = 'both' # fix


    args.model_name += '.emb_'+args.emb_type


    if args.debug:
        args.model_name += '.debug'

    if args.saved_bert != '':
        args.model_name += '.saved_bert'
    
    if args.reset_weights:
        args.model_name += '.reset_weights'
    

    if args.saved_model != '':
        args.model_name += '.saved_model'

    if args.esm:
        args.model_name += '.esm'

    if args.freeze_bert:
        args.model_name += '.freeze_bert'

    if args.train_dataset:
        if 'h1n1' in args.train_dataset:
            args.model_name += '.h1n1'
        elif 'ebola' in args.train_dataset:
            args.model_name += '.ebola'
        elif 'barman' in args.train_dataset:
            args.model_name += '.barman'
        elif 'denovo' in args.train_dataset.lower():
            args.model_name += '.denovo'

    if args.name != '':
        args.model_name += '.'+args.name

    
    if os.path.isdir(args.model_name) and args.dont_overwrite:
        exit(0)
    if os.path.isdir(args.model_name) and (not args.overwrite) and (not args.debug):
        print(args.model_name)
        overwrite_status = input('Already Exists. Overwrite?: ')
        if overwrite_status == 'rm':
            os.system('rm -rf '+args.model_name)
        elif not 'y' in overwrite_status:
            exit(0)
    if not os.path.exists(args.model_name):
        os.makedirs(args.model_name)

    
    args.evo_size = 30

    return args
