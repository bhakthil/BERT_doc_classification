import sys, os, logging, torch, time, configargparse, socket, shutil

#appends current directory to sys path allowing data imports.
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append('/home/azureuser/projects/BERT_doc_classification/bert_document_classification')

from data import load_polarity_for_torch
from bert_document_classification.document_bert import BertForDocumentClassification

log = logging.getLogger()

def _initialize_arguments(p: configargparse.ArgParser, architecture='DocumentBertMaxPool', model_name='bert-base-uncased'):
    p.add('--model_storage_directory', help='The directory caching all model runs')
    p.add('--bert_model_path', help='Model path to BERT')
    p.add('--labels', help='Numbers of labels to predict over', type=str)
    p.add('--architecture', help='Training architecture', type=str)
    p.add('--freeze_bert', help='Whether to freeze bert', type=bool)

    p.add('--batch_size', help='Batch size for training multi-label document classifier', type=int)
    p.add('--bert_batch_size', help='Batch size for feeding 510 token subsets of documents through BERT', type=int)
    p.add('--epochs', help='Epochs to train', type=int)
    #Optimizer arguments
    p.add('--learning_rate', help='Optimizer step size', type=float)
    p.add('--weight_decay', help='Adam regularization', type=float)

    p.add('--evaluation_interval', help='Evaluate model on test set every evaluation_interval epochs', type=int)
    p.add('--checkpoint_interval', help='Save a model checkpoint to disk every checkpoint_interval epochs', type=int)

    #Non-config arguments
    p.add('--cuda', action='store_true', help='Utilize GPU for training or prediction')
    p.add('--device')
    p.add('--timestamp', help='Run specific signature')
    p.add('--model_directory', help='The directory storing this model run, a sub-directory of model_storage_directory')
    p.add('--use_tensorboard', help='Use tensorboard logging', type=bool)

    # clean dirs
    p.add('--clean_run')
    args = p.parse_args()

    args.labels = [x for x in args.labels.split(', ')]

    # Change model storage dir
    args.architecture = architecture
    args.model_storage_directory = './results_{}'.format(architecture.lower())
    args.bert_model_path = model_name

    #Set run specific envirorment configurations

    args.timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + "_{machine}".format(machine=socket.gethostname())
    args.model_directory = os.path.join(args.model_storage_directory, args.timestamp) #directory
    if(args.clean_run=='True'):
        try:
            shutil.rmtree(args.model_storage_directory)
        except OSError as e:
            print("Error: %s : %s" % (args.model_storage_directory, e.strerror))
    os.makedirs(args.model_directory, exist_ok=True)

    #Handle logging configurations
    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(args.model_directory, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)
    log.info(p.format_values())


    #Set global GPU state
    #if torch.cuda.is_available() and args.cuda:
    #    if torch.cuda.device_count() > 1:
    #        log.info("Using %i CUDA devices" % torch.cuda.device_count() )
     #   else:
     #       log.info("Using CUDA device:{0}".format(torch.cuda.current_device()))
        #args.device =  args.device  #'cuda:1'
    #else:
    #   log.info("Not using CUDA :(")
    #    args.dev = 'cpu'

    return args


if __name__ == "__main__":
    torch.cuda.empty_cache()
    base_data_path = '/home/azureuser/cloudfiles/code/Users/bhakthil'
    train_title, train_content, train_labels, \
    validate_title, validate_content, validate_labels,\
    test_title, test_content, test_labels =  load_polarity_for_torch(os.path.join(base_data_path, 'articles_train.tsv'),
														os.path.join(base_data_path, 'articles_validate.tsv'),
              											os.path.join(base_data_path, 'articles_test.tsv'))  

    document_bert_architectures = [
    {'architectuer':'DocumentBertLSTM', 'model':'bert-base-uncased'},
    {'architectuer':'DocumentDistilBertLSTM','model':'distilbert-base-uncased'},
    {'architectuer':'DocumentBertTransformer','model':'bert-base-uncased'},
    {'architectuer':'DocumentBertLinear','model':'bert-base-uncased'},
    {'architectuer':'DocumentBertMaxPool','model':'bert-base-uncased'}]

    for architecture in document_bert_architectures:
        p = configargparse.ArgParser(default_config_files=["/home/azureuser/projects/BERT_doc_classification/bert_document_classification/polarity_train_config.ini"])
        args = _initialize_arguments(p, architecture=architecture['architectuer'], model_name=architecture['model'])
        model = BertForDocumentClassification(args=args)
        model.fit((train_content, train_labels), (validate_content,validate_labels))

    