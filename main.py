import sys
import getopt
from DRNN_model import train_DRNN
from LSTM_model import train_LSTM

def message():
    print ('main.py --bs <batch_size> \n\
                   --hs <hidden_size>   \n\
                   --nc <num_classes> \n\
                   --ps <padding_size>   \n\
                   --lr <learning_rate> \n\
                   --ep <epochs_of_training> \n\
                   --fs <feature_size> \n\
                   --na <network_architecture> \n\
                   --ol <output_layers> \n\
                   --dr <True: trains DRNN, False: trains LSTM>\n\
                   --td <train_dataset>\n\
                   --vd <validation_dataset>\n\
                   --help \n\
                   See the documentation for more info\n')

def description():
    print ('            --bs: Number of samples to consider in each gradient update\n\
            --hs: Size of the recurrent state   \n\
            --nc: Number of different classes in the dataset file \n\
            --ps: Size to pad input to   \n\
            --lr: Learning rate \n\
            --ep: Number of times the network will iterate the entire dataset \n\
            --fs: Size of the input vector \n\
            --na: A list with the delays of the recurrent units. \n\
                  [1,2,3] will create a network with 3 recurrent units\n\
                  each with 1, 2 and 3 instants of delay respectively\n\
            --ol: Number of output layers. This increases depth \n\
            --dr: If True: trains DRNN  else: trains LSTM> \n\
            --td: Directory of the training dataset file. See documentation for more details \n\
            --vd: Directory of the validation dataset file. See documentation for more details\n\
            See tesis_V1_6_es for more info on the model \n')

def main(argv):
    try:
        opts, args = getopt.getopt(argv,'',["bs=","hs=","nc=","ps=","lr=","ep=",
                                                "fs=","na=","ol=","dr=","td=","vd=","help="])
    except getopt.GetoptError:
        print ("Please insert all parameters correctly")
        message()
        sys.exit(2)

    for opt, arg in opts:
        if opt == "--bs":
            batch_size = int(arg)
        elif opt == "--hs":
            hidden_size = int (arg)
        elif opt == "--nc":
            num_classes = int (arg)
        elif opt == "--lr":
            learning_rate = float(arg)
        elif opt == "--ps":
            padding_size = int (arg)
        elif opt == "--ep":
            num_epochs = int(arg)
        elif opt == "--fs":
            feature_size = int (arg)
        elif opt == "--na":
            net_arch = [int(x) for x in arg.split(',')]
        elif opt == "--ol":
            num_out_layers = int (arg)
        elif opt == "--dr":
            dr = arg
        elif opt == "--td":
            train_dataset = arg
        elif opt == "--vd":
            validation_dataset = arg
        elif opt == "--help":
            description()
            sys.exit(2)

    if len(sys.argv) == 25:

        if (dr=='True'):
            train_DRNN(batch_size = batch_size,
                        hidden_size = hidden_size,
                        num_classes = num_classes,
                        learning_rate = learning_rate,
                        padding_size = padding_size,
                        num_epochs = num_epochs,
                        feature_size = feature_size,
                        net_arch = net_arch,
                        out_layers = num_out_layers,
                        train_dataset = train_dataset,
                        validation_dataset = validation_dataset)
        else:
            train_LSTM(batch_size = batch_size,
                        hidden_size = hidden_size,
                        num_classes = num_classes,
                        learning_rate = learning_rate,
                        padding_size = padding_size,
                        num_epochs = num_epochs,
                        feature_size = feature_size,
                        net_arch = net_arch,
                        out_layers = num_out_layers,
                        train_dataset = train_dataset,
                        validation_dataset = validation_dataset)
    else:
        print ("Missing parameters, please insert all")
        message()

if __name__ == "__main__":
    main(sys.argv[1:])
