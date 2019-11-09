from rnn import main as rnn_main
from ffnn import main as ffnn_main
import argparse

parser = argparse.ArgumentParser(description='Model and Parameter Selection')
parser.add_argument('--model', type=str, default='RNN', help='Model to run, either "RNN" or "FFNN"')
parser.add_argument('--embedding', type=int, default=128, help='Embedding dimension size')
parser.add_argument('--hidden', type=int, default=64, help='Hiddem dimension size')
parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--output', type=int, default=5, help='Output dimension size')
# parser.add_argument('--gpu', type=bool, default=False, help='Boolean value representing whether to run on GPU')

args = parser.parse_args()

def main():

    if args.model == 'RNN':
        rnn_main(args.embedding, args.hidden, args.layers, args.epochs, args.output)
    elif args.model == 'FFNN':
        hidden_dim = 32
        number_of_epochs = 10
        ffnn_main(hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)
    else:
        return 'Incompatible model declaration'


if __name__ == '__main__':
    main()
