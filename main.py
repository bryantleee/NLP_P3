from rnn import main as rnn_main
from ffnn import main as ffnn_main
import argparse

parser = argparse.ArgumentParser(description='Model and Parameter Selection')
parser.add_argument('name', type=str, help='Name of the model for saving after training')
parser.add_argument('model', type=str, help='Model to run, either "RNN" or "FFNN"')
parser.add_argument('--embedding', type=int, default=64, help='Embedding dimension size')
parser.add_argument('--hidden', type=int, default=32, help='Hiddem dimension size')
parser.add_argument('--layers', type=int, default=1, help='Number of hidden layers')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

args = parser.parse_args()

def main():
    if args.model == 'RNN':
        rnn_main(args.name, args.embedding, args.hidden, args.layers, args.epochs)
    elif args.model == 'FFNN':
        ffnn_main(args.name, hidden_dim=args.hidden, number_of_epochs=args.epochs)
    else:
        return 'Incompatible model declaration'


if __name__ == '__main__':
    main()
