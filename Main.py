import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
from Network import LSTM
from Train import Train
from Test import Test
from Loader import dataset
from torch.utils.data import DataLoader

def plotChart():
    plt.show()
    return

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(args.path)
    df = df.sample(frac = 1)
    dfX_train = df.iloc[:int(len(df)*args.split),0]
    dfY_train = df.iloc[:int(len(df)*args.split),1]
    dataset_ = dataset(dfX_train, dfY_train)
    data_loader = DataLoader(dataset=dataset_, batch_size=1, shuffle=True, num_workers=8)
    print("Carregou o dataset")
    rnn = LSTM()
    losses, rnn, hidden = Train(data_loader, rnn, args.epochs, args.lr, args.batch_size, device)
    plt.plot(losses, color='blue')
    #plotChart()
    dfX_test = df.iloc[int(len(df)*args.split):args.reviews, 0]
    dfY_test = df.iloc[int(len(df)*args.split):args.reviews, 1]
    dataset_ = dataset(dfX_test, dfY_test)
    data_loader = DataLoader(dataset=dataset_, batch_size=1, num_workers=8)
    ax = Test(data_loader, rnn, hidden)
    plotChart()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./IMDB_Dataset.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--reviews', type=int, default=10)
    parser.add_argument('--split', type=float, default=0.8)

    args = parser.parse_args()
    main(args)