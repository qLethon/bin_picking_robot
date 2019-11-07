import argparse 
import sys
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('tkagg')

def make_label(log_file):
    f = open(log_file)
    
    lines  = [line.rstrip("\n") for line in f.readlines()]

    iters = []
    loss = []

    for line in lines:
        args = line.split()
        iters.append(int(args[0]))
        loss.append(float(args[1]))

    return iters, loss

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_file",
        help = "path to log file",
        nargs='*'
        )
    args = parser.parse_args()

    fig,ax = plt.subplots()
    for log_file in args.log_file:
        iters, loss = make_label(log_file)
        ax.plot(iters, loss, label=log_file)
    
    ax.set_xlabel('iters')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()

    ticks = range(0,1000,10)
    
    #ax.set_yticks(ticks)
    plt.show(fig)
    
if __name__ == "__main__":
    main(sys.argv)
