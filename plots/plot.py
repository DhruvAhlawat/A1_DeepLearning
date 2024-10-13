import pickle
import matplotlib.pyplot as plt

with open('traindata.pickle', 'rb') as f:
    traindata = pickle.load(f)
def plot_epoch(data, name):
    index = list(range(1, len(data) + 1))
    accuracy, micro_f1, macro_f1 = zip(*data)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(index, accuracy, marker='o', label='Accuracy')
    ax.plot(index, micro_f1, marker='s', label='Micro F1')
    ax.plot(index, macro_f1, marker='^', label='Macro F1')

    ax.set_title('Performance Metrics for ' + name)
    ax.set_xlabel('List Index')
    ax.set_ylabel('Score')
    ax.set_xticks(index)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(name) #The fig should be saved before doing plt.show().
    plt.show()
plot_epoch(traindata['train'], 'train plot');
plot_epoch(traindata['val'], 'val plot');
plot_epoch(traindata['test'], 'test plot');