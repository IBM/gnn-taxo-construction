import pandas as pd
from global_config import Config
from graph2taxo_supervisor import graph2taxoSupervisor

############## Load Data ##################
def load_data():
    # Load labels
    file_name = 'labels_input.pkl'
    (train_labels, val_labels, __, semeval_labels_RL, __) = pd.read_pickle(file_name)
    return train_labels, val_labels, semeval_labels_RL

############## Main ##################
def main():
    # Load Data
    train_labels, val_labels, semeval_labels_RL = load_data()

    # Num of epochs
    epochs = Config.epochs
    # Early Stop
    patience_num = Config.patience_num
    early_F = Config.early_F

    # Train the model
    supervisor = graph2taxoSupervisor()
    for epoch in range(epochs):
        supervisor.train(epoch, train_labels, 'train')

        # Validatiion
        if (epoch + 1) % 1 == 0:
            F_score = supervisor.test(epoch, val_labels, 'val')
            # Early Stop
            if F_score > early_F:
                patience_num -= 1
                if patience_num < 1:
                    break
    # Test
    supervisor.test(epoch, semeval_labels_RL, 'semeval') # Output the average results
    supervisor.test(epoch, semeval_labels_RL, 'sep_semeval') # Output the results of all domains separately

if __name__ == '__main__':
    main()




