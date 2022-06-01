import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class CharRNN(nn.Module):

    def __init__(self, text, tokens, n_hidden=612, n_layers=4,
                 drop_prob=0.5, lr=0.001,batch_size=64,seq_length=160):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.train_on_gpu = torch.cuda.is_available()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.text = text
        self.epochs_trained = 0


        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.data = np.array([self.char2int[ch] for ch in self.text])


    ## TODO: define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        ## TODO: define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))


    def forward(self, x, hidden):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## TODO: Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)

        ## TODO: pass through a dropout layer
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)

        ## TODO: put x through the fully-connected layer
        out = self.fc(out)



        # return the final output and the hidden state
        return out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

    def one_hot_encode(self,arr, n_labels):

        one_hot_array = np.zeros((np.multiply(*arr.shape),n_labels),dtype=np.float32)

        one_hot_array[np.arange(one_hot_array.shape[0]),arr.flatten()] = 1.

        one_hot_array = one_hot_array.reshape((*arr.shape, n_labels))

        return one_hot_array

    def get_batches(self):
        '''Create a generator that returns batches of size
           batch_size x seq_length from arr.

           Arguments
           ---------
           arr: Array you want to make batches from
           batch_size: Batch size, the number of sequences per batch
           seq_length: Number of encoded chars in a sequence
        '''

        batch_size_total = self.batch_size * self.seq_length
        # total number of batches we can make, // integer division, round down
        n_batches = len(self.data)//batch_size_total

        # Keep only enough characters to make full batches
        self.data = self.data[:n_batches * batch_size_total]
        # Reshape into batch_size rows, n. of first row is the batch size, the other lenght is inferred
        self.data = self.data.reshape((self.batch_size, -1))

        # iterate through the array, one sequence at a time
        for n in range(0, self.data.shape[1], self.seq_length):
            # The features
            x = self.data[:, n:n+self.seq_length]
            # The targets, shifted by one
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], self.data[:, n+self.seq_length]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], self.data[:, 0]
            yield x, y

    def train_model(self, epochs=1, clip=5, val_frac=0.1, print_every=10):
        '''
        Training a network

        Arguments
        ---------

        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

        '''
        loss_arr = []
        val_loss_arr = []
        self.epochs_trained+=epochs

        if self.train_on_gpu:
            self.cuda()

        self.train()

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        # create training and validation data
        val_idx = int(len(self.data)*(1-val_frac))
        data, val_data = self.data[:val_idx], self.data[val_idx:]



        counter = 0
        n_chars = len(self.chars)
        for e in range(epochs):
            # initialize hidden state
            h = self.init_hidden(self.batch_size)

            for x, y in self.get_batches():
                counter += 1

                # One-hot encode our data and make them Torch tensors
                x = self.one_hot_encode(x, n_chars)
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                if(self.train_on_gpu):
                    inputs, targets = inputs.cuda(), targets.cuda().type(torch.cuda.LongTensor)

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                self.zero_grad()

                # get the output from the model
                output, h = self(inputs, h)

                # calculate the loss and perform backprop
                loss = criterion(output.cuda(), targets.view(self.batch_size*self.seq_length))
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(self.parameters(), clip)
                opt.step()

                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_h = self.init_hidden(self.batch_size)
                    val_losses = []
                    self.eval()
                    for x, y in self.get_batches():
                        # One-hot encode our data and make them Torch tensors
                        x = self.one_hot_encode(x, n_chars)
                        x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        inputs, targets = x, y
                        if(self.train_on_gpu):
                            inputs, targets = inputs.cuda(), targets.cuda()

                        output, val_h = self(inputs, val_h)
                        val_loss = criterion(output, targets.view(self.batch_size*self.seq_length).type(torch.cuda.LongTensor))

                        val_losses.append(val_loss.item())
                        val_loss_arr.append(val_loss.item())

                    self.train() # reset to train mode after iterationg through validation data

                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.4f}...".format(loss.item()),
                          "Val Loss: {:.4f}".format(np.mean(val_losses)))

                    loss_arr.append(loss.item())
        return (loss_arr,val_loss_arr)