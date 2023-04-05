import torch

import random

class DataPool:
    
    def __init__(self, pool_size = 16384, seq_length = 4096, num_alphabet = 256, timesteps = 25):
    
        self.pool_size = pool_size
        self.seq_length = seq_length
        self.num_alphabet = num_alphabet
        self.timesteps = timesteps
        self.data = torch.zeros(timesteps, pool_size, seq_length, dtype = torch.long)
        self.datacount = [0 for i in range(timesteps)]
        self._datapointer = [0 for i in range(timesteps)]

    # def initialize(self, initialize_siz):
        # self.get_and_store_uniform_batch(self.pool_size)

    # Use uniform distribution to fill the pool for time = 0
    def get_and_store_uniform_batch(self, batch_size):
        uniform_batch = torch.randint(low=0, high=self.num_alphabet, size=(batch_size, self.seq_length))
        self.store_data(uniform_batch, 0)
        return uniform_batch

    # Get a batch of data from the pool
    def get_batch(self, batch_size, timestep = -1):

        if timestep < 0:
            # Get random timestep for all timestep has more than batch_size data
            timesteps = [i for i in range(self.timesteps) if self.datacount[i] >= batch_size]
            if len(timesteps) == 0:
                return None
            timestep = random.choice(timesteps)

        # Get batch by composing random data
        rand_indices = torch.randint(low=0, high=self.datacount[timestep], size=(batch_size,))
        batch = self.data[timestep,rand_indices]

        return batch, timestep

    # Stores a batch of data in a FIFO fashion, treating self.data as a circular buffer
    def store_data(self, data_batch, timestep):

        # Reshape data if necessary
        if len(data_batch.shape) == 1:
            data_batch = data_batch.unsqueeze(0)

        batch_size, seq_length = data_batch.shape

        # Compute the indices to store the data
        indices = torch.arange(self._datapointer[timestep], self._datapointer[timestep]+batch_size) % self.pool_size

        # Store the data using tensor indexing
        self.data[timestep, indices] = data_batch

        # Update the datacount and datapointer
        self.datacount[timestep] = min(self.datacount[timestep] + batch_size, self.pool_size)
        self._datapointer[timestep] = (self._datapointer[timestep] + batch_size) % self.pool_size

        # Return indices of stored data
        return indices.tolist()
