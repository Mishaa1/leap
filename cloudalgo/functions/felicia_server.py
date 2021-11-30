"""
FELICIA server implementation with progans
"""
import time
import numpy as np
from tqdm import tqdm
from felicia import privacy_disc

class FeliciaServer():
    """
    Coordinates training of progans on the different FELICIA sites and central privacy discriminator
    """
    def __init__(self):
        """
        Initialize server
        """
        self.clients = []
        self.p_disc = None

    def add_client(self, client):
        """
        Connect a new client to server

        Args:
            client (FeliciaClient)
        """
        self.clients.append(client)

    def setup_training_params(self, pdisc_res_dir, pdisc_delay, pdisc_lambda, batch_size, target_res, train_step_ratio, learning_rate, beta_1, beta_2, epsilon, unique_training_params = False):
        """
        Sets up training params for felicia run
        """
        self.p_disc = privacy_disc.PrivacyDiscriminator(pdisc_res_dir, len(self.clients), resolution=target_res)
        self.start_log2_res = 2
        self.log2_resolution = int(np.log2(target_res))
        self.train_step_ratio = train_step_ratio
        self.batch_size = batch_size
        self.pdisc_delay = pdisc_delay
        self.pdisc_lamba = pdisc_lambda
        
        if unique_training_params:
            # can set batch size, learning rate, beta 1 and 2, and epsilon differently
            for i in range(len(self.clients)):
                self.client[i].set_training_params(batch_size[i], self.start_log2_res, target_res, learning_rate[i], beta_1[i], beta_2[i], epsilon[i])
        else: 
            for client in self.clients:
                client.set_training_params(batch_size, self.start_log2_res, target_res, learning_rate, beta_1, beta_2, epsilon)

    def train(self, steps_per_phase=1000, tick_interval=500):
        """
        Main training loop
        - iterates over clients and starts local training at each site
        - coordinates training of privacy discriminator

        Args:
            steps_per_phase (int, optional): number of steps for each resolution. Defaults to 1000.
            tick_interval (int, optional): step interval to checkpoint model at. Defaults to 500.
        """
        for log2_res in range(self.start_log2_res, self.log2_resolution+1, 1):
            start_time = time.time()
            self.current_log2_res = log2_res

            res = 2**log2_res
            print(f"Resolution {res}x{res}")

            for state in ['TRANSITION', 'STABLE']:
                if state == 'TRANSITION' and log2_res == 2:
                    continue

                steps = int(self.train_step_ratio[log2_res] * steps_per_phase)
                interval = int(self.train_step_ratio[log2_res] * tick_interval)
                for step in tqdm(range(0, steps)):
                    alpha = step/steps if state == 'TRANSITION' else 1.
                    alpha = np.array([[alpha]])

                    dp_train_data = []
                    dp_train_labels = []

                    ''' This becomes for i in range(len(self.clients)) '''
                    for client in self.clients:
                        if step%interval == 0:
                            print('alpha', alpha)
                            elapsed = time.time() - start_time
                            start_time = time.time()
                            minutes = int(elapsed//60)
                            seconds = int(elapsed%60)
                            print(f"elapsed {minutes} min {seconds} sec")
                            
                            ''' 
                            This becomes whatever function leap uses to send from cloud connecter to 
                            site connector
                            - will need to provide information to the cloud connector to identify for each client
                            '''
                            client.checkpoint(log2_res, state, step)

                            if state == "STABLE":
                                self.p_disc.checkpoint(log2_res, step)

                        # train generator and discriminator
                        dp_labels = None
                        if state == 'STABLE':
                            dp_labels = self._get_true_labels_for_gan_dp_loss(client.client_num, self.batch_size[log2_res])

                        client.train_step(alpha, self.p_disc, dp_labels, self.pdisc_lamba)

                        # generate data for privacy discriminator training
                        if state == 'STABLE':
                            fake_batch = client.get_data_for_dp(self.batch_size[log2_res])
                            if len(dp_train_data) == 0:
                                dp_train_data = fake_batch
                                dp_train_labels = [client.client_num-1 for _ in range(self.batch_size[log2_res])]
                            else:
                                dp_train_data = np.vstack((dp_train_data, fake_batch))
                                dp_train_labels += [client.client_num-1 for _ in range(self.batch_size[log2_res])]

                    # train privacy discriminator on stable steps
                    if state == 'STABLE' and step > self.pdisc_delay:
                        self.p_disc.train_step(dp_train_data, np.array(dp_train_labels), alpha)

            # grow models
            if log2_res != self.log2_resolution:
                target_log2_res = log2_res+1
                '''
                call to the cloud connector with site info
                '''
                for client in self.clients:
                    client.grow_model(target_log2_res)

                self.p_disc.grow_model(target_log2_res)

        print("Done training")
        last_step_num = int(self.train_step_ratio[self.current_log2_res] * steps_per_phase) + 1
        '''
        Again here we will need some call to the cloud connector
        '''
        for client in self.clients:
            client.checkpoint(log2_res, "STABLE", last_step_num)
        self.p_disc.checkpoint(log2_res, last_step_num)

    def _get_true_labels_for_gan_dp_loss(self, client_num, batch_size):
        """
        Generate "true" labels for training GAN on privacy discriminator
        labels are random set of values in the range of num_clients, but doesn't include current client being trained

        Args:
            client_num (int): current client being trained
            batch_size (int): number of labels to generate
        """
        possible_labels = [client.client_num-1 for client in self.clients]
        possible_labels.remove(client_num-1)
        return np.random.choice(possible_labels, (batch_size,)).astype("float")
    
