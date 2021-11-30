# Felicia algorithm
# Copied from fl_fn
# TODO-FELICIA: Change the 8 functions below to match felicia structure

import json
from proto import computation_msgs_pb2
import io
import numpy as np
import pandas as pd
import torch
import ujson as json
import torch.utils.data
import time
# import felicia_client
import tqdm
import felicia_server

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def map_fns():
    # Expects model, dataloader, optimizer, criterion to be predefined
    # in progan.py and felicia.py
    def map_fn_transition(data, state):
        hyperparams["site_id"] = state["site_id"]
        
        dataloader, dataloader_val = get_dataloader(hyperparams, data)
        # for log2_res in range(hyperparams['start_log2_res'], hyperparams['log2_resolution']+1, 1):
        start_time = time.time()
        ## set the site's current resolution
        # state['current_log2_res'] = log2_res

        res = 2**state['log2_res']
        # print(f"Resolution {res}x{res}")

        # for client_state in ['TRANSITION', 'STABLE']:
        if state['log2_res'] == 2:
            return

        steps = int(self.train_step_ratio[state['log2_res']] * steps_per_phase)
        interval = int(self.train_step_ratio[state['log2_res']] * tick_interval)

        for step in tqdm(range(0, steps)):
            alpha = step/steps
            alpha = np.array([[alpha]])

            # dp_train_data = []
            # dp_train_labels = []

            ''' This becomes for i in range(len(self.clients)) '''
            # for client in self.clients:
            if step%interval == 0:
                print('alpha', alpha)
                elapsed = time.time() - start_time
                start_time = time.time()
                minutes = int(elapsed//60)
                seconds = int(elapsed%60)
                # print(f"elapsed {minutes} min {seconds} sec")

                state['site'].checkpoint(state['log2_res'], client_state, step)

                # if state == "STABLE":
                #     self.p_disc.checkpoint(state['log2_res'], step)

            # train generator and discriminator
            '''This needs to be implemented in the server'''
            # if state == 'STABLE':
            #     dp_labels = _get_true_labels_for_gan_dp_loss(client.client_num, self.batch_size[state['log2_res']])

            state['site'].train_step(alpha, state['p_disc'], None, state['pdisc_lambda'])
            
            '''this logic needs to go in stable function'''
            # generate data for privacy discriminator training
            # if state == 'STABLE':
            #     fake_batch = client.get_data_for_dp(self.batch_size[state['log2_res']])
            #     if len(dp_train_data) == 0:
            #         dp_train_data = fake_batch
            #         dp_train_labels = [client.client_num-1 for _ in range(self.batch_size[state['log2_res']])]
            #     else:
            #         dp_train_data = np.vstack((dp_train_data, fake_batch))
            #         dp_train_labels += [client.client_num-1 for _ in range(self.batch_size[state['log2_res']])]
            '''this logic needs to go in stable function'''
            # # train privacy discriminator on stable steps
            # if state == 'STABLE' and step > self.pdisc_delay:
            #     self.p_disc.train_step(dp_train_data, np.array(dp_train_labels), alpha)

        # grow models

        ''''this logic needs to go in update function'''
        # if log2_res != self.log2_resolution:
        #     target_log2_res = state['log2_res']+1
        #     '''
        #     call to the cloud connector with site info
        #     '''
        '''this logic needs to go in stable function'''
        #     for client in self.clients:
        #         client.grow_model(target_log2_res)

        #     self.p_disc.grow_model(target_log2_res)
        result = {'site_id': state['site_id']}
        fake_batch = state['site'].get_data_for_dp(hyperparams['batch_size'][state['log2_res']])
        # if len(state['dp_train_data']) == 0:
        result['dp_train_data'] = fake_batch
        result['dp_train_labels'] = [state['site_id']-1 for _ in range(hyperparams['batch_size'][state['log2_res']])]
        
        '''This logic needs to go in server'''
        # else:
        #     dp_train_data = np.vstack((dp_train_data, fake_batch))
        #     dp_train_labels += [state['site'].client_num-1 for _ in range(hyperparams['batch_size'][state['log2_res']])]

        # result['dp_train_labels'] = dp_train_labels
        # result['dp_train_data'] = dp_train_data
        print("Done training client", state['site_id'])
        
        '''
        Checkpointing logic is second-order priority
        '''
        # last_step_num = int(self.train_step_ratio[self.current_log2_res] * steps_per_phase) + 1
        # for client in self.clients:
        #     client.checkpoint(log2_res, "STABLE", last_step_num)
        # self.p_disc.checkpoint(log2_res, last_step_num)

        return result
    
    def map_fn_stable(data, state):
        hyperparams["site_id"] = state["site_id"]
        
        dataloader, dataloader_val = get_dataloader(hyperparams, data)
        # for log2_res in range(hyperparams['start_log2_res'], hyperparams['log2_resolution']+1, 1):
        start_time = time.time()
        steps = int(state['train_step_ratio'][state['log2_res']] * state['steps_per_phase_stable'])
        interval = int(state['train_step_ratio'][state['log2_res']] * state['tick_interval'])

        '''need to call map_fn_stable #(steps) in server'''
        # for step in tqdm(range(0, steps)):
        alpha = 1
        alpha = np.array([[alpha]])

        print('alpha', alpha)
        elapsed = time.time() - start_time
        start_time = time.time()
        minutes = int(elapsed//60)
        seconds = int(elapsed%60)
        print(f"elapsed {minutes} min {seconds} sec")

        state['site'].train_step(alpha, state['p_disc'], state['dp_labels'], state['pdisc_lambda'])
        result = {'site_id': state['site_id']}

        # generate data for privacy discriminator training
        fake_batch = state['site'].get_data_for_dp(state['batch_size'][state['log2_res']])
        result['dp_train_data'] = fake_batch
        result['dp_train_labels'] = [state['site_id']-1 for _ in range(hyperparams['batch_size'][state['log2_res']])]

        # grow models
        '''Will need to check this in the server logic'''
        # if log2_res != self.log2_resolution:

        print("Done training")
        return result

    def map_fn_grow(state, data):
        state['site'].grow_model(state['target_log2_res'])
        return True

    return [map_fn_transition, map_fn_stable, map_fn_grow]


def agg_fns():
    def agg_fn1(map_results):
        first_result = json.loads(map_results[0].response)
        agg_grad = torch.load(io.BytesIO(map_results[0].grad))
        agg_grad[0] = agg_grad[0] / len(map_results)
        
        loss_meter = AverageMeter()
        loss_meter.update(first_result['loss'])
        
        for i in range(1, len(map_results)):
            grad_result = torch.load(io.BytesIO(map_results[i].grad))
            
            for j in range(len(agg_grad)):

                agg_grad[j] = (agg_grad[j] + (grad_result[j] / len(map_results)))

        for j in range(len(agg_grad)):
            agg_grad[j] = agg_grad[j]
            
        loss_meter.update(first_result['loss'])
        result = {
            "grad": agg_grad,
            "loss":loss_meter.avg
        }
        
        return result

    return [agg_fn1]

def update_fns():
    # Expects model and optimizer in global state
    def update_fn1(agg_result, state):
        state["i"] += 1
        if "loss_history" in state:
            state["loss_history"].append(agg_result["loss"])
        else:
            state["loss_history"] = [agg_result["loss"]]

        # update model weights
        agg_grad = agg_result["grad"]
        for i, (name, params) in enumerate(model.named_parameters()):
            if params.requires_grad:
                params.grad = torch.tensor(agg_grad[i])
            optimizer.step()
            optimizer.zero_grad()

        model_weights = []

        for name, params in model.named_parameters():
            model_weights.append(params)

        request = computation_msgs_pb2.MapRequest()
        req_body = {} 
        req_body["state"] = state
        request.req = json.dumps(req_body)
        buff = io.BytesIO()
        torch.save(model_weights, buff)
        buff.seek(0)
        request.model_weights = buff.getvalue()
        return state

    return [update_fn1]

# Returns which map/agg fn to run
def choice_fn(state):
    return 0

# Formats the raw data into data usable by map_fn
# ex: Converting types, extracting rows/columns
def dataprep_fn(data):
    return data
    #data = pd.DataFrame(data)
    #X = data[].astype('float').to_numpy()
    #Y = data["grade"].astype('long').to_numpy()
    #return X, Y

def stop_fn(agg_result, state):
    return state["i"] == hyperparams["max_iters"]

def postprocessing_fn(agg_result, state):
    return agg_result["loss"]

def init_state_fn():
    state = {
        "i": 0,
    }
    return state
