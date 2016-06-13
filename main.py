from DQN import *
from database import *
from emulator import *
import tensorflow as tf
import numpy as np
import time
from ale_python_interface import ALEInterface
import cv2
from scipy import misc
import gc #garbage colloector

gc.enable()

params = {
    'ckpt_file':None,
    'num_episodes': 250000,
    'rms_decay':0.99,
    'rms_eps':1e-6,
    'db_size': 100000,
    'batch': 32,
    'num_act': 0,
    'input_dims' : [210, 160, 3],
    'input_dims_proc' : [84, 84, 4],
    'episode_max_length': 100000,
    'learning_interval': 1,
    'eps': 1.0,
    'eps_step':1000000,
    'discount': 0.95,
    'lr': 0.0002,
    'save_interval':20000,
    'train_start':100,
    'eval_mode':False
}

class deep_atari:
    def __init__(self, params):
        print("Initializing Module...")
        self.params = params
        self.sess = tf.Session()
        self.DB = database(self.params['db_size'],
                           self.params['input_dims_proc'])
        self.engine = emulator(rom_name='breakout.bin', vis=True)
        self.params['num_act'] = len(self.engine.legal_actions)
        self.build_nets()
        self.Q_global = 0
        self.cost_disp = 0

    def build_nets(self):
        print('Building QNet and Targetnet...')
        self.qnet = DQN(self.params)        

    def start(self):
        print('Start training...')
        cnt = self.qnet.sess.run(self.qnet.global_step)
        print(('Global step = ', str(cnt)))
        local_cnt = 0
        s = time.time()     

        for numeps in range(self.params['num_episodes']):
            self.Q_global = 0

            state_proc = np.zeros((84,84,4));
            state_proc_old = None;
            action = None;
            terminal = None;
            delay = 0;

            state = self.engine.newGame()
            state_resized = cv2.resize(state,(84,110))
            state_gray = cv2.cvtColor(state_resized, cv2.COLOR_BGR2GRAY)
            state_proc[:,:,3] = state_gray[26:110,:]/255.0
            total_reward_ep = 0

            for maxl in range(self.params['episode_max_length']):
                if state_proc_old is not None:
                    self.DB.insert(state_proc_old[:,:,3],reward,action,terminal)

                action = self.perceive(state_proc, terminal)
                if action == None: #TODO - check [terminal condition]
                    break               
                if local_cnt > self.params['train_start'] and local_cnt % self.params['learning_interval'] == 0:
                    bat_s,bat_a,bat_t,bat_n,bat_r = self.DB.get_batches(self.params['batch'])
                    bat_a = self.get_onehot(bat_a)
                    cnt,self.cost_disp = self.qnet.train(bat_s,bat_a,bat_t,bat_n,bat_r)
                if local_cnt > self.params['train_start'] and local_cnt % self.params['save_interval'] == 0:
                    self.qnet.save_ckpt('ckpt/model_'+str(cnt))
                    print('Model saved')
                
                state_proc_old = np.copy(state_proc)            
                state, reward, terminal = self.engine.next(action) #IMP: newstate contains terminal info
                state_resized = cv2.resize(state,(84,110))
                state_gray = cv2.cvtColor(state_resized, cv2.COLOR_BGR2GRAY)
                state_proc[:,:,0:3] = state_proc[:,:,1:4]
                state_proc[:,:,3] = state_gray[26:110,:]/255.0
                total_reward_ep = total_reward_ep + reward
                local_cnt+=1
                #params['eps'] =0.05    
                self.params['eps'] = max(0.1,1.0 - float(cnt)/float(self.params['eps_step']))
                #self.params['eps'] = 0.00001

            sys.stdout.write("Epi: %d | frame: %d | train_step: %d | time: %f | reward: %f | eps: %f " % (numeps,local_cnt,cnt, time.time()-s, total_reward_ep,self.params['eps']))
            sys.stdout.write("| max_Q: %f\n" % (self.Q_global))         
            #sys.stdout.write("%f, %f, %f, %f, %f\n" % (self.t_e[0],self.t_e[1],self.t_e[2],self.t_e[3],self.t_e[4]))
            sys.stdout.flush()
            

    def select_action(self,state):
        if np.random.rand() > self.params['eps']:
            #greedy with random tie-breaking
            Q_pred = self.qnet.sess.run(self.qnet.y, feed_dict = {self.qnet.x: np.reshape(state, (1,84,84,4)),self.qnet.q_t: np.zeros(1) , self.qnet.actions: np.zeros((1,self.params['num_act'])), self.qnet.terminals:np.zeros(1), self.qnet.rewards: np.zeros(1)})[0] #TODO check
            self.Q_global = max(self.Q_global,np.amax(Q_pred))
            a_winner = np.argwhere(Q_pred == np.amax(Q_pred))
            if len(a_winner) > 1:
              return self.engine.legal_actions[a_winner[np.random.randint(0, len(a_winner))][0]]
            else:
              return self.engine.legal_actions[a_winner[0][0]]
        else:
            #random
            return self.engine.legal_actions[np.random.randint(0,len(self.engine.legal_actions))]

    def perceive(self,newstate, terminal):
        if not terminal: 
            action = self.select_action(newstate)
            return action

    def get_onehot(self,actions):
        actions_onehot = np.zeros((self.params['batch'], self.params['num_act']))
        for i in range(len(actions)):
            actions_onehot[i][self.engine.action_map[int(actions[i])]] = 1
        return actions_onehot


if __name__ == "__main__":
    if len(sys.argv) > 1:
        params['ckpt_file'] = sys.argv[1]
    da = deep_atari(params)
    da.start()
