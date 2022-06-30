import pickle
import torch
import torch.nn as nn

from time import time

from copy import deepcopy
from pprint import pprint

from executor import Executor
from simulation import Simulation

from matplotlib import pyplot as plt

Thr = 100  # threshold
Sub = 1  # sub-threshold activation OR inhibition

O_attrs = { # Object attributes
    'shape': ['cube', 'cylinder', 'sphere'],
    'material': ['rubber', 'metal'],
    'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
}

Neurons = [ # O( o*o_fun + e*e_fun )
# Declare neurons with local codes, making their activations explainable.
    
  ## object neurons
    # universal neurons, O(1) to objects
    'objects', # init objects
    
    'shape', 'material', 'color', # attribute type gates
    
    'cube', 'cylinder', 'sphere', # shape
    'rubber', 'metal', # metarial
    'gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow', #  color
    
    'count', # counting neuron
    'belong', 'belong_gate', # belong-to neurons
    
    
    # basic gates: excitatory and inhibitory
    'o_ex_gate', 'o_inh_gate', 
    # functional gates
    'o_copy_fun', 'o_inh_fun', 'o_mov_fun', 'o_sta_fun', 'o_att_fun',
        
    # atteched neurons, O(n) to objects
    'o',    # objects
    'o_copy',   # copiers
    'o_inh',    # self-inhibition
    'o_mov',    # visible and moving
    'o_sta',    # visible and stationary
    'o_att',    # attended events

    
  ## event neurons
    # universal neurons, O(1) to events
    'events', # init events
    
    'start', 'end', # event types
    
    'e_ex_gate', 'e_inh_gate',
    'e_copy_fun',
    'e_in_fun', 'e_out_fun', 'e_col_fun',
    'e_ancestor_fun',
    'e_before_fun', 'e_after_fun', 'e_first_fun', 'e_second_fun', 'e_last_fun',
     
    # atteched neurons, O(n) to events
    'e',    # events
    'e_copy',     # copiers
    'e_att',      # attended objects
    'e_ancestor', # ancestor
    'e_before', 'e_after', # ordering
   
]
    
__all__ = ['SRN']


class SRN(nn.Module):
    def __init__(self, scene_path, debug = False):
        super().__init__()
        
        # settings
        self.path = scene_path
        self.type = torch.int16
        self.decay = 0
        self.threshold = Thr
        self.debug = debug
        
        # scene --> event sequence --> connection
        sim = Simulation(scene_path, use_event_ann= True)
        self.executor = Executor(sim)
        seq = self.scene2sequence(self.executor)
        con, self.N = self.sequence2connection(seq)

        self.c = nn.Parameter(con.transpose(0,1), requires_grad=False)

           
            
    def scene2sequence(self, executor, split='normal'):        
        with open(self.path, 'rb') as f:
            anno = pickle.load(f)
        seq = {}
        seq['objects'] = anno['objects']
        seq['events'] = []
        
        if split == 'normal':
            for event in executor.events()[:20]: # clip at most 20 events
                event['moving'] = executor.filter_moving(executor.objects(), event['frame'])
                event['stationary'] = executor.filter_stationary(executor.objects(), event['frame'])
                if 'object' not in event.keys():
                    event['object'] = []
                elif len(event['object']) == 2:
                    event['moving'] = list(set(event['moving']+event['object']))
                seq['events'].append(event)
        
        elif split == 'predictive': ### unseen
            for event in executor.unseens[:5]:
                if event['type'] == 'collision':
                    event['moving'] = []
                    event['stationary'] = []
                    seq['events'].append(event)
        
        elif split in executor.objects(): ### counter factual
            for event in executor.counterfact_events(split)[:5]:
                if event['type'] == 'collision':
                    event['moving'] = []
                    event['stationary'] = []
                    seq['events'].append(event)
            
#         else:
#             print('####### scene2sequence type error ######')
#             print(split)
        
        if self.debug:
            pprint(seq)
        
        
        return seq
 

    def sequence2connection(self, seq):

    # connection[][]
        num_objects = len(seq['objects'])
        num_events = len(seq['events'])
        num_neurons = len(Neurons) + 6*(num_objects-1) + 6*(num_events-1) # number of neurons

        c = torch.zeros( (num_neurons,num_neurons), dtype = self.type) # neural connections

    # neural indices
        N = {}
        tmp = Neurons.index('events')
        for i in range(len(Neurons)):
            if i < tmp:   
                N[ Neurons[i] ] = i
            else:
                N[ Neurons[i] ] = i + 6*(num_objects-1)

    ##########  general connections O( o*o_fun + e*e_fun  + e*e )  ##########
     # object related
        # object attributes
        for key in O_attrs:
            for attr in O_attrs[key]:
                c[N[key]] [N[attr]] = Thr-Sub # gates

        # object functions
        for o in range(num_objects):
            # basic        
            c[N['objects']]   [N['o']+6*o] = Thr # init
            c[N['o']+6*o]     [N['o']+6*o] = Thr # hold
            c[N['o_ex_gate']] [N['o']+6*o] = Thr-Sub # excitatory gate
            c[N['o_inh_gate']][N['o']+6*o] = -Sub # inhibitory gate
            c[N['o']+6*o]     [N['count']] = Sub # count neuron

            # copiers
            c[N['o_copy_fun']]  [N['o_copy']+6*o] = Thr-Sub # gates
            c[N['o']+6*o]       [N['o_copy']+6*o] = Sub # channels
            c[N['o_copy']+6*o]  [N['o_copy']+6*o] = Thr # hold
            c[N['belong_gate']] [N['o_copy']+6*o] = -Sub # belong-to gate
            c[N['o_att']+6*o]   [N['o_copy']+6*o] = Sub # belong-to channels
            c[N['o_copy']+6*o]  [N['belong']] = Sub # belong-to neuron

            # self-inhibition
            c[N['o_inh_fun']] [N['o_inh']+6*o] = Thr-Sub # gates
            c[N['o']+6*o]     [N['o_inh']+6*o] = Sub # channels
            c[N['o_inh']+6*o] [N['o']+6*o] = -Thr # back-channels

            # moving
            c[N['o_mov_fun']] [N['o_mov']+6*o] = Thr-Sub # gates
            c[N['o_mov']+6*o] [N['o']+6*o] = Sub # back-channels

            # stationary
            c[N['o_sta_fun']] [N['o_sta']+6*o] = Thr # gates
            c[N['o_sta']+6*o] [N['o']+6*o] = Sub # back-channels

            # attended event
            c[N['o_att_fun']] [N['o_att']+6*o] = Thr-Sub #  gates
            c[N['o_att']+6*o] [N['o']+6*o] = Sub # back-channels

     # event related
        # event functions
        for e in range(num_events):
            # basic
            c[N['events']]    [N['e']+6*e] = Thr # init
            c[N['e']+6*e]     [N['e']+6*e] = Thr # hold
            c[N['e_ex_gate']] [N['e']+6*e] = Thr-Sub # excitatory gate
            c[N['e_inh_gate']][N['e']+6*e] = -Sub # inhibitory gate
            c[N['e']+6*e]     [N['count']] = Sub # count neuron

            # copiers
            c[N['e_copy_fun']] [N['e_copy']+6*e] = Thr-Sub # gates
            c[N['e']+6*e]          [N['e_copy']+6*e] = Sub # channels
            c[N['e_copy']+6*e]     [N['e_copy']+6*e] = Thr # hold
            c[N['belong_gate']]    [N['e_copy']+6*e]= -Sub # belong-to gate
            c[N['e_copy']+6*e]     [N['belong']] = Sub # belong-to neuron

            # attended object
            c[N['e_att']+6*e] [N['e']+6*e] = Sub # back-channels

            # ancestor functions
            c[N['e_ancestor_fun']] [N['e_ancestor']+6*e] = Thr-Sub # gates
            c[N['e_ancestor']+6*e]     [N['e']+6*e] = Sub # back-channels

            # ordering functions
            c[N['e_before_fun']] [N['e_before']+6*e] = Thr # gates
            c[N['e_first_fun']]  [N['e_before']+6*e] = Thr+Sub 
            c[N['e_second_fun']] [N['e_before']+6*e] = Thr+Sub*2
            c[N['e_after_fun']]  [N['e_after']+6*e] = Thr
            c[N['e_last_fun']]   [N['e_after']+6*e] = Thr+Sub
            c[N['e_before']+6*e]     [N['e']+6*e] = Sub # back-channels
            c[N['e_after']+6*e]      [N['e']+6*e] = Sub
            for ee in range(e+1):
                c[N['e']+6*ee] [N['e_before']+6*e] = -Sub # channels
                c[N['e']+6*e]  [N['e_after']+6*ee] = -Sub


    ##########  special connections O( o*o_attr + e )  ##########
        # static
        for o, obj in enumerate(seq['objects']):
            for key in O_attrs.keys():
                c[N['o']+6*o]  [N[obj[key]]]= Sub # to attributes
                c[N[obj[key]]] [N['o']+6*o] = Sub # from attributes

        # dynamic
        for e, eve in enumerate(seq['events']):
            # types
            if eve['type'] in ['start', 'end']:
                c[N[eve['type']]] [N['e']+6*e] = Sub

            # attended objects
            for obj in eve['object']:
                c[N['e']+6*e]   [N['o_att']+6*obj] = Sub # event to object
                c[N['o']+6*obj] [N['e_att']+6*e] = Sub # object to event

            if eve['type'] == 'in':
                c[N['e_in_fun']] [N['e_att']+6*e] = Thr-Sub # in-gate
            elif eve['type'] == 'out':
                c[N['e_out_fun']] [N['e_att']+6*e] = Thr-Sub # out-gate
            elif eve['type'] == 'collision':
                c[N['e_col_fun']] [N['e_att']+6*e] = Thr-Sub # collision-gate

            # moving and stationary
            for obj in eve['moving']:
                c[N['e']+6*e] [N['o_mov']+6*obj] = Sub
            for o in range(num_objects):
                c[N['e']+6*e] [N['o_sta']+6*o] = -Sub # inhibitory stationary filter
            for obj in eve['stationary']:
                c[N['e']+6*e] [N['o_sta']+6*obj] = 0

        # ancestor
        square = []
        for e in range(num_events):
            line = []
            for ee in range(e):
                for obj in seq['events'][e]['object']:
                    if obj in seq['events'][ee]['object']:
                        line.append(ee)
                        line.extend(square[ee])
            square.append(line)

        for e, line in enumerate(square):
            for ee in line:
                c[N['e']+6*e] [N['e_ancestor']+6*ee] = Sub

        return c, N


    
    
    def forward(self, _program):
        program = deepcopy(_program)
        mat = self.c
        N = self.N
        
        # program --> instruction --> stimuli
        if 'unseen_events' in program:
            program[0] = 'events'
            program = program[:-2]+['unseen']
            seq = self.scene2sequence(self.executor, split='predictive')
            con, N = self.sequence2connection(seq)
            mat = con.transpose(0,1)        
        
        elif 'filter_counterfact' in program:            
            program[0] = 'events'
            begin = program.index('all_events')
            end = program.index('filter_counterfact')
            
            # the counterfactual object
            p0 = program[begin:end]
            x0 = self.program2stimuli(p0, N, mat)
            mem0 = torch.zeros(x0[0].shape, dtype = x0.dtype) 
            spike0 = torch.zeros(x0[0].shape, dtype = x0.dtype)
            
            for xt in x0:
                mem0 = mem0 * self.decay * (1 - spike0) \
                            + xt + torch.mm(mat, spike0)
                spike0 = mem0.ge(self.threshold).to(x0.dtype)
            remove = -1
            for o in self.executor.objects():
                if mem0[N['o']+6*o]:
                    remove = o
                    
            # counterfactual events
            if program[-1] == 'negate':
                program = program[:begin] + ['counter_negate']
            else:
                program = program[:begin] + ['counter']
            seq = self.scene2sequence(self.executor, split=remove)
            con, N = self.sequence2connection(seq)
            mat = con.transpose(0,1)

        x = self.program2stimuli(program, N, mat) #[time_steps, n_neuron, 1]
        mem = torch.zeros(x[0].shape, device = x.device, dtype = x.dtype) # [N, 1]
        spike = torch.zeros(x[0].shape, device = x.device, dtype = x.dtype) # [N, 1]
        
        a = []
        b = []
        
        for xt in x:
            mem = mem * self.decay * (1 - spike) \
                        + xt + torch.mm(mat, spike)
            spike = mem.ge(self.threshold).to(x.dtype)
            
            
            if self.debug:
                a.append(list(mem[:,0].detach().cpu().numpy()))
                b.append(list(spike[:,0].detach().cpu().numpy()))
        if self.debug:
            plt.figure( figsize = (13,7) )
            plt.subplot(3,1,1)
            plt.imshow(a)
            plt.subplot(3,1,2)
            plt.imshow(b)
            plt.subplot(3,1,3)
            plt.imshow(x)        
        
        
        return mem2pred(mem, program[-1], N)



    
    def to_instructions(self, p):

        instructions = []
        p_idx = len(p)
        while p_idx:
            p_idx = p_idx-1
            if 'filter' in p[p_idx]:
                if p[p_idx].split('_')[1] in ['color', 'material', 'shape', 'order']:
                    p_idx = p_idx-1
                    p[p_idx] = ('filter_' + p[p_idx])
            instructions.append(p[p_idx])

        instructions.reverse()
        p = instructions

        if 'query_frame' in p:
            a = p.index('events')
            b = p.index('query_frame')+1
            p = p[a:b] + p[:a] + p[b:]

        instructions = []
        for a in p:
            if a not in ['unique','query_frame']:
                instructions.append(a)


        if ('filter_stationary' in instructions) or ('filter_moving' in instructions):
            if instructions[0]!='events':
                instructions = ['events'] + instructions
                
        if self.debug:
            print(instructions)
        
        return instructions
    
    

    def program2stimuli(self, programs, N, mat):
        instructions = self.to_instructions(programs)

        num_neurons = mat.shape[0]
        stimuli = [[0]*num_neurons]
        i = len(instructions)
        while i:
            i = i-1
            stimuli.append([0]*num_neurons)
            stimuli.append([0]*num_neurons)

            if instructions[i].startswith('filter'):
                fil = instructions[i].split('_')[-1]
                if fil in (O_attrs['shape']+O_attrs['material']+O_attrs['color']):
                    stimuli[-1][N[fil]] = 2*Thr
                    stimuli[-1][N['o_inh_gate']] = Thr

                elif fil == 'moving':
                    stimuli[-1][N['o_inh_gate']] = Thr
                    stimuli[-1][N['e_inh_gate']] = Thr # forget events
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['o_mov_fun']] = Thr

                elif fil == 'stationary':
                    stimuli[-1][N['o_inh_gate']] = Thr
                    stimuli[-1][N['e_inh_gate']] = Thr # forget events
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['o_sta_fun']] = Thr


                # type
                elif fil in ['start', 'end']:
                    stimuli[-1][N[fil]] = Thr
                    stimuli[-1][N['e_inh_gate']] = Thr

                elif fil == 'in':
                    stimuli[-1][N['e_inh_gate']] = Thr
                    stimuli[-1][N['o_inh_gate']] = Thr # forget objects
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['e_in_fun']] = Thr

                elif fil == 'out':
                    stimuli[-1][N['e_inh_gate']] = Thr
                    stimuli[-1][N['o_inh_gate']] = Thr # forget objects
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['e_out_fun']] = Thr

                elif fil == 'collision':
                    stimuli[-1][N['e_inh_gate']] = Thr
                    stimuli[-1][N['o_inh_gate']] = Thr # forget objects
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['e_col_fun']] = Thr

                # order
                elif fil == 'ancestor':
                    stimuli[-1][N['e_ex_gate']] = Thr
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['e_ancestor_fun']] = Thr
                    stimuli[-1][N['e_inh_gate']] = Thr

                elif fil == 'before':
                    stimuli[-1][N['e_ex_gate']] = Thr
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['e_before_fun']] = Thr
                    stimuli[-1][N['e_inh_gate']] = Thr

                elif fil == 'after':
                    stimuli[-1][N['e_ex_gate']] = Thr
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['e_after_fun']] = Thr
                    stimuli[-1][N['e_inh_gate']] = Thr

                elif fil == 'first':
                    stimuli[-1][N['e_inh_gate']] = Thr
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['e_first_fun']] = Thr

                elif fil == 'last':
                    stimuli[-1][N['e_inh_gate']] = Thr
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['e_last_fun']] = Thr

                elif fil == 'second':
                    stimuli[-1][N['e_inh_gate']] = Thr
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['e_last_fun']] = Thr
                    stimuli.append([0]*num_neurons)
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['e_inh_gate']] = Thr
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['e_second_fun']] = Thr


            elif instructions[i].startswith('query'):
                query = instructions[i].split('_')[-1]
                if query in O_attrs.keys():
                    stimuli.pop()
                    stimuli[-1][N[query]] = Thr

                elif query == 'object':
                    stimuli[-1][N['o_ex_gate']] = Thr # not a filter but an activator
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['o_att_fun']] = Thr

                elif query == 'partner':
                    stimuli[-1][N['o_ex_gate']] = Thr # not a filter but an activator
                    stimuli.append([0]*num_neurons)
                    stimuli[-1][N['o_att_fun']] = Thr
                    stimuli[-1][N['o_inh_fun']] = Thr
                    stimuli[-1][N['o_inh_gate']] = Thr


            elif instructions[i] == 'belong_to':
                stimuli[-1][N['belong_gate']] = Thr
                stimuli.append([0]*num_neurons)
                stimuli[-1][N['o_att_fun']] = Thr


            elif instructions[i] == 'events':
                stimuli[-1][N['events']] = Thr

                if i and instructions[i-1] == 'events':
                    i = i-1
                    stimuli[-1][N['o_copy_fun']] = Thr
                    stimuli[-1][N['e_copy_fun']] = Thr


            elif instructions[i] == 'objects': 
                stimuli[-1][N['objects']] = Thr

        return torch.tensor(stimuli, dtype = self.type).flip(0).unsqueeze(2)

def mem2pred(mem, quest, N):
    if quest == 'count':
        return str(int(mem[N['count']]))

    elif quest == 'exist':
        if mem[N['count']]:
            return 'yes'
        else:
            return 'no'

    elif quest == 'negate':
        if int(mem[N['belong']]):
            return 'wrong'
        else:
            return 'correct'

    elif quest == 'belong_to':
        if int(mem[N['belong']]):
            return 'correct'
        else:
            return 'wrong'

    elif quest == 'unseen' or quest == 'counter':
        if mem[N['count']]:
            return 'correct'
        else:
            return 'wrong'
        
    elif quest == 'counter_negate':
        if mem[N['count']]:
            return 'wrong'
        else:
            return 'correct'
    else:
        for attr in O_attrs[quest.split('_')[1]]:
            if mem[N[attr]]>=Thr:
                return attr
