import numpy as np
import pylab
import os
import pickle
# the following pylab command is needed to run on the Google Cloud!
pylab.switch_backend('agg')
#
#
clrs = [np.array([0.000, 1.000, 0.000]), np.array([0.000, 1.000, 1.000]),
        np.array([0.000, 0.000, 1.000]), np.array([0.980, 0.502, 0.447]),
        np.array([1.000, 0.000, 0.000]), np.array([1.000, 0.000, 1.000]),
        np.array([0.000, 1.000, 1.000]), np.array([0.236, 0.123, 0.110])]

class LSTMs():
    
    def __init__(self, path, act, eta):
        self.path = path
        self.act = act
        self.eta = eta
        self.max_val_accs = []
        os.chdir(path)
        
    def draw(self, lstms, epoch):
        
        pylab.figure(1 , figsize=(6,6))
        pylab.subplot(211)
        epc = np.arange(1,epoch+1)
        
        
        for i, lstm in enumerate(lstms):
            lstmp = '%s.p' % lstm
            lstmi = pickle.load(open( lstmp, "rb" ))
            acci = lstmi['acc']
            pylab.plot(epc, acci, c=clrs[i], label=lstm)  
            #max_acci = max(acci)
            #t = lstm,max_acci
        pylab.legend(loc='best')
        pylab.ylabel('training accuracy')
        pylab.xlabel('epoch')
        pylab.title('eta = %.4g - %s'% (self.eta, self.act))
        pylab.grid()
        
        pylab.subplot(212)
        for i, lstm in enumerate(lstms):
            lstmp = '%s.p' % lstm
            lstmi = pickle.load(open( lstmp, "rb" ))
            val_acci = lstmi['val_acc']
            pylab.plot(epc, val_acci, c=clrs[i], label=lstm)
            max_val_acci = max(val_acci)
            t = lstm, max_val_acci
            self.max_val_accs.append(t)
        pylab.ylabel('testing accuracy')
        pylab.xlabel('epoch')
        pylab.grid()
#    
        fig_name = '%s-eta%.4g.png'% (self.act , self.eta)
        pylab.savefig(fig_name)
        
    def get_max(self):
        # for now run draw class first, to get lstms pickle
        print ('eta = %.4g, act = %s' % (eta,act))
        for e in self.max_val_accs:
            print('max val_acc of {} : {}'.format(e[0],e[1]))
                

        

#%%        
act = 'sigmoid'
eta = 2e-3
path = 'YOUR_path_to_the_results_Folder/%s-eta%.4g'% (act , eta)  

lstms = LSTMs(path, act, eta)
#list the binary dot_p files in the folder that need to be plotted

l= ['lstm0', 'lstm6', 'lstm10']
#Number of epochs must match the # in the binary file
epoch = 100
lstms.draw(l, epoch)
lstms.get_max()

