from pickle import dump
import sys

# %% Tool Functions

def save(item,name):
    """Saves a python object into a picke file"""

    file = open(name+'.pickle','wb')
    dump(item,file)
    file.close()

def timer(step,end,message):
    """Prints a progression timer in percentage"""

    sys.stdout.write('\r'+message+str(round(100*(step+1)/end))+' %\t')
    sys.stdout.flush()

def printer(end,message):
    """Prints the begining and the end of a function"""

    sys.stdout.write('\r'+message+'\t')
    sys.stdout.flush()
    if end: sys.stdout.write('\n')