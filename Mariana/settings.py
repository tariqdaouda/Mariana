#Personal tribute to nomad soul
OMICRON_SIGNATURE = ">|\/| /-\ |-> | /-\ |\| /-\>"

VERBOSE = True
SAVE_MESSAGE_LOG = False
SAVE_MESSAGE_LOG_FILE = "Mariana_logs.txt"
if SAVE_MESSAGE_LOG:
    MESSAGE_LOG_FILE = open(SAVE_MESSAGE_LOG_FILE, "w")

AUTOCAST = True #attempt to cast inputs to float32 if theano raises a TypeError

#The random used for initialising weights
RANDOM_SEED = None

SAVE_OUTPUTS_DEFAULT = False
