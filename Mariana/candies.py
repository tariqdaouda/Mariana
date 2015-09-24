import sys, time
import Mariana.settings as MSET

MESSAGE_LOG_FILE = None

def friendly(subject, msg, flush = True) :
	"""Prints a friendly message"""
	
	global MESSAGE_LOG_FILE
	
	m = "  " + msg.replace("\n", '\n  ')
	
	s = """\n%s:\n%s\n%s\n\n  Cheers :),\n\n  Mariana\n""" %(subject, "-"*(len(subject) + 1), m)
	if MSET.VERBOSE :
		print s
		if flush :
			sys.stdout.flush()

	if MSET.SAVE_MESSAGE_LOG :
		if not MESSAGE_LOG_FILE :
			MESSAGE_LOG_FILE = open(MSET.SAVE_MESSAGE_LOG_FILE, "w")
		MESSAGE_LOG_FILE.write("\ntimestamp:%s, human time:%s\n%s" % (time.time(), time.ctime(), s))
		if flush :
			MESSAGE_LOG_FILE.flush()