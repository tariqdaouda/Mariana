import sys, time
import Mariana.settings as MSET

def friendly(subject, msg, flush = True) :
	"""Prints a friendly message"""
	m = "  " + msg.replace("\n", '\n  ')
	
	s = """\n%s:\n%s\n%s\n\n  Cheers :),\n\n  Mariana\n""" %(subject, "-"*(len(subject) + 1), m)
	if MSET.VERBOSE :
		print s
		if flush :
			sys.stdout.flush()

	if MSET.SAVE_MESSAGE_LOG :
		MSET.MESSAGE_LOG_FILE.write("\ntimestamp:%s, human time:%s\n%s" % (time.time(), time.ctime(), s))
		if flush :
			MSET.MESSAGE_LOG_FILE.flush()