import sys

def friendly(subject, msg, flush = True) :
	"""Prints a friendly message"""
	m = "  " + msg.replace("\n", '\n  ')
	
	s = """\n%s:\n%s\n%s\n\n  Cheers :),\n\n  Mariana\n""" %(subject, "-"*(len(subject) + 1), m)
	print s
	if flush :
		sys.stdout.flush()