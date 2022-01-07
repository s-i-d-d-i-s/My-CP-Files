import os, sys, pwd
if len(sys.argv) == 2:
	sys.stdout.write('Codeforces Tool (cf) v1.0.0\n')
	sys.exit()
try:
	import pyastyle
except:
	print("AStyle Not Found")

def cleanCode(code):
	return Cleaner(code)


username = pwd.getpwuid(os.getuid())[0]

BASE = f"/Users/{username}/Documents/Coding"




def Cleaner(code):        
	from pcpp import Preprocessor
	from io import StringIO

	class Parser(Preprocessor):
		def __init(self,code):
			super(CmdPreprocessor, self).__init__()
			
	def _demacro(code):
		p = Parser()
		p.parse(code)
		oh = StringIO()
		p.write(oh)
		return oh.getvalue()
	
	
	def removeLine(code):
		def _removeLine(code):
			for space in range(0,10):
				slug = '#'+" "*space + "line"
				if code.find(slug)!=-1:
					s = code.find(slug)            
					e = s+code[s:].find('\n')
					slug = code[s:e]
					code = code.replace(slug,"\n")
					return code
			return None
		while True:
			temp = _removeLine(code)
			if temp==None:
				return code
			code=temp
			
	def removeInclude(code):
		def _removeInclude(code):
			for space in range(0,10):
				slug = '#'+" "*space + "include"
				if code.find(slug)!=-1:
					s = code.find(slug)            
					e = s+code[s:].find('\n')
					slug = code[s:e]
					code = code.replace(slug,"\n")
					return code,slug
			return None,None
		includes = []
		while True:
			temp,slug = _removeInclude(code)
			if temp==None:				
				return code,includes
			includes.append(slug.strip())
			code=temp

	def demacro(code):
		code,includes = removeInclude(code)
		
		code = _demacro(code)
		code = removeLine(code)
		try:
			code = pyastyle.format(code, '--style=java')
		except:
			print("Skipping Formatting")
		includes = '\n'.join(includes)
		code = includes+'\n'+code
		return code
	return demacro(code)


	
src_path = sys.argv[3]+" "+sys.argv[4]
src_code = open(src_path).read()
src_code = cleanCode(src_code)
with open(BASE+"/temp.cpp","w") as f:
	f.write(src_code)
args = sys.argv
args[0] = BASE+"/cf"
args[3] = BASE+"/temp.cpp"
args[4] = args[5]
args = args[:-1]
os.system(' '.join(args))




