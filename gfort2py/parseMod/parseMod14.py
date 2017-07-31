from . import parseModCommon as pmc


class parseMod(pmc.parseModBase):
	def __init__(self,*args):
		self.mod_version=14
		super(parseMod,self).__init__(*args)


