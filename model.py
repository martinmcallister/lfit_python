from abc import ABCMeta, abstractmethod
import numpy as np

class Model():
	"""Base (abstract) class for complex models to be fit with emcee
	
	Atttributes:
		attr1 (type): description, fill these in when we know what they are
	"""
	__metaclass__ = ABCMeta
	def __init__(self,plist):
		"""Initialise model.
		
		Args:
			plist (list): List of mcmc_utils.Param objects
		"""
		self.plist = plist
		
	@property
	def lookuptable(self):
		"""Get lookup table for model
		
		A lookup table maps between the model VARIABLE parameters and their names
		"""
		return [p.name for p in self.plist if p.isVar]
		
	@property
	def npars(self):
		return len(self.pars)
		
	@property
	def pars(self):
		"""Model parameters.
		
		This is a list of the Param objects of all the model's variable parameters.
		The setter will update the *current values* of all variable parameters from
		a list of new values.
		"""
		return [p for p in self.plist if p.isVar]
	
	@pars.setter
	def pars(self,value_list):
		# get a list of all variable params
		variable_params = [p for p in self.plist if p.isVar]
		# loop over supplied values and update
		# current value of params
		for i, val in enumerate(value_list):
			variable_params[i].currVal = val
			
	def ln_prior(self):
		"""Return the natural log of the prior probability of this model
		
		If your model has more prior information not captured in the priors of
		the parameters you may have to override this method, or call it using super
		and add to the probability returned
		"""
		lnp = 0.0
		for param in self.pars:
			lnp += param.prior.ln_prob(param.currVal)
		return lnp
		
	@abstractmethod
	def ln_like(self):
		"""Calculate the natural log of the likelihood"""
		pass
		
	@abstractmethod
	def ln_prob(self):
		"""Calculate the natural log of the posterior probability"""
		pass
		
	def getIndex(self,name):
		if not name in self.lookuptable:
			raise Exception("This is not one of the variable parameters in this model")
		return self.lookuptable.index(name)

	def getValue(self,name):
		return self.pars[self.getIndex(name)].currVal
		
	def getParam(self,name):
		return self.pars[self.getIndex(name)]

class CrapCV(Model):
	"""A totally useless model just for illustration purposes"""
	def __init__(self,plist,complex):
		"""CVs need to know whether or not to use a complex BS"""
		#Model.__init__(plist) # this works poorly with multiple inheritance apparently
		super(CrapCV,self).__init__(plist)
		self.complex = True
		
	def ln_like(self,pars):
		return 100
	def ln_prob(self,pars):
		return self.ln_like(pars)+self.ln_prior(pars)
	def ln_prior(self):
		"""This is how you would implement a custom ln_prior"""
		lnp = 0.0
		lnp += super(CrapCV,self).ln_prior()
		# cases where az > q are not allowed
		if self.getValue('az') > self.getValue('q'):
			lnp += -np.inf
		return lnp
		