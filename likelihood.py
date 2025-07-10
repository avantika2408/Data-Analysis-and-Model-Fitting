import numpy as np
from utilities import svd_inv

from cobaya.likelihood import Likelihood
from cobaya.theory import Theory


class Chi2Like(Likelihood):
    data = None
    cov_mat = None
    auto_scale = True  # <--- enable automatic scaling

    def initialize(self):
        if self.data is None:
            raise Exception('data must be specified in Chi2Like.')
        if self.cov_mat is None:
            raise Exception('cov_mat must be specified in Chi2Like.')

        self.data = np.asarray(self.data)
        self.cov_mat = np.asarray(self.cov_mat)

        if len(self.data.shape) != 1:
            raise Exception('data should be 1-d array in Chi2Like.')
        if len(self.cov_mat.shape) != 2:
            raise Exception('cov_mat should be 2-d array in Chi2Like.')
        if self.cov_mat.shape != (self.data.size, self.data.size):
            raise Exception('Incompatible cov_mat and data sizes in Chi2Like.')

        # Set scale factor based on mean of data
        self.scale = 1.0
        if self.auto_scale:
            self.scale = 1.0 / np.mean(self.data)

        self.data_scaled = self.data * self.scale
        self.cov_mat_scaled = self.cov_mat * (self.scale ** 2)

        self.invcov_mat, self.det_C = svd_inv(self.cov_mat_scaled)

    def get_requirements(self):
        return {'model': None}

    def logp(self, **params_values_dict):
        try:
            model = self.provider.get_model()
            if not np.all(np.isfinite(model)):
                print("Invalid model output:", model)
                return -np.inf
            
            residual = self.data_scaled - (model * self.scale)
            chi2 = np.dot(residual, np.dot(self.invcov_mat, residual))
            
            if not np.isfinite(chi2):
                print("Invalid chi²:", chi2)
                return -np.inf
            
            return -0.5 * chi2
        except Exception as e:
            print(f"Error in logp: {e}")
            return -np.inf
    
class PolyTheory(Theory):
    x = None

    def initialize(self):
        if self.x is None:
            raise Exception("xvals must be specified in PolyTheory.")
        if len(self.x.shape) != 1:
            raise Exception("xvals should be a 1D array.")

        
        
        
    def calculate(self,state, want_derived=False, **params_values_dict):
        keys = list(params_values_dict.keys())
        params = np.array([params_values_dict[key] for key in keys])
        output = np.sum(np.array([params[p]*self.x**p for p in range(len(keys))]),axis=0)
        # parameter dictionary dynamically decides degree of polynomial
        state['model'] = output
    #########################################

    #########################################
    def get_model(self):
        return self.current_state['model']
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True


class SinExpoTheory(Theory):
    xvals = None # expect 1-d array of same size as data input to likelihood
    #########################################
    def initialize(self):
        if self.xvals is None:
            raise Exception('xvals must be specified in GaussTheory.')
        if len(self.xvals.shape) != 1:
            raise Exception('xvals should be 1-d array in GaussTheory.')
    #########################################

    #########################################
    def pSinExpo(self, params):
        ncomp = len(params) // 2
        out = np.zeros_like(self.xvals)

        for n in range(ncomp):
            a = params[2*n]
            b = params[2*n+1]
            

            comp = a*np.sin(self.xvals) + b*np.exp(self.xvals)
            
            out += comp

        return out

    #########################################

    #########################################
    def calculate(self,state, want_derived=False, **params_values_dict):
        keys = list(params_values_dict.keys())
        params = np.array([params_values_dict[key] for key in keys])
        output = self.pSinExpo(params)
        state['model'] = output
    #########################################

    #########################################
    def get_model(self):
        return self.current_state['model']
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True
    #########################################
    

class GaussTheory(Theory):
    xvals = None # expect 1-d array of same size as data input to likelihood
    #########################################
    def initialize(self):
        if self.xvals is None:
            raise Exception('xvals must be specified in GaussTheory.')
        if len(self.xvals.shape) != 1:
            raise Exception('xvals should be 1-d array in GaussTheory.')
    #########################################

    #########################################
    def pGauss(self, params):
        ncomp = len(params) // 3
        out = np.zeros_like(self.xvals)

        for n in range(ncomp):
            A = params[3*n]
            mu = params[3*n+1]
            sigma = params[3*n+2]

        # Ensure sigma is positive and reasonable
            if sigma <= 0 or not np.isfinite(sigma):
                return np.full_like(self.xvals, np.nan)  # Make model invalid

            comp = -0.5 * ((self.xvals - mu) / sigma) ** 2
            comp = A * np.exp(comp) / (np.sqrt(2 * np.pi) * sigma)
            out += comp

        return out

    #########################################

    #########################################
    def calculate(self,state, want_derived=False, **params_values_dict):
        keys = list(params_values_dict.keys())
        params = np.array([params_values_dict[key] for key in keys])
        output = self.pGauss(params)
        state['model'] = output
    #########################################

    #########################################
    def get_model(self):
        return self.current_state['model']
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True
    #########################################
    
class SinTheory(Theory):
    xvals = None # expect 1-d array of same size as data input to likelihood
    #########################################
    def initialize(self):
        if self.xvals is None:
            raise Exception('xvals must be specified in GaussTheory.')
        if len(self.xvals.shape) != 1:
            raise Exception('xvals should be 1-d array in GaussTheory.')
    #########################################

    #########################################
    def pSin(self, params):
        ncomp = len(params) // 2
        out = np.zeros_like(self.xvals)

        for n in range(ncomp):
            a = params[2*n]
            b = params[2*n+1]

            comp = a*np.sin(b*self.xvals)
            out += comp

        return out

    #########################################

    #########################################
    def calculate(self,state, want_derived=False, **params_values_dict):
        keys = list(params_values_dict.keys())
        params = np.array([params_values_dict[key] for key in keys])
        output = self.pSin(params)
        state['model'] = output
    #########################################

    #########################################
    def get_model(self):
        return self.current_state['model']
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True
    #########################################
    
class GaussChebyTheory(Theory):
    xvals = None  # expect 1-d array of same size as data input to likelihood
    
    def initialize(self):
        if self.xvals is None:
            raise Exception('xvals must be specified in GaussChebyTheory.')
        if len(self.xvals.shape) != 1:
            raise Exception('xvals should be 1-d array in GaussChebyTheory.')
    
    def pGaussCheby(self, params):
        """
        Gaussian + Chebyshev background model
        params: [A, mu, sig, a0, a1]
        """

        
        
        # Gaussian component
        A1, mu1, sig1, A2, mu2, sig2,a0, a1 = params  
            # Validate parameters
        if sig1 <= 0 or sig2 <= 0 or not np.all(np.isfinite(params)):
            return np.full_like(self.xvals, np.nan)  # Invalid model
        gauss1 = A1 * np.exp(-((self.xvals - mu1)**2) / (2 * sig1**2))
        gauss2 = A2 * np.exp(-((self.xvals - mu2)**2) / (2 * sig2**2))  # Fixed: Added "-" before exponent

        cheby = a0 + a1 * self.xvals
    
        return gauss1+ gauss2 + cheby
    
    
        # Extract parameters in correct order
    def calculate(self, state, want_derived=False, **params_values_dict):
        params = [
            params_values_dict['a0'],  # A1
            params_values_dict['a1'],  # mu1
            params_values_dict['a2'],  # sig1
            params_values_dict['a3'],  # A2
            params_values_dict['a4'],  # mu2
            params_values_dict['a5'],  # sig2
            params_values_dict['a6'],  # a0 (Cheby)
            params_values_dict['a7']   # a1 (Cheby)
        ]
        state['model'] = self.pGaussCheby(params)
    
    def get_model(self):
        return self.current_state['model']
    
    def get_allow_agnostic(self):
        return True