import pandas as pd
import numpy as np
import datetime as dt
import dateutil
import utils
import logging
from scipy import stats
from scipy.optimize import minimize
from scipy.special import beta

logger = logging.getLogger(__name__)

class ShiftedBetaGeom():
    """ Implementation of shifted-beta-geometric distribution (sBG) for predicting retention curves

    General Usage:
    - create an instance of sGB class
    - train the model using historical data by running train() method
        - provide historical counts of customers over time in a list format e.g. [1000,900,800,500]
        - recommended minimum is at least 6 periods of data
        - it is important to provide actual customer count. this approach is not be tested with normalized count (ie. 1 or 100)
    - predict retention curve into the future using predict() method
        - provide the number of period you'd like to predict as a 'periods' parameter (detault = 36)
        - the output is provided in the form of the list of values over time (similar to the training data input)
        - however, the output is normalized (starting value is 1)
    - use the retention_plotter() helper function to plot actual and predicted values
        - both actual and predicted values should be passed as list (can be of different lengths)
        - the function will normalize both actual and predicted values before plotting

    Theoratical reference for shifted-beta-geometric distribution:
    https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf

    """

    # data loading function to be called by the train() method
    def load_training_data(self,data):
        ''' Load input data for training. Performs some value checks on the input'''
        if data == []:
            raise ValueError('Data is missing')
        elif not utils.is_numeric(data) :
            raise ValueError('Data is not numeric')
        elif not utils.is_non_increasing(data):
            raise ValueError('Data cannot be increasing')
        elif min(data) < 0:
            raise ValueError('Data cannot be negative')
        elif sum(data) == 0:
            raise ValueError('Data cannot be all zero')
        else:
            self.retained_cust = data #number of customers retained over time. list indexs are assumed to be time periods
            self.time_periods = len(data)
            self.churned_cust = [0] + [data[i-1]-data[i] for i in range(1,self.time_periods)] #number of customers leaving at every time period. list indexs are time periods

    def get_churn_prob_t(self,t:int,alpha,beta):
        '''
        method for calculating churn probability P(T=t) for given alpha and beta - using sBG distribution
        '''
        assert (t >= 0) & (t<=500), 't must be in [0,500]'
        if t == 0:
            return 1
        elif t == 1:
            return alpha/(alpha+beta)
        else:
            return ((beta+t-2)/(alpha+beta+t-1))*self.get_churn_prob_t(t-1,alpha,beta)


    def get_retention_t(self,t,alpha,beta): #assumes self.churn_probs are already set
        '''
        method for geting survival/retention function S(t) - using sBG distribution. assuming alpha and beta
        '''
        if t == 0:
            return 1
        elif t == 1:
            return 1 - self.get_churn_prob_t(t,alpha,beta)
        else:
            return self.get_retention_t(t-1,alpha,beta) - self.get_churn_prob_t(t,alpha,beta)

    def get_ll(self,params):
        '''
        method calculate log likelihood given alpha and beta values as a list params = [alpha,beta]. This will be used by optimizer in train() method
        '''
        alpha = params[0]
        beta = params[1]

        churn_probs = [self.get_churn_prob_t(t,alpha,beta) for t in range(0,self.time_periods)]
        ret_prob = self.get_retention_t(self.time_periods-1,alpha,beta)

        #log of zero or less is inf. So check if probs are zero before calculating max likelihood
        if (any([x <=0 for x in churn_probs])) | (ret_prob <= 0):
            return np.inf
        else:
            log_liklihood_churned = [self.churned_cust[t]*np.log(p) for t,p in zip(range(0,self.time_periods),churn_probs)]
            log_liklihood_survived = self.retained_cust[-1]*np.log(ret_prob)
            return  -1*(sum(log_liklihood_churned) + log_liklihood_survived)

    def train(self,data):
        '''
        estimate alpha and beta parameters by fitting data
        '''

        self.load_training_data(data)

        ### maximim likelihood estimatation of sBG parameters using training data
        #set up
        x0 = [0.1,0.1] #arbitrary starting values
        bnds = [(0,None),(0,None)] # non-negative parameters
        self.train_result = minimize(self.get_ll,x0,method='Nelder-Mead',options={'disp': False})

        #Nelder-Mead method found to be most stable among the options
        #save full optimization results in case need to debug
        #more reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
        #scipy.optimize.OptimizeResult

        self.params_estimate = list(self.train_result.x) #[alpha,beta]
        logger.debug('Params: alpha,beta = {}'.format(self.params_estimate))

    def predict(self,periods=36):
        '''
        estimate survival/retention curve using trained parameter
        '''
        if self.params_estimate == '':
            print('Train the model using train() method')
        else:
            a,b = self.params_estimate
            if beta(a,b) == 0:
                raise ValueError('Check the training data')
            else:
                pred = [beta(a,b+t)/beta(a,b) for t in range(0,periods+1)]
                return pred
