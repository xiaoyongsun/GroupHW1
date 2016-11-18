__author__ = 'marcopereira'
import numpy as np
from pandas import DataFrame
from Scheduler.Scheduler import Scheduler
import pandas as pd
from scipy.optimize import minimize

class CouponBond(object):
    def __init__(self, fee, start,coupon,  maturity, freq, referencedate, observationdate,notional):
        self.fee = fee
        self.coupon=coupon
        self.start = start
        self.maturity = maturity
        self.freq= freq
        self.referencedate = referencedate
        self.observationdate = observationdate
        self.myScheduler = Scheduler()
        self.delay = self.myScheduler.extractDelay(freq=freq)
        self.getScheduleComplete()
        self.ntimes=len(self.datelist)
        self.referencedate = referencedate
        self.pvAvg=0.0
        self.cashFlows = DataFrame()
        self.cashFlowsAvg = []
        self.yieldIn = 0.0
        self.notional=notional

    def getScheduleComplete(self):
        self.datelist = self.myScheduler.getSchedule(start=self.start,end=self.maturity,freq=self.freq,referencedate=self.referencedate)
        fullset = list(sorted(list(set(self.datelist)
                                   .union([self.referencedate])
                                   .union([self.start])
                                   .union([self.maturity])
                                   .union([self.observationdate])
                                   )))
        return fullset,self.datelist

    def setLibor(self,libor):
        self.libor = libor/libor.loc[self.referencedate]
        self.ntimes = np.shape(self.datelist)[0]
        self.ntrajectories = np.shape(self.libor)[1]
        self.ones = np.ones(shape=[self.ntrajectories])

    def getExposure(self, referencedate):
        if self.referencedate!=referencedate:
            self.referencedate=referencedate
            self.getScheduleComplete()
        deltaT= np.zeros(self.ntrajectories)
        if self.ntimes==0:
            pdzeros= pd.DataFrame(data=np.zeros([1,self.ntrajectories]), index=[referencedate])
            self.pv=pdzeros
            self.pvAvg=0.0
            self.cashFlows=pdzeros
            self.cashFlowsAvg=0.0
            return self.pv
        for i in range(1,self.ntimes):
            deltaTrow = ((self.datelist[i]-self.datelist[i-1]).days/365)*self.ones
            deltaT = np.vstack ((deltaT,deltaTrow) )
        self.cashFlows= self.coupon*deltaT
        principal = self.ones
        if self.ntimes>1:
            self.cashFlows[-1:]+= principal
        else:
            self.cashFlows = self.cashFlows + principal
        if(self.datelist[0]<= self.start):
            self.cashFlows[0]=-self.fee*self.ones
    
        if self.ntimes>1:
            self.cashFlowsAvg = self.cashFlows.mean(axis=1)*self.notional
        else:
            self.cashFlowsAvg = self.cashFlows.mean() * self.notional
        pv = self.cashFlows*self.libor.loc[self.datelist]
        self.pv = pv.sum(axis=0)*self.notional
        self.pvAvg = np.average(self.pv)*self.notional
        return self.pv

    def getPV(self,referencedate):
        self.getExposure(referencedate=referencedate)
        return self.pv/self.libor.loc[self.observationdate]

    def getLiborAvg(self, yieldIn, datelist):
        self.yieldIn = yieldIn
        time0 = datelist[0]
        # this function is used to calculate exponential single parameter (r or lambda) Survival or Libor Functions
        Z = np.exp(-self.yieldIn * pd.DataFrame(np.tile([(x - time0).days / 365.0 for x in self.datelist], reps=[self.ntrajectories,1]).T,index=self.datelist))
        return Z

    def getYield(self,price):
        # Fit model to curve data
        yield0 = 0.05 * self.ones
        self.price = price
        self.yieldIn = self.fitModel2Curve(x=yield0)
        return self.yieldIn


    def fitModel2Curve(self, x ):
        # Minimization procedure to fit curve to model
        results = minimize(fun=self.fCurve, x0=x)
        return results.x

    def fCurve(self, x):
        # raw data error function
        calcCurve = self.getLiborAvg(x, self.datelist)
        thisPV = np.multiply(self.cashFlows,calcCurve).mean(axis=1).sum(axis=0)
        error = 1e4 * (self.price - thisPV) ** 2
        return error
