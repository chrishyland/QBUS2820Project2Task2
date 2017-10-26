#importlib.reload(forecasting)
#model=forecasting.ses(y)
#model.fit()


# Imports
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns


def plot_components_x13(y, label=''):
    colours=['#D62728', '#FF7F0E', '#2CA02C', '#1F77B4']
    fig, ax = plt.subplots(2,2, figsize=(12,8))
    ax[0,0].plot(results.observed, color=colours[0], alpha=0.95)
    ax[0,0].set(ylabel=label, title='Observed')
    ax[0,1].plot(results.trend, color=colours[1], alpha=0.95)
    ax[0,1].set(title='Trend')
    ax[1,0].plot(results.observed/results.seasadj, color=colours[2],  alpha=0.95)
    ax[1,0].set(ylabel=label, title='Seasonal')
    ax[1,1].plot(results.irregular, color=colours[3],  alpha=0.95)
    ax[1,1].set(title='Irregular')
    plt.tight_layout()
    plt.show()
    return fig, ax



@jit
def exponentialsmoothing(y, alpha):
    n=len(y)
    l=np.zeros(n)
    l[0]=y[0]
    for i in range(1,n):
        l[i]=alpha*y[i-1]+(1-alpha)*(l[i-1])
    return l

class ses:
    def __init__(self, y):
        assert type(y)==pd.core.series.Series, 'Input must be a pandas series'
        self.time=y.index
        self.y=y.as_matrix()
        self.n=len(self.y)

    def smooth(self):
        return exponentialsmoothing(self.y, self.alpha)

    def mse(self, alpha):
        x=exponentialsmoothing(self.y, alpha)
        return np.mean(np.power(self.y-x,2))

    def loss(self, alpha):
        return (1/2)*np.log(self.mse(alpha))

    def fit(self, start=0.1):
        result=minimize(self.loss, start, bounds=((0,1),), tol=1e-6, method='L-BFGS-B')
        self.alpha=float(result.x)
        self.sigma2=self.mse(self.alpha)
        self.se=np.sqrt(np.diag((1/self.n)*result.hess_inv.todense()))
        self.fitted=self.smooth()
        self.resid=self.y-self.fitted

    def forecast(self, h):
        return np.repeat(self.smooth()[-1], h)

    def intervalforecast(self, h, level=.95):
        fitted=self.smooth()
        sigma2=self.sigma2
        forecast=np.repeat(fitted[-1], h)
        crit=stats.norm.ppf(1-(1-level)/2)
        ci=np.zeros((h,2))
        lvar=0
        for i in range(h):
            lvar+=(self.alpha**2)*(sigma2)
            var=lvar+sigma2
            ci[i,0], ci[i,1]= forecast[i]-crit*np.sqrt(var), forecast[i]+crit*np.sqrt(var)
        return ci

    def summary(self):
        N=self.n
        mse=self.sigma2
        loglik=-(N/2)*(1+np.log(2*np.pi))-(N/2)*np.log(mse)
        aic=-2*loglik+2*3
        bic=-2*loglik+np.log(N)*3
        print(' Simple exponential smoothing\n')
        print(' Smoothing parameter:')
        print(' alpha  {0:.3f} ({1:.3f}) \n'.format(self.alpha, self.se[0]))
        print(' In-sample fit:')
        print(' MSE               {0:.3f}'.format(mse))
        if loglik>0:
            print(' Log-likelihood    {0:.3f}'.format(loglik))
        else:
            print(' Log-likelihood   {0:.3f}'.format(loglik))
        print(' AIC               {0:.3f}'.format(aic))
        print(' BIC               {0:.3f}'.format(bic))
        self.aic=aic
        self.bic=bic


@jit
def holtsmooth(y, alpha, beta, phi):
    n=len(y)
    l=np.zeros(n)
    b=np.zeros(n)
    l[0]=y[0]
    b[0]=0
    for i in range(1,n):
        l[i]=alpha*y[i-1]+(1-alpha)*(l[i-1]+phi*b[i-1])
        b[i]=beta*(l[i]-l[i-1])+(1-beta)*(phi*b[i-1])
    return l, b

class holt:
    def __init__(self, y, damped=False):
        assert type(y)==pd.core.series.Series, 'Input must be a pandas series'
        self.time=y.index
        self.y=y.as_matrix()
        self.n=len(self.y)
        self.damped=damped

    def smooth(self):
        if self.damped:
            l, b =holtsmooth(self.y, self.alpha, self.beta, self.phi)
        else:
            l, b =holtsmooth(self.y, self.alpha, self.beta, 1.0)
        return l+b

    def mse(self, theta):
        if self.damped:
            l, b =holtsmooth(self.y, theta[0], theta[1], theta[2])
        else:
            l, b= holtsmooth(self.y, theta[0], theta[1], 1.0)
        return np.mean(np.power(self.y-l-b,2))

    def loss(self, theta):
        mse=self.mse(theta)
        return (1/2)*np.log(mse)

    def fit(self, start=np.array([0.3,0.05])):
        if self.damped==False:
            result=minimize(self.loss, start,  bounds=((0,1),(0,1)), tol=1e-6, method='L-BFGS-B')
        else:
            if len(start)==2:
                start=np.array([0.1,0.1, 0.98])
            result=minimize(self.loss, start,  bounds=((0,1),(0,1), (0,1)), tol=1e-6, method='L-BFGS-B')
            self.phi=float(result.x[2])
        self.alpha=float(result.x[0])
        self.beta=float(result.x[1])
        self.params=result.x
        self.sigma2=self.mse(self.params)
        self.se=np.sqrt(np.diag((1/self.n)*result.hess_inv.todense()))
        self.fitted=self.smooth()
        self.resid=self.y-self.fitted

    def forecast(self, h):
        if self.damped:
            l, b= holtsmooth(self.y, self.alpha, self.beta, self.phi)
            prediction=np.zeros(h)
            yhat=l[-1]
            for i in range(h):
                yhat+=np.power(self.phi, i+1)*b[-1]
                prediction[i]=yhat
            return prediction
        else:
            l, b= holtsmooth(self.y, self.alpha, self.beta, 1.0)
            return l[-1]+(1+np.arange(h))*b[-1]

    def forecastvariance(self, h):
        sigma2=self.sigma2
        result=np.zeros(h)
        var=sigma2
        aux=1
        for i in range(h):
            result[i]=var
            if self.damped:
                aux+=(self.phi**(1+i))*(self.beta)
            else:
                aux+=self.beta
            var+=np.power(self.alpha*aux,2)*sigma2
        return result

    def intervalforecast(self, h, level=.95):
        sigma2=self.sigma2
        crit=stats.norm.ppf(1-(1-level)/2)
        forecast=np.reshape(self.forecast(h), (h,1))
        var=np.reshape(self.forecastvariance(h), (h,1))
        return np.hstack((forecast-crit*np.sqrt(var), forecast+crit*np.sqrt(var)))

    def summary(self):
        N=self.n
        mse=self.sigma2
        loglik=-(N/2)*(1+np.log(2*np.pi))-(N/2)*np.log(mse)
        if self.damped:
            aic=-2*loglik+2*4
            bic=-2*loglik+np.log(N)*4
        else:
            aic=-2*loglik+2*3
            bic=-2*loglik+np.log(N)*3
        if self.damped:
            print(' Holt exponential smoothing (damped trend)\n')
        else:
            print(' Holt (trend corrected) exponential smoothing\n')
        print(' Smoothing parameters: ')
        print(' alpha (level) {0:.3f} ({1:.3f})'.format(self.alpha, self.se[0]))
        print(' beta (trend)  {0:.3f} ({1:.3f})'.format(self.beta, self.se[1]))
        if self.damped:
            print(' phi (damping) {0:.3f} ({1:.3f})'.format(self.phi, self.se[2]))
        print(' \n In-sample fit:')
        print(' MSE               {0:.3f}'.format(mse))
        if loglik>0:
            print(' Log-likelihood    {0:.3f}'.format(loglik))
        else:
            print(' Log-likelihood   {0:.3f}'.format(loglik))
        print(' AIC               {0:.3f}'.format(aic))
        print(' BIC               {0:.3f}'.format(bic))
        self.aic=aic
        self.bic=bic


@jit
def ahw(y, alpha, beta, delta, phi, m):
    n=len(y)
    l=np.zeros(n)
    b=np.zeros(n)
    S=np.zeros(n)
    l[:m]=np.mean(y[:m])
    b[:m]=0
    S[:m]=y[:m]-l[0]
    for i in range(m,n):
        l[i]=alpha*(y[i-1]-S[i-1])+(1-alpha)*(l[i-1]+phi*b[i-1])
        b[i]=beta*(l[i]-l[i-1])+(1-beta)*(phi*b[i-1])
        S[i]=delta*(y[i-m]-l[i-m+1])+(1-delta)*S[i-m]
    return l, b, S

@jit
def mhw(y, alpha, beta, delta, phi, m):
    n=len(y)
    l=np.zeros(n)
    b=np.zeros(n)
    S=np.zeros(n)
    l[:m]=np.mean(y[:m])
    b[:m]=0
    S[:m]=y[:m]/l[0]
    for i in range(m,n):
        l[i]=alpha*(y[i-1]/S[i-1])+(1-alpha)*(l[i-1]+phi*b[i-1])
        b[i]=beta*(l[i]-l[i-1])+(1-beta)*(phi*b[i-1])
        S[i]=delta*(y[i-m]/l[i-m+1])+(1-delta)*S[i-m]
    return l, b, S


class holtwinters:
    def __init__(self, y, additive=True, damped=False, m=12):
        assert type(y)==pd.core.series.Series, 'Input must be a pandas series'
        self.time=y.index
        self.y=y.as_matrix()
        self.n=len(self.y)
        self.additive=additive
        self.damped=damped
        self.m=12

    def smooth(self):
        if self.damped:
            if self.additive:
                l, b, S =ahw(self.y, self.alpha, self.beta, self.delta, self.phi, self.m)
            else:
                l, b, S =mhw(self.y, self.alpha, self.beta, self.delta, self.phi, self.m)
        else:
            if self.additive:
                l, b, S =ahw(self.y, self.alpha, self.beta, self.delta, 1.0, self.m)
            else:
                l, b, S =mhw(self.y, self.alpha, self.beta, self.delta, 1.0, self.m)
        if self.additive:
            return l+b+S
        else:
            return (l+b)*S

    def mse(self, theta):
        if self.damped:
            if self.additive:
                l, b, S =ahw(self.y, theta[0], theta[1], theta[2], theta[3], self.m)
            else:
                l, b, S =mhw(self.y, theta[0], theta[1], theta[2], theta[3], self.m)
        else:
            if self.additive:
                l, b, S =ahw(self.y, theta[0], theta[1], theta[2], 1.0, self.m)
            else:
                l, b, S =mhw(self.y, theta[0], theta[1], theta[2], 1.0, self.m)
        if self.additive:
            return np.mean(np.power(self.y-l-b-S,2))
        else:
            return np.mean(np.power(self.y-(l+b)*S,2))

    def loss(self, theta):
        mse=self.mse(theta)
        return (1/2)*np.log(mse)

    def fit(self, start=np.array([0.1,0.1, 0.05])):
        if self.damped==False:
            result=minimize(self.loss, start,  bounds=((0,1),(0,1),(0,1)), tol=1e-6, method='L-BFGS-B')
        else:
            if len(start)==3:
                start=np.array([0.1, 0.1, 0.05, 0.98])
            result=minimize(self.loss, start,  bounds=((0,1),(0,1),(0,1),(0,1)), tol=1e-6, method='L-BFGS-B')
            self.phi=float(result.x[3])
        self.alpha=float(result.x[0])
        self.beta=float(result.x[1])
        self.delta=float(result.x[2])
        self.params=result.x
        self.sigma2=self.mse(self.params)
        self.se=np.sqrt(np.diag((1/self.n)*result.hess_inv.todense()))
        self.fitted=self.smooth()
        self.resid=self.y-self.fitted

    def forecast(self, h):
        if self.damped:
            if self.additive:
                l, b, S =ahw(self.y, self.alpha, self.beta, self.delta, self.phi, self.m)

            else:
                l, b, S =mhw(self.y, self.alpha, self.beta, self.delta, self.phi, self.m)
            phi=self.phi
        else:
            if self.additive:
                l, b, S =ahw(self.y, self.alpha, self.beta, self.delta, 1.0, self.m)
            else:
                l, b, S =mhw(self.y, self.alpha, self.beta, self.delta, 1.0, self.m)
            phi=1.0
        yhat=l[-1]
        b=b[-1]
        S=S[-self.m:]
        prediction=np.zeros(h)
        for i in range(h):
            yhat+=np.power(phi, i+1)*b
            if self.additive:
                prediction[i]=yhat+S[i%self.m]
            else:
                prediction[i]=yhat*S[i%self.m]
        return prediction

    def forecastvariance(self, h):
        if self.additive==False:
            assert h<=self.m, 'Forecast variance not available for h>m in the multiplicative model'
        sigma2=self.sigma2
        result=np.zeros(h)
        var=sigma2
        aux=1
        for i in range(h):
            result[i]=var
            if self.damped:
                aux+=self.alpha*(self.phi**(1+i))*(self.beta)
            else:
                aux+=self.alpha*self.beta
            if (i>0) and (i%self.m==0):
                var+=np.power(aux+self.delta*(1-self.alpha),2)*sigma2
            else:
                var+=np.power(aux,2)*sigma2
        return result

    def intervalforecast(self, h, level=.95):
        sigma2=self.sigma2
        crit=stats.norm.ppf(1-(1-level)/2)
        forecast=np.reshape(self.forecast(h), (h,1))
        var=np.reshape(self.forecastvariance(h), (h,1))
        return np.hstack((forecast-crit*np.sqrt(var), forecast+crit*np.sqrt(var)))

    def summary(self):
        N=self.n
        mse=self.sigma2
        loglik=-(N/2)*(1+np.log(2*np.pi))-(N/2)*np.log(mse)
        if self.damped:
            aic=-2*loglik+2*5
            bic=-2*loglik+np.log(N)*5
        else:
            aic=-2*loglik+2*4
            bic=-2*loglik+np.log(N)*4
        if self.damped:
            if self.additive:
                print(' Additive Holt-winters exponential smoothing (damped trend)\n')
            else:
                print(' Multiplicative Holt-winters exponential smoothing (damped trend)\n')
        else:
            if self.additive:
                print(' Additive Holt-winters exponential smoothing\n')
            else:
                print(' Multiplicative Holt-winters exponential smoothing\n')
        print(' Smoothing parameters:')
        print(' alpha (level)    {0:.3f} ({1:.3f})'.format(self.alpha, self.se[0]))
        print(' beta  (trend)    {0:.3f} ({1:.3f})'.format(self.beta, self.se[1]))
        print(' delta (seasonal) {0:.3f} ({1:.3f})'.format(self.delta, self.se[2]))
        if self.damped:
            print(' phi (damping)    {0:.3f} ({1:.3f})'.format(self.phi, self.se[3]))
        print(' \n In-sample fit:')
        print(' MSE               {0:.3f}'.format(mse))
        if loglik>0:
            print(' Log-likelihood    {0:.3f}'.format(loglik))
        else:
            print(' Log-likelihood   {0:.3f}'.format(loglik))
        print(' AIC               {0:.3f}'.format(aic))
        print(' BIC               {0:.3f}'.format(bic))
        self.aic=aic
        self.bic=bic


def fanchart(y, forecast, intv1, intv2, intv3):
    assert type(y)==pd.core.series.Series, 'The time series must be a pandas series'
    assert type(forecast)==pd.core.series.Series, 'The forecast must be a pandas series'

    last=y.iloc[-1:]
    extended=last.append(forecast)

    with sns.axes_style('ticks'):
        fig, ax= plt.subplots(figsize=(8,5))
        y.plot(color='#D62728')
        extended.plot(color='black', alpha=0.4, label='Point forecast')
        ax.fill_between(extended.index, last.append(intv3.iloc[:,0]), last.append(intv3.iloc[:,1]), facecolor='#FAB8A4', lw=0)
        ax.fill_between(extended.index, last.append(intv2.iloc[:,0]), last.append(intv2.iloc[:,1]), facecolor='#F58671', lw=0)
        ax.fill_between(extended.index, last.append(intv1.iloc[:,0]), last.append(intv1.iloc[:,1]), facecolor='#F15749', lw=0)
        hold = ax.get_ylim()
        ax.fill_betweenx(ax.get_ylim(), extended.index[0], extended.index[-1], facecolor='grey', alpha=0.15)
        ax.set_ylim(hold)
    return fig, ax


def sarimaforecast(y, model, h=1, m=12):

    n=len(y)
    x=np.zeros((n+h))
    x[:n]=y

    forecast_diff=model.forecast(steps=h)[0]

    for i in range(h):
        x[n+i]=x[n+i-1]+x[n+i-m]-x[n+i-m-1]+forecast_diff[i]

    return x[-h:]
