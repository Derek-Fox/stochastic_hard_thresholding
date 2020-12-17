import abc
import numpy as np

class Loss(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def grad(self, x, y, w, lambda_):
        pass

    @abc.abstractmethod
    def obj(self, x, y, w, lambda_):
        pass


class LogisticLoss(Loss): # different logistic ridge regression and don't use it anymore

    # def __init__(self, name):
    #     pass
    
    def sigmoid (self, vector ):
        return np.array( list( map( lambda x: 1./(1. + np.exp(-x)), vector ) ) )

    def grad(self, x, y, w, lambda_):
        return (-1*x.T * (y - self.sigmoid(x.dot(w))).T + 2*lambda_*w)*1./x.shape[0]

    def obj(self, x, y, w, lambda_):
        return (-((y.dot( np.log( self.sigmoid(x.dot(w)) ) )) + (( 1- y ).dot( np.log(1- self.sigmoid(x.dot(w))) ))) + (lambda_)*np.square( np.linalg.norm(w)))*1./x.shape[0]


class LogisticLoss_version2(Loss): # logistic ridge regression

    # def __init__(self, name):
    #     pass
    def sigmoid (self, vector ):
        return np.array( list( map( lambda x: 1./(1. + np.exp(-x)), vector ) ) )

    def grad(self, x, y, w, lambda_):
        #return (-1*x.T * (y - self.sigmoid(x.dot(w))).T + 2*lambda_*w)*1./x.shape[0]
        return ( - x.T.dot( np.multiply( self.sigmoid( - np.multiply( y, x.dot( w )) ), y))*1./x.shape[0]+ 2*lambda_*w)
    
    def inv_H_sk( self, x, y, w, lambda_,sk) :
        dot = np.multiply( y, x.dot( w ) )
        tmp1 = np.multiply( self.sigmoid( - dot) , self.sigmoid(  dot ))
        #tmp2 = np.multiply( tmp1, y)
        part1 = x.T.dot(np.diag( tmp1))
        part2 = x.dot( sk )
        return part1.dot( part2 )*1.0/x.shape[0] + 2*lambda_*sk

    def obj(self, x, y, w, lambda_):
        #return (-((y.dot( np.log( self.sigmoid(x.dot(w)) ) )) + (( 1- y ).dot( np.log(1- self.sigmoid(x.dot(w))) ))) + (lambda_)*np.linalg.norm(w))*1./x.shape[0]
        return( np.sum( np.log(1+np.exp( -np.multiply( y, x.dot( w ))))))*1./x.shape[0]+ (lambda_)*np.square(np.linalg.norm(w))
class ridge_regression( Loss ):
    def grad( self, x, y,w, lambda_):
        return( x.T.dot( x.dot( w ) - y) )*2./x.shape[0]+ 2*lambda_*w
    def obj( self, x, y, w, lambda_ ):
        return( np.square( np.linalg.norm( x.dot( w ) - y)))*1./x.shape[0]+ (lambda_)*np.square(np.linalg.norm(w))
    def inv_H_sk( self, x, y, w, lambda_, sk ):
        return( x.T.dot( x.dot(sk ))*1./x.shape[0] + 2*lambda_*sk )
        
class multi_class_softmax_regression( Loss ):
    def grad( self, x, y, w, lambda_):
        grad_ = np.zeros_like( w )
        
        for j in range( w.shape[1]):
            x_1 = x[ y == j, :]
            tmp_1 = np.exp( x_1.dot( w ))
            num_1 =  tmp_1.sum( axis=1) - tmp_1[:, j]**2
            denom_1 = np.multiply( tmp_1[:, j], tmp_1.sum( axis=1))
            x_2 = x[ y!= j, :]
            y_2 = y[y!=j]
            tmp_2 =np.exp(  x_2.dot( w )) 
            num_2 =  tmp_2[ range( len( y_2)), j]
            denom_2 = tmp_2.sum( axis=1)
            grad_[:, j ] = (-np.transpose( x_1).dot( np.divide( num_1, denom_1) ) + np.transpose( x_2).dot( np.divide( num_2, denom_2)))/x.shape[0]
        grad_ = grad_ + lambda_*w
        return grad_
    def obj( self, x, y, w, lambda_ ):
        temp = np.exp( x.dot( w ))
        row = range( len( y ))
        col = y.astype(int)
        pick_numerator = temp[ row, col]
        cal_denom =  temp.sum( axis=1)
        obj_value = -np.sum( np.log( np.divide( pick_numerator, cal_denom)))/x.shape[0] + lambda_*np.square( np.linalg.norm( w ))/2
        return obj_value

