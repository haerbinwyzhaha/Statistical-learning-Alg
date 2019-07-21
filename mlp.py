#encoding=utf-8
import numpy as np
##完成感知器算法
class MLP(object) :
    
    def __init__(self,dim):
        #[dim,1]
        self.w = np.zeros([dim,1]) 
        #[1,1]
        self.b = np.zeros(1)
        # learning_rate
        self.lr = 1.0

    def _distance(self,x):
        # x [batch_size,dims]
        dis = np.squeeze(np.dot(x,self.w) + self.b)
        return dis
    
    def predict(self, x):
        # x [batch_size,dims]
        res = self._distance(x)
        return res , np.sign(res)

    def _loss(self,x,y):
        dis , y_pred = self.predict(x)
        mis_pred = np.equal(y,y_pred)[0]
        mis_pred_float = np.array(mis_pred,dtype=np.float32)

        mis_case = [dis[i] for i in range(len(y)) if i not in mis_pred_float]
        loss = -sum(mis_case)
        return loss

    def sgd(self,x,y) :
        step = 0
        ## 当存在误分类点
        while(sum(self.predict(x)[-1]==y) != len(y) ):
            for i in range(len(y)):
                while self.predict(x[i])[-1] != y[i] :
                    step+=1 
                    self.w += self.lr * np.expand_dims(x[i],-1) * y[i]
                    self.b += self.lr * y[i]
                    print "update step %s , w = %s , b = %s"%(step,np.squeeze(self.w),np.squeeze(self.b))

    def toString(self) :
        print "w : " , np.squeeze(self.w)
        print "b : " , np.squeeze(self.b)
        


if __name__ == '__main__' :
    # assume input dim=2  
    np.random.seed(2)
    # input dim
    DIM = 2
    # input num
    N = 5
    mlp = MLP(DIM)

    X = [ [np.random.randint(-10,10) for i in range(DIM)] for k in range(N) ]
    X = np.array(X)
    Y = np.array([np.random.choice([-1,1]) for i in range (N)])
    #X = np.array([[3,3],[4,3],[1,1]])
    #Y = np.array([1,1,-1])


    print "Prediction Before SGD : " , mlp.predict(X)[-1]
    print "Label : " , Y 
    mlp.sgd(X,Y)
    print "Prediction AFter SGD : " , mlp.predict(X)[-1]
    print "Label : " ,Y 
    mlp.toString()
