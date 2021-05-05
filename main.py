import numpy as np
import matplotlib.pyplot as plt
import math

class PolynomialFunct:
  def __init__(self, P, alpha, features, y=0.9):
    self.P = P
    self.alpha = alpha
    self.features = features
    self.w = []
    self.prev_momentum = np.full(P,0)
    self.prev_adagrad = np.full(P,1)
    self.avg_adadelta = np.full(P,0)
    self.O_adadelta = np.full(P,0)
    self.Adam_m = np.full(P,0)
    self.Adam_v = np.full(P,0)
    self.y = y
    for i in range(P):
      self.w.append(np.random.rand(features))

  def __change_params_trad(self, grads):
    for i in range(self.P):
      self.w[i] = self.w[i] - self.alpha*(grads[i])

  def __change_params_momentum(self, grads):
    for i in range(self.P):
      v_new = (self.y * self.prev_momentum[i] + self.alpha * (grads[i]))
      self.w[i] = self.w[i] - v_new
      self.prev_momentum[i] = v_new
    return

  def __change_params_AdaGrad(self, grads): #ASK
    for i in range(self.P):
      v_new = (self.alpha * (grads[i]))/math.sqrt(self.prev_adagrad[i]+1e-8)
      self.w[i] = self.w[i] - v_new
      print(v_new)
    self.prev_adagrad = self.prev_adagrad+grads**2
    return

  def __change_params_Adadelta (self, grads): 
    self.avg_adadelta = self.y*self.avg_adadelta + (1-self.y)*(grads**2)
    for i in range(self.P):
      rms_o = math.sqrt(self.O_adadelta[i] + 1e-3)
      v_new = -(rms_o*(grads[i]))/math.sqrt(self.avg_adadelta[i] + 1e-3)
      self.w[i] = self.w[i] + v_new
      self.O_adadelta[i] = self.y*self.O_adadelta[i] + (1-self.y)*(v_new**2)
    return

  def __change_params_Adam (self, grads, b1 = 0.9, b2 = 0.999): 
    self.Adam_m = b1*self.Adam_m + (1-b1)*grads
    self.Adam_v = b2*self.Adam_v + (1-b2)*(grads**2)
    mt = self.Adam_m/(1-b1)
    vt = self.Adam_v/(1-b2)
    for i in range(self.P):
      v_new = - self.alpha*mt[i]/(math.sqrt(vt[i]) + 1e-8)
      self.w[i] = self.w[i] + v_new
    return

  def change_params(self, grads, opt):
    if(opt == None):
      self.__change_params_trad(grads)
    elif(opt == "momentum"):
      self.__change_params_momentum(grads)
    elif(opt == "adagrad"):
      self.__change_params_AdaGrad(grads)
    elif(opt == "adadelta"):
      self.__change_params_Adadelta(grads)
    elif(opt == "adam"):
      self.__change_params_Adam(grads)
    return

  def calc_one(self, x): # cambiar a vectorial (?)
    sum = 0
    for i in range(self.P):
      sum += self.w[i][0] * (x ** i)
    return sum
  
  def calc_all(self, x):
    return [self.calc_one(i) for i in x]

class Error_funct:
  def __init__(self, lambd, errtype):
    self.lambd = lambd
    self.errtype = errtype
############################# Errores ##############################

  def __mse(self,y,y_pd,w): # minimum square error
    return sum([(e[0]-e[1])**2 for e in zip(y,y_pd)])/(2*len(y))

  def __mae(self,y,y_pd,w): # minimum absolute error
    return sum([abs(e[0]-e[1]) for e in zip(y,y_pd)])/(2*len(y))

  def __regularizador(self, reg, w):
    if(reg == 1):
      return sum(self.lambd*i[0] for i in w)
    else:
      return sum(self.lambd*i[0]**2 for i in w)

  def __calc_err(self,y,y_pd,w):
    if(self.errtype == "mse"):
      return self.__mse(y, y_pd, w)
    elif(self.errtype == "mae"):
      return self.__mae(y, y_pd, w)

  def calc_loss(self,y,y_pd,w,reg=0):
    if(reg):
      return self.__calc_err(y, y_pd, w) + self.__regularizador(reg, w)
    else:
      return self.__calc_err(y, y_pd, w)

############################# Gradientes ##############################
  def __grad_mse(self,y,x,y_pd,function):
    gradient_list = np.zeros(function.P)
    for i in range(function.P):
      sum = 0
      for j in range(len(x)):
        sum += (y[j] - y_pd[j]) * -x[j]**i
      gradient_list[i] = sum/len(x)
    return gradient_list 

  def __grad_mae(self,y,x,y_pd,function): 
    gradient_list = np.zeros(function.P)
    for i in range(function.P):
      sum = 0
      for j in range(len(x)):
        sum += (y[j] - y_pd[j])/abs(y[j] - y_pd[j]) * -x[j]**i
      gradient_list[i] = sum/len(x)
    return gradient_list
  
  def __grad_regularizador(self, reg, w):
    if(reg==1):
      return [self.lambd for i in range(len(w))]
    elif(reg==2):
      return [2*self.lambd*w[i][0] for i in range(len(w))]

  def __grad_err(self, y, x, y_pd, function):
    if(self.errtype == "mse"):
      return self.__grad_mse(y,x,y_pd,function)
    elif(self.errtype == "mae"):
      return self.__grad_mae(y,x,y_pd,function)

  def calc_grad(self,y,x,y_pd,function, reg=0):
    grads = self.__grad_err(y,x,y_pd,function)
    if(reg):
      return grads + self.__grad_regularizador(reg, function.w)
    return grads

def h(x,w): #creo que esto se reemplazaria en calc_one
  j = np.arange(len(w))
  j = np.power(x,j)
  return w @ j


def model_exec(error, function, x_ds, y_ds, iter, reg, opt=None):
  y_pd = function.calc_all(x_ds)
  loss = error.calc_loss(y_ds,y_pd,function.w, reg)
  i=0
  while(i < iter):
    grads = error.calc_grad(y_ds,x_ds,y_pd,function, reg)
    function.change_params(grads, opt)
    y_pd = function.calc_all(x_ds)
    loss = error.calc_loss(y_ds,y_pd,function.w, reg)
    i+=1
  print("Loss with", error.errtype,":", loss, "in", i, "iterations in main1")
  print("W es ", function.w)
  print("----------------------------")
  return y_pd

if __name__ == "__main__":
  x_ds = np.arange(0,1,0.05)
  real_sine = np.array([ np.sin(2*i*np.pi) for i in x_ds])
  y_ds = np.array([ np.sin(2*i*np.pi) + np.random.normal(0, 0.2) for i in x_ds])
  sine_function = PolynomialFunct(7, 0.7, 1, 0.9)
  mse_error = Error_funct(0.0001, "mse")
  mae_error = Error_funct(0.0001, "mae")

  # -----------------------TESTS-------------------------
  #mse_pd = model_exec(mse_error, sine_function, x_ds, y_ds, 10000, 0, "momentum")  #MSE
  #sine_function = PolynomialFunct(4, 0.7, 1, 0.9)
  #mae_pd = model_exec(mae_error, sine_function, x_ds, y_ds, 10000, 0, "momentum")  #MAE
  #sine_function = PolynomialFunct(4, 0.7, 1, 0.9)
  mae_pd_adagrad = model_exec(mae_error, sine_function, x_ds, y_ds, 10000, 0, "adam")  #MAE
  # -----------------------GRAFICOS-------------------------
  plt.plot(x_ds, y_ds, '*')
  #plt.plot(x_ds, real_sine, '*', color='green')
  # plt.plot(x_ds, mse_pd, color='blue')
  # plt.plot(x_ds, mae_pd, color='green')
  plt.plot(x_ds, mae_pd_adagrad, color='red')

  plt.show()

 # https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0