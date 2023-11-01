# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:58:51 2023

@author: henri
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr 

""" Regressão Linear simples com dados sintéticos """
x = np.arange(-1,5,0.2)
N = len(x)
# y = 1 -4x + x^2 = [1 x x^2]*[1 -4 +1]
beta = np.asmatrix([[1],[-4],[1]])
X0 = np.ones((N,1))
X1 = np.transpose(np.mat(x))
X2 = np.transpose(np.mat(x**2))
X = np.hstack((X0,X1,X2))
Y = np.matmul(X,beta)

# Adicionando ruído à saída e exibindo
Yn = Y + np.random.normal(0,1.0,size=(N,1))
plt.figure(1)
plt.plot(X[:,1],Y)
plt.plot(X[:,1],Yn,'r*')
plt.grid(True)

# Calculando parâmetros com mínimos quadrados
Xt = np.transpose(X)
pinv = np.linalg.inv(np.matmul(Xt,X)) #quase a pseudo-inversa
aux = np.matmul(Xt,Yn)
beta_est = np.matmul(pinv,aux)

print("\nParâmetros originais:")
print(beta)
print("Parâmetros estimados:")
print(beta_est)

# Comparando a curva real, os dados e a reta estimada
Y_est = np.matmul(X,beta_est)
plt.figure(2)
plt.plot(X[:,1],Y)
plt.plot(X[:,1],Yn,'r*')
plt.plot(X[:,1],Y_est)
plt.grid(True)
plt.legend(("Original","Dados","Estimativa"))

# Calculando os intervalos de confiança
res = Yn-Y_est #resíduos
stdev = np.matmul(np.transpose(res),res)/(N-len(beta_est)) #variância
stdev = np.float64(stdev) # forçando para virar um número escalar
covmat = stdev*pinv #matriz de covariância dos betas

errp = np.transpose(np.mat(np.diag(covmat))) 
errp = np.sqrt(errp) #erros-padrão são a raiz quadrada da diagonal da matriz de
                     #covariância dos betas
                                           
conf_int = np.hstack((beta_est-1.96*errp,beta_est+1.96*errp))
print("\nIntervalos de confiança 95%: ")
print(conf_int)

# Testando a correlação dos resíduos com o estimado
aux1 = np.squeeze(np.asarray(Y_est))
aux2 = np.squeeze(np.asarray(res))
stats = pearsonr(aux1,aux2)
print("\nCorrelação Resíduos-Estimativa")
print(stats)

aux1 = np.squeeze(np.asarray(X[:,1]))
aux2 = np.squeeze(np.asarray(res))
stats = pearsonr(aux1,aux2)
print("\nCorrelação Resíduos-Var. Indepdente")
print(stats)

# Análise gráfica dos resíduos
plt.figure(5)
plt.plot(X[:,1],res,'+b')
plt.ylabel("Resíduos")






