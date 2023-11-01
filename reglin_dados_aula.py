# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 19:58:11 2023

@author: henri
"""

import numpy as np
import pyreadstat


""" Regressão linear múltipla com dados reais """
df, meta = pyreadstat.read_sav("alunos.sav")
N = np.size(df,0) 

Y = np.transpose(np.mat(df['raclog'])) #Variável dependente

# Pegando grupo e redação como variáveis independentes
X0 = np.ones((N,1))
X1 = np.transpose(np.mat(df['redacao']))
X2 = np.transpose(np.mat(df['grupo']))
X2 = X2 - 1 # Ajuste para que CDF seja 0 e Fundão seja 1
X = np.hstack((X0,X1,X2)) #Matriz de regressores

# Calculando os betas
Xt = np.transpose(X)
pinv = np.linalg.inv(np.matmul(Xt,X)) #pseudo-inversa
aux = np.matmul(Xt,Y)
beta_est = np.matmul(pinv,aux)

print("Parâmetros estimados:")
print(beta_est)

# Calculando os intervalos de confiança
Y_est = np.matmul(X,beta_est)
res = Y-Y_est #resíduos
stdev = np.matmul(np.transpose(res),res)/(N-len(beta_est)) #variância
stdev = np.float64(stdev) # forçando para virar um número escalar
covmat = stdev*pinv #matriz de covariância dos betas

errp = np.transpose(np.mat(np.diag(covmat))) 
errp = np.sqrt(errp) #erros-padrão são a raiz quadrada da diagonal da matriz de
                     #covariância dos betas
                                           
conf_int = np.hstack((beta_est-1.96*errp,beta_est+1.96*errp))
print("Intervalos de confiança 95%: ")
print(conf_int)