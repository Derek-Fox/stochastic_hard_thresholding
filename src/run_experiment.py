# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK
"""
Created on Fri Mar  1 21:14:55 2019

@author: gul15103, defox
"""

import svrg_ht
import sgd_ht
import scsg_ht

import argparse

parser = argparse.ArgumentParser(description='hardthresholding methods')
parser.add_argument('--epoch', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--file_path', type=str)
parser.add_argument('--output_folder', default='../logs/', type=str)
parser.add_argument('--batchsize', default=1, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--htk', default=200, type=int)
parser.add_argument('--B_type', default=1, type=int)
parser.add_argument('--opt', type=str)
parser.add_argument('--data_type', type=str)
parser.add_argument('--regularizer', default=1e-5, type=float)
parser.add_argument('--etas', nargs='+', type=float)
parser.add_argument('--B_types', nargs='+', type=int)

args = parser.parse_args()

f = args.file_path
epoch = args.epoch
batch_size = args.batchsize

if args.data_type =='multi':
    loss = 'multi_class_softmax_regression'
    multi_class = True
elif  args.data_type =='binary':
    loss = 'logistic'
    multi_class = False
elif  args.data_type =='ridge':
    loss = 'ridge'
    multi_class = False
    
batch_size_B_type='fixed'
ht_k=args.htk
log_interval=1000
regularizer = args.regularizer
output_folder = args.output_folder


etas = args.etas

for eta  in etas:
  if args.opt == 'svrg':
      svrg_ht.svrg_ht( f,regularizer,  epoch, batch_size,  stepsize = eta, stepsize_type = "fixed", verbose = True, optgap = 10**(-30), loss = loss ,ht_k=ht_k, log_interval=log_interval,output_folder=output_folder, multi_class= multi_class)
  elif args.opt == 'sgd':
    sgd_ht.sgd_ht( f,regularizer,  epoch, batch_size,  stepsize = eta, stepsize_type = "fixed", verbose = True, optgap = 10**(-30), loss = loss , ht_k=ht_k ,log_interval=log_interval, output_folder=output_folder, multi_class= multi_class)
  elif args.opt == 'scsg':
     for batch_size_B in args.B_types:
             scsg_ht.scsg_ht( f, regularizer, epoch, batch_size_B,batch_size_B_type,batch_size,  stepsize = eta, stepsize_type = "fixed", verbose = True, optgap = 10**(-30), loss = loss , ht_k=ht_k, log_interval=log_interval, output_folder=output_folder , multi_class= multi_class)
