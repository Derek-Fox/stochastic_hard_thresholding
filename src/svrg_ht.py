#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:24:52 2018

@author: Neil, defox
"""

import numpy as np

import os
import time
import Loss
import Util
import sys

from tensorboardX import SummaryWriter


def svrg_ht(
    f,
    regularizer,
    epochs,
    batch_size,
    stepsize=10**-5,
    stepsize_type="fixed",
    verbose=False,
    optgap=10 ** (-3),
    loss="logistic",
    ht_k=100,
    log_interval=1000,
    output_folder="../logs/",
    multi_class=False,
):
    """
    f: location of data file
    regularizer: default val 1e-5
    epochs: default val 10
    batch_size: default val 1
    stepsize: default val 1e-5
    stepsize_type: fixed, decay, sqrtdecay, squaredecay - default val "fixed"
    verbose: default val False
    optgap: default val 10**(-3)
    loss: regression type - logistic, svm, ridge, multi_class_softmax_regression - default val logistic
    ht_k: sparsity - default val 100
    log_interval: default val 1000
    output_folder: default val '../logs/'
    multi_class: default val False
    """

    if "simulate" in f:
        data = np.load(f)
        x = data["x"]
        y = data["y"]
    else:
        x, y = Util.readlibsvm(f)
    if "news20" in f:
        y = y - 1

    index = np.arange(np.shape(x)[0])
    np.random.shuffle(index)
    x = x[index, :]
    y = y[index]

    output_folder_suffix = f"{os.path.basename(f)}_loss_{loss}_opt_svrg_stepsize_{stepsize}_batchsize_{batch_size}_ht_k_{ht_k}_log_interval_{log_interval}_epoch_{epochs}"
    output_folder_name = output_folder + output_folder_suffix
    logger = SummaryWriter(output_folder_name)
    file = open(output_folder_name + ".txt", "w")

    file.write(f"The dataset: {f}\n")
    file.write(f"The number of instances N: {x.shape[0]}\n")
    file.write(f"The number of features p: {x.shape[1]}\n")
    file.write(f"The step size: {stepsize}\n")
    file.write(f"The epoch: {epochs}\n")
    file.write(f"The loss type: {loss}")
    file.write(f"The regularizer: {regularizer}\n")
    file.write(f"The ht_k: {ht_k}\n")
    file.write(f"The batch_size: {batch_size}\n")

    file.write("num_epoch,num_IFO,num_HT,loss\n")

    if loss == "logistic":
        loss = Loss.LogisticLoss_version2()
    elif loss == "svm":
        loss = Loss.svm_quadratic()
    elif loss == "ridge":
        loss = Loss.ridge_regression()
    elif loss == "multi_class_softmax_regression":
        loss = Loss.multi_class_softmax_regression()

    print(f"The dataset : {f}")
    print(f"The number of instances N : {x.shape[0]}")
    print(f"The number of features p : {x.shape[1]}")
    print(f"The batch size for SGD : {batch_size}")
    print(f"The step size : {stepsize}")
    print(f"The epoch : {epochs}")
    print(f"The loss type: {loss}")
    print(f"The regularizer : {regularizer}")

    if multi_class:
        w = np.zeros((x.shape[1], len(np.unique(y))))
    else:
        w = np.zeros(x.shape[1])

    obj_list = []
    t0 = time.time()
    iter = 0

    for epoch in range(epochs):
        # print("master iteration ", k)

        eta = updateStepsize(stepsize, stepsize_type, epoch)

        w_multi = np.copy(w)
        mu = loss.grad(x, y, w, regularizer)

        for batch in range(int(x.shape[0] / batch_size)):

            # print("master iteration ", k, t)
            if epoch < 1:
                # lock.acquire()
                sample_id = np.random.randint(x.shape[0] - batch_size, size=1)
                sample_id = sample_id[0]
                g1 = loss.grad(
                    x[sample_id : sample_id + batch_size],
                    y[sample_id : sample_id + batch_size],
                    w,
                    regularizer,
                )
                # print(multiprocessing.current_process(), "Grad w ", w[ 0:3])
                # lock.release()
                v = g1
            else:
                # lock.acquire()
                sample_id = np.random.randint(x.shape[0] - batch_size, size=1)
                sample_id = sample_id[0]
                g1 = loss.grad(
                    x[sample_id : sample_id + batch_size],
                    y[sample_id : sample_id + batch_size],
                    w,
                    regularizer,
                )
                # print(multiprocessing.current_process(), "Grad w ", w[ 0:3])
                # lock.release()
                g2 = loss.grad(
                    x[sample_id : sample_id + batch_size],
                    y[sample_id : sample_id + batch_size],
                    w_multi,
                    regularizer,
                )
                v = g1 - g2 + mu

            # print(multiprocessing.current_process(), "Before Update w ", w[0:3], w[ -3:-1])
            w = w + eta * -v
            if multi_class:
                for j in range(w.shape[1]):
                    w[np.absolute(w[:, j]).argsort()[:-ht_k][::-1], j] = 0
            else:
                w[np.absolute(w).argsort()[:-ht_k][::-1]] = 0
                # print(multiprocessing.current_process(), "After Update w ", w[0:3], w[ -3:-1])
            iter = iter + 1

            if (batch + 1) % log_interval == 0:
                obj_temp = loss.obj(x, y, w, regularizer)
                time_k = time.time()
                print("------------------------------------------------")
                if verbose:
                    print(f"Epoch: {epoch+1}, data passes: {epoch+1}, time: {time_k - t0}, stepsize: {eta}")
                    print(f"Norm for w : {np.square(np.linalg.norm(w))}")
                print(f"Iteration {iter}: objective value = {obj_temp}")  # always print this
                obj_list.append(obj_temp)

                if epoch > 0 and np.abs(obj_list[-1] - obj_list[-2]) < optgap:
                    print(f"Optimality gap tolerance reached: {optgap}")
                    break
                logger.add_scalar("loss_number-IFO", obj_temp, (2 * epoch) * x.shape[0] + 2 * batch * batch_size,)
                logger.add_scalar("loss_number-HT", obj_temp, iter)
                file.write(f"{((2 * epoch) * x.shape[0] + 2 * batch * batch_size) / x.shape[0]},{iter},{obj_temp}\n")
    file.close()


def updateStepsize(stepsize, stepsize_type, epoch):
    if stepsize_type == "fixed":
        return stepsize

    if stepsize_type == "decay":
        eta = stepsize * 1.0 / (epoch + 1)
    elif stepsize_type == "sqrtdecay":
        eta = stepsize * 1.0 / np.sqrt(epoch + 1)
    elif stepsize_type == "squaredecay":
        eta = stepsize * (1.0 / np.square(epoch + 1))
    return eta


def main():
    if len(sys.argv) < 3:
        print("Usage: python svrg_ht.py <dataset_name> <loss_function>")
    f = "../data/" + sys.argv[1]
    loss = sys.argv[2]
    multi_class = loss == "multiclass"

    svrg_ht(
        f,
        regularizer=1e-5,
        epochs=10,
        batch_size=1,
        stepsize=1,
        stepsize_type="fixed",
        verbose=True,
        optgap=10 ** (-3),
        loss=loss,
        ht_k=1000,
        multi_class=multi_class,
    )


if __name__ == "__main__":
    main()
