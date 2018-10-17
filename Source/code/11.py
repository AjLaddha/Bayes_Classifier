from collections import defaultdict
import numpy as np
# from tqdm import tqdm
import re
import os
import math
import random
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import argparse

def drange2(start, stop, step):
    numelements = int((stop-start)/float(step))
    for i in range(numelements+1):
            yield start + i*step
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]
class GaussNB:
    def __init__(self):
        pass
    def load_file(self,data):
        """
        :param data: raw space separated file
        :param header: remove header if it exits
        : return:
        Load and convert each string of data into a float
        """
        dataset=np.loadtxt(data,usecols=(0,1))
        return dataset


    def split_data_randomn(self,data,weight):
        """
        :param data:
        :param weight: indicates the % of rows that will be used for training
        :return : Randomnly selects rows from training according to the weight and rest as test dataset
        """
        train_size=int(len(data)*weight)
        train_set=[]
        c=0
        for i in range (train_size):
            index=random.randrange(len(data))
            train_set.append(data[index])
            data=np.delete(data,index,0)
            c=c+1
        return [np.array(train_set),data]


    def split_data(self,data,weight):
        train_size=int(len(data)*weight)
        train_set=[]
        test_set=[]
        c=0

        for i in range (len(data)):
            if i < train_size:
                train_set.append(data[i])
            else :
                    test_set.append(data[i])

        return [np.array(train_set),np.array(test_set)]

    def cal_mean(self,data):
        mean=np.mean(data,dtype=np.float32,axis=0)
        # print mean
        return mean
    def mean_manual(self,data):
        meanx=0
        meany=0
        for i in range (len(data)):
            meanx=meanx+data[i][0]
            meany=meany+data[i][1]
        meanx=meanx/len(data)
        meany=meany/len(data)
        mean=[meanx,meany]
        mean=np.array(mean)
        # print mean
        return mean

    def covmatrix(self,data,mean):
        varx=0
        vary=0
        varxy=0
        for i in range (len(data)):
        	varxy=varxy+((data[i][0]-mean[0])*(data[i][1]-mean[1]))
        	varx=varx + math.pow((data[i][0]-mean[0]),2)
        	vary= vary + math.pow((data[i][1]-mean[1]),2)
        varx = varx /len(data)
        vary = vary / len(data)
        varxy=varxy/len(data)
        var = [varx,varxy,varxy,vary]
        var=np.array(var)
        # print np.reshape(var,(2,2))
        return var


    def cov_matrix_avg(self,var1,var2):

        cov_matrix=(var1+var2)/2
        v1=(cov_matrix[0]+cov_matrix[3])/2
        cov_matrix[0]=v1
        cov_matrix[3]=v1
        cov_matrix[1]=0
        cov_matrix[2]=0
        # print np.reshape(cov_matrix,(2,2))
        return cov_matrix

    def g(self,mean,cov,p):
        var=cov[0]
        w=mean/var
        w0=np.dot(np.transpose(mean),mean)
        w0=w0*-1
        w0=w0/(2*var)
        w0=w0+math.log(p)
        return [w,w0]
    def boundary_decision(self,mean1,mean2,cov,p1,p2,x):
        # print "hola"
        v= cov[0]
        w1,w01=self.g(mean1,cov,p1)
        w2,w02=self.g(mean2,cov,p2)
        w=np.transpose(w1-w2)
        w0=w01-w02
        g12=np.dot(w,x)
        g12=g12+w0


        vote=[0,0]
        if g12>=0 :
            vote[0]=vote[0]+1
        else:
            vote[1]=vote[1]+1
        max=-1
        ipos=-1
        for i in range (2):
            if max<=vote[i]:
                max=vote[i]
                ipos=i

                if ipos==0:
            return 'C1'
        if ipos==1:
            return 'C2'






def main():
        nb=GaussNB()
        parser=argparse.ArgumentParser()

        parser.add_argument("-f",help="Class 1 dataset")
        parser.add_argument("-a",help="Class 2 dataset")

        args=parser.parse_args()

        data1=open(args.f,"r")
        data2=open(args.a,"r")

        dataset1=nb.load_file(data1)
        dataset2=nb.load_file(data2)

        train_list1 , test_list1 = nb.split_data(dataset1,weight=.75)

        train_list2 , test_list2 = nb.split_data(dataset2,weight=.75)

        mean1=nb.mean_manual(train_list1)
        mean2=nb.mean_manual(train_list2)

        color=['r','b','g']

        cov_matrix1=nb.covmatrix(train_list1,mean1)
        cov_matrix2=nb.covmatrix(train_list2,mean2)

        cov_matrix=nb.cov_matrix_avg(cov_matrix1,cov_matrix2)
        p1=(len(train_list1)*1.0)/(len(train_list1)+len(train_list2))
        p2=(len(train_list2)*1.0)/(len(train_list1)+len(train_list2))

        my_dict=defaultdict(list)

        true_pred_c1=0
        true_pred_c2=0

        conf_matrix=[0,0,0,0]
        for i in range(len(test_list1)):
            class_identified = nb.boundary_decision(mean1,mean2,cov_matrix,p1,p2,test_list1[i])

            if class_identified=='C1':
                true_pred_c1=true_pred_c1+1
                conf_matrix[0]=conf_matrix[0]+1
            elif class_identified=='C2':
                conf_matrix[1]=conf_matrix[1]+1

            my_dict[class_identified].append([test_list1[i],'C1'])


        for i in range(len(test_list2)):
            class_identified=nb.boundary_decision(mean1,mean2,cov_matrix,p1,p2,test_list2[i])
            if class_identified=='C2':
                true_pred_c2=true_pred_c2+1
                conf_matrix[3]=conf_matrix[3]+1
            elif class_identified=='C1':
                conf_matrix[2]=conf_matrix[2]+1
            my_dict[class_identified].append([test_list2[i],'C2'])
                val = [0.0,0.0]
        for key in my_dict:
            li=my_dict.get(key)
            for i in range (len(li)):
                if 'C1'==li[i][1]:
                    val[0]=val[0]+1
                if 'C2'==li[i][1]:
                    val[1]=val[1]+1

        recall_c1=(true_pred_c1*1.0)/val[0]
        recall_c2=(true_pred_c2*1.0)/val[1]

        prec_c1=(true_pred_c1*1.0)/(len(my_dict['C1']))
        prec_c2=(true_pred_c2*1.0)/(len(my_dict['C2']))

        f_c1= (2.0*prec_c1*recall_c1)/(prec_c1+recall_c1)
        f_c2= (2.0*prec_c2*recall_c2)/(prec_c2+recall_c2)

        mean_prec=(prec_c1+prec_c2)/2
        mean_recall=(recall_c1+recall_c2)/2
        f_mean=(2.0*mean_prec*mean_recall)/(mean_prec+mean_recall)
        print true_pred_c1 , true_pred_c2
        accuracy=((true_pred_c1+true_pred_c2)*1.0)/(len(test_list1)+len(test_list2))
        print "prec_c1=" , prec_c1
        print "prec_c2=" , prec_c2

        print "recall_c1=" , recall_c1
        print "recall_c2=" , recall_c2

        print "f_c1=" , f_c1
        print "f_c2=" , f_c2

        print "mean_prec=" , mean_prec
        print "mean_recall=" , mean_recall
        print "f_mean=" , f_mean
        print "Accuracy=" , accuracy

        print np.reshape(conf_matrix,(2,2))

        d1_x = dataset1[:, 0]
        d2_x = dataset2[:, 0]

        all_x = np.concatenate((d1_x, d2_x), axis=0)
        d1_y = dataset1[:, 1]
        d2_y = dataset2[:, 1]

        all_y = np.concatenate((d1_y, d2_y), axis=0)

        min_x, max_x, min_y, max_y = min(all_x)-1, max(all_x)+1, min(all_y)-1, max(all_y)+1

        a = []
        b = []

        for i in drange2(int(min_y-1), int(max_y+1), 0.05):
            for j in drange2(int(min_x-1), int(max_x+1), 0.05):
                temp = []
                temp.append(j)
                temp.append(i)
                class_identified = nb.boundary_decision(mean1, mean2, cov_matrix,p1,p2,temp)
                Z = -1
                if class_identified == "C1" :
                    a.append(temp)
                elif class_identified == "C2":
                    b.append(temp)

        na = np.array(a)
        nb = np.array(b)

        plt.plot(na[:, 0], na[:, 1], marker='.', c='#d37e7e')
        plt.plot(nb[:, 0], nb[:, 1], marker='.', c='#6b98bf')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(d1_x, d1_y, '#c42d2f', marker="o", linestyle="None", label="Class-1")
        plt.plot(d2_x, d2_y, '#283766', marker="o", linestyle="None", label="Class-2")

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
          ncol=2, fancybox=True, shadow=True)
        plt.show()


        # _------------------------------------------------------------------

        # code snippet to plot contours

        d1_x = dataset1[:, 0]
        d2_x = dataset2[:, 0]
        # d3_x = dataset3[:, 0]
        all_x = np.concatenate((d1_x, d2_x), axis=0)
        d1_y = dataset1[:, 1]
        d2_y = dataset2[:, 1]
        # d3_y = dataset3[:, 1]
        all_y = np.concatenate((d1_y, d2_y), axis=0)

        x = d1_x
        y = d1_y
        nstd = 2
        ax = plt.subplot(111)
        cov = np.cov(x, y)
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        lambda_ = np.sqrt(vals)
        for j in xrange(1, 4):
            ell = Ellipse(xy=(np.mean(x), np.mean(y)),width=lambda_[0]*j*2, height=lambda_[1]*j*2,angle=theta, color='#c42d2f')
            ell.set_facecolor('none')
            ax.add_artist(ell)
        plt.scatter(x, y, label="Class-1", color="#d37e7e")


        x = d2_x
        y = d2_y
        nstd = 2
        ax = plt.subplot(111)
        cov = np.cov(x, y)
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        lambda_ = np.sqrt(vals)
        for j in xrange(1, 4):
            ell = Ellipse(xy=(np.mean(x), np.mean(y)),width=lambda_[0]*j*2, height=lambda_[1]*j*2,angle=theta, color='#283766')
            ell.set_facecolor('none')
            ax.add_artist(ell)
        plt.scatter(x, y, label="Class-2", color="#6b98bf")

        plt.xlabel('x')
        plt.ylabel('y')


        # x = d3_x
        # y = d3_y
        # nstd = 2
        # ax = plt.subplot(111)
        # cov = np.cov(x, y)
        # vals, vecs = eigsorted(cov)
        # theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        # lambda_ = np.sqrt(vals)
        # for j in xrange(1, 4):
        #     ell = Ellipse(xy=(np.mean(x), np.mean(y)),width=lambda_[0]*j*2, height=lambda_[1]*j*2,angle=theta, color='#1b421f')
        #     ell.set_facecolor('none')
        #     ax.add_artist(ell)
        # plt.scatter(x, y, label="Class-3", color="#7bbf6b")

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
          ncol=2, fancybox=True, shadow=True)

        plt.show()






if __name__=='__main__':
        main()
