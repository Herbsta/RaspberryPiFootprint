import time
from mpu6050 import mpu6050
import math as m
import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Data Record System##############

totalTime = 600
g = 9.80665
sensor = mpu6050(0x68)
sensor.set_accel_range(0x18)
interval = 0.007
acc = 2
datalength = 415
evalPercentage = 0.2

filePath =r'/home/pi/Desktop/Data/'
activeUserNum = None
standStillName = None

names = []

def RotationMatrix(c, theta):
    sinA = m.sin(theta)
    cosA = m.cos(theta)

    o = 1.0 - cosA
    return np.matrix([
        [c[0] * c[0] * o + cosA, c[1] * c[0] * o - sinA * c[2], c[2]*c[0]*o + sinA*c[1]],
        [c[0] * c[1] * o + sinA*c[2], c[1] * c[1] * o + cosA, c[2]*c[1]*o - sinA*c[0]],
        [c[0]*c[2]*o - sinA*c[1], c[1]*c[2]*o + sinA* c[0],c[2]*c[2]*o +cosA]
        
        ])


def rotateAlign(v1,v2):
    cross = np.cross(v1,v2)
    normal = np.linalg.norm(cross)
    

    axis = cross / normal
    
    dot = v1.dot(v2)

    angleRadians = m.acos(dot)
    result = RotationMatrix(axis,angleRadians)
    
    return result

def Normalise(x,y,z):
    total = m.sqrt(m.pow(x,2) + m.pow(y,2) + m.pow(z,2))
    return x/total,y/total,z/total,total

def returnMatrix(x,y,z):
    v1 = np.array([x,y,z])
    v2 = np.array([0,0,-1])

    return rotateAlign(v1,v2)

def GetDataOneSet():
    array = []
    deltaT = time.time()
    gravity = [0,0,0]


    for i in range(datalength):
    
        data = sensor.get_accel_data()
        deltaT = round(time.time() - deltaT,acc)
        data = [round(data['x'],acc),round(data['y'],acc),round(data['z'],acc)]
        array.append([data[0],data[1],data[2],deltaT])
        gravity[0] += data[0]
        gravity[1] += data[1]
        gravity[2] += data[2]
        

        deltaT = time.time()
        time.sleep(interval)

    gravity[0] /= datalength
    gravity[1] /= datalength
    gravity[2] /= datalength

    gravity[0] = round(gravity[0],acc)
    gravity[1] = round(gravity[1],acc)
    gravity[2] = round(gravity[2],acc)
    
    dictionary = {}
    dictionary["RawData"] = array
    dictionary["Gravity"] = gravity
    return dictionary
    


def GetData(tt):
    points = [] 
    startT = time.time()

    while (time.time() - startT < tt):
         
        
        points.append(GetDataOneSet())
    return points



    
            
        
    

class DataSet():

    def __init__(self,username):
        self.name = username
        self.data = []

    def LogData(self):
        self.data = GetData(totalTime)


    
    def WriteToFile(self):
        if(os.path.exists(filePath + self.name)):
            file = open(filePath + self.name,'rb')
            info = pickle.load(file)
            file.close()

            for i in info:
                self.data.append(i)
                
        file = open(filePath + self.name,'wb')
        pickle.dump(self.data,file)
        file.close()


def AddUser():
    name = input("Enter Name: ")
    print("Please walk continuously for 10 minutes. The clock will start in 1 minute")
    print("When walking, please keep at your normal steady speed and try and avoid turning and non-flat terrain (some turnings and uneven terrain is fine)")

    time.sleep(60)

    
    Set1 = DataSet(name)

    print("Logging Data From Now")
    Set1.LogData()
    Set1.WriteToFile()

#########################################

#AI Build System

def NormaliseDataSet(data):
    dataset = []
    
    for d in data:
        
        rawData = d["RawData"]
        gravity = d["Gravity"]

        x,y,z,total = Normalise(gravity[0],gravity[1],gravity[2])
        x,y,z = round(x,acc),round(y,acc),round(z,acc)

        rotationMatrix = returnMatrix(x,y,z)
        partSet = []
        

        for values in rawData:
            
            numpyVector = np.array([values[0],values[1],values[2]])
            #rotation data so that gravity lines up to [0,0,-1]
            newData = rotationMatrix.dot(numpyVector)
            newData = [round(newData[0,0],acc),round(newData[0,1],acc),round(newData[0,2],acc),values[3]]
            #print(newData)

            partSet.append(newData)
            

        dataset.append(partSet)

    return dataset
            
            


def Read():
    dataset = []
    numNames = []

    test_dataset = []
    test_numNames = []
    
    
    for i,file in enumerate(os.listdir(filePath)):
        print("Loading: " + file)    
        file = open(filePath + file ,'rb')
        data = pickle.load(file)
        file.close()

        names.append(file)
        data = NormaliseDataSet(data)
        
        for m,j in enumerate(data):
            length = len(data)
            numToEval = int(length * evalPercentage)

            if(m < numToEval):
                test_dataset.append(j)
                test_numNames.append(i)
            else:
                
                dataset.append(j)
                numNames.append(i)
    

    
    
    train_data = tf.constant(dataset,shape=(len(dataset),datalength,4))
    train_data = tf.reshape(train_data, [-1,datalength*4])
    train_labels = tf.constant(numNames)

    test_data = tf.constant(test_dataset,shape=(len(test_dataset),datalength,4))
    test_data = tf.reshape(test_data, [-1,datalength*4])
    test_labels = tf.constant(test_numNames)

    print("Reshaped data...")

    print("Creating Model")
    #Model Section - might need to tweak values
    
    inputs = keras.Input(shape=(datalength*4,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)

    outputs = layers.Dense(len(numNames),activation='softmax')(x)


    
    def Fit(inputs,outputs):
        model = keras.Model(inputs = inputs,outputs=outputs, name="Test")
        model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      optimizer=keras.optimizers.Adam(),
                      metrics=["accuracy"],
                      )
        history = model.fit(train_data,train_labels, epochs=30,steps_per_epoch=2,verbose=2)
        return model

    model = Fit(inputs,outputs)
    print()
    print()
    
    e = model.evaluate(test_data,test_labels,steps=2,verbose=2)
        


    

    return model,names
    

    


    

        

    

        
    
    




#########################################

#Select Active User System

def ActiveUser(names):
    print()
    for j,i in enumerate(names):
        print(str(i) + ": ",end='')
        print(j)


    
    
    while True:
        inp = int(input("Please Enter number: "))

        for i,j in enumerate(names):
            if i ==inp:
                return j
        print("This is not An Option")
    

        

#########################################

#Activate System

def Activate(model):
    while True:
        dictionary = GetDataOneSet()
        data = NormaliseDataSet([dictionary])

        data = tf.constant(data,shape=(1,datalength,4))
        data = tf.reshape(data, [-1,datalength*4])
        l = model.predict(data,steps=2)
        
        #print(names[np.argmax(l)])
        if names[np.argmax(l)] == activeUserNum:
            print("Correct Person")
        elif names[np.argmax(l)] == standStillName:
            print("Standing Still")
        else:
            print("Wrong Person")
                

        
        
        

#########################################




#########################################

#Menu System

while True:
    print("\n\n\n")
    try:
        num = int(input("Enter 0 to Record Data\nEnter 1 to Build Model\nEnter 2 to set active user\nEnter 3 to activate\n"))
    except:
        print("that is not a Number")
        continue

    if num == 0:
        AddUser()

    elif num == 1:
        tf.keras.backend.clear_session()
        model,names = Read()
    elif num == 2:
        print("Select active user")
        activeUserNum = ActiveUser(names)
        print("Select Stand Still data")
        standStillName = ActiveUser(names)
    elif num == 3:
        Activate(model)
        


            
    

    
    

              




















    
                               
