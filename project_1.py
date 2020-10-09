import sys
import subprocess
import pkg_resources

required = {'xlrd', 'openpyxl',
            'numpy','tensorflow==2.1.0',
            'matplotlib'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    try:
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing, '--user'], stdout=subprocess.DEVNULL)
    except:
        pass

import xlrd
import openpyxl as xlwt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

file_location = input("Enter the full location (file path) of the Excel file: ")
doc_name = file_location.split('/')
doc_name = doc_name[-1].split('.')[0]

def get_winning_numbers(path=file_location):
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_name('Winning Numbers')
    column = sheet.col(0)
    column = [x.value for x in column]
    column = [str(x) for x in column]
    column = [x for x in column if x != '']
    column = [x for x in column if '-1' not in x]
    column = [x.split(' ') for x in column]
    column = [[str(int(float(x))) for x in y] if len(y)!=1 else y for y in column]
    column = [''.join(x) for x in column]
    column = [x if len(x)==3 else str(int(float(x))) for x in column]
    return column
    
    

def get_final_picks(path=file_location):
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_name('Final Picks')

    column = sheet.col(2)
    column = [str(x.value) for x in column]
    column = [x for x in column if x != '']
    return column

def get_frequencies(path=file_location):
    nums = get_winning_numbers(path)
    picks = get_final_picks(path)
    occurrences = []
    for pick in picks:
        current = 0
        for num in nums:
            counter = 0
            for x in pick:
                if x in num:
                    counter += 1
            if counter == len(pick):
                current += 1
        occurrences.append(current)
    return occurrences, picks, nums

def update_frequencies(path=file_location):
    occs = get_frequencies(path)
    workbook = xlwt.load_workbook(path)
    sheet = workbook['Final Picks']
    for x in range(len(occs[0])):
        sheet['D'+str(x+1)].data_type='s'
        sheet['D'+str(x+1)] = str(occs[0][x]) + ' / ' + str(len(occs[-1]))
    workbook.save(path)

def make_model(num_look_back_steps=10):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((num_look_back_steps*3,)))
    model.add(tf.keras.layers.Dense(200, 'relu'))
    model.add(tf.keras.layers.Dense(100, 'relu'))
    model.add(tf.keras.layers.Dense(50, 'relu'))
    model.add(tf.keras.layers.Dense(10, 'relu'))
    model.add(tf.keras.layers.Dense(3))
    model.compile('adam','mse')
    return model

def train_model(path=file_location, num_look_back_steps=10):
    model = make_model(num_look_back_steps)
    data = get_winning_numbers(path)
    data = [[int(y) for y in x] for x in data]
    inputs = [data[x-num_look_back_steps:x] for x in range(num_look_back_steps,len(data))]
    outputs = [data[x] for x in range(num_look_back_steps, len(data))]
    inputs = np.array(inputs).astype(np.float32)
    inputs = np.array([x.flatten() for x in inputs])
    outputs = np.array(outputs).astype(np.float32)
    response = model.fit(inputs, outputs, epochs = 50, verbose=0)
    return model

def load_model(path='pick3_model_'+doc_name+'.h5'):
    model = tf.keras.models.load_model(path)
    return model

def save_model(model,path='pick3_model_'+doc_name+'.h5'):
    model = model.save(path)
    return 1

def load_up_that_model(path=file_location, num_look_back_steps=10):
    try:
        model = load_model()
    except:
        model = train_model(path, num_look_back_steps)
        save_model(model)
    return model

def predict(path=file_location,num_look_back_steps=10):
    model = load_up_that_model(path, num_look_back_steps)
    data = get_winning_numbers(path)
    data = [[int(y) for y in x] for x in data]
    inputs = [data[x-num_look_back_steps:x] for x in range(num_look_back_steps,len(data))]
    inputs = np.array(inputs).astype(np.float32)
    inputs = np.array([x.flatten() for x in inputs])[:1]
    predictions = model.predict(inputs)
    predictions = np.round(predictions).astype(np.int32)
    predictions = list(predictions[0])
    predictions = [x if x!=10 else 9 for x in predictions]
    predictions = [str(x) for x in predictions]
    predictions = ''.join(predictions)
    return predictions

def most_likely_from_grid(path=file_location):
    data = get_final_picks(path)
    indices = np.random.uniform(0,len(data),4)
    indices = [int(x) for x in indices]
    most_likely = [data[x] for x in indices]
    return most_likely

def most_likely(path=file_location,num_look_back_steps=10):
    predictions = most_likely_from_grid(path)
    predictions.append(predict(path, num_look_back_steps))
    return predictions

update_frequencies()
predictions = most_likely(file_location)
predictions = [' '.join([x for x in y]) for y in predictions]
indices = [x for x in range(len(predictions))]
np.random.shuffle(indices)
print('5 most likely combinations:')
for x in range(len(indices)):
    print('\t'+str(x+1)+'.\t'+predictions[indices[x]])
