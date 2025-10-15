import numpy as np
import array
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from river import metrics, ensemble, preprocessing
from river.forest import ARFClassifier
from river.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
import csv
import os
import copy
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA

incrementalLearerGlobal = None
train_df = None
raw_df = None
model = None
XTest = None
YTest = None

class dataProcessing:
  def __init__(self, dfData,refinedColumns,target_Columns):  # Constructor method
        self.dfData = dfData.copy()
        self.dfRawData = dfData.copy()
        self.refinedColumns = refinedColumns
        self.target_Columns = target_Columns

  def __init__(self, dfData):  # Constructor method
        self.dfData = dfData.copy()
        self.dfRawData = dfData.copy()
        self.refinedColumns = None
        self.target_Columns = None

  def scaleData(self,dataX):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    if self.refinedColumns is None:
      refinedX= dataX
    else:
      refinedX = self.dfData[self.refinedColumns]
    scaledX = scaler.fit_transform(refinedX)
    return scaledX

  def getX(self):
    columns_to_drop = ['bearings','wpump','radiator','exvalve','Timestamp']
    refinedX = self.dfData.drop(columns_to_drop, axis=1,errors='ignore')
    return refinedX

  def getProcessedX(self):
    return self.scaleData(self.getX())

  def getProcessedY(self,targetColumn):

    if targetColumn is not None:
      self.y = self.dfRawData[targetColumn]
    elif self.target_Columns is not None:
      self.y = self.dfRawData[self.target_Columns]
    else:
      self.y = self.dfRawData['bearings']
    return self.y

  def getDuplicateValueCount(self):
    dupes = self.dfData.duplicated()
    return sum(dupes)

  def get_summary(self):
    """Return basic statistics."""
    return self.dfData.describe()

  def dropColumns(self,columnsToRemove):
    self.dfData.drop(columnsToRemove, axis=1,errors='ignore', inplace=True)

  def get_rawdata(self):
    return self.dfRawData

  def get_data(self):
    return self.dfData

  def getMissingValueList(self):
    return pd.DataFrame( self.dfData.isnull().sum(), columns= ['Number of missing values'])

  def addTimestamp(self,deltaDays):
    now = datetime.now()
    histEdDate = now - timedelta(days=deltaDays)
    histStDate = histEdDate - timedelta(days=(len(self.dfData)/24))
    print(histStDate)

    formatted_time = histStDate.strftime("%Y-%m-%d %H:%M:%S")
    formatted_time = pd.to_datetime(formatted_time)
    self.dfData['Timestamp'] = [formatted_time + pd.Timedelta(hours=i) for i in range(len(self.dfData))]

class incrementalLearning:
    
    def __init__(self, dfDataX,dfDataY,model,target_Columns):  # Constructor method
        self.dfDataX = dfDataX
        self.dfDataY = dfDataY
        self.model =  model
        self.target_Columns = target_Columns

        self.accuracy = metrics.Accuracy()
        self.f1 = metrics.F1()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.conf_matrix =metrics.ConfusionMatrix()
        self.kappa = metrics.CohenKappa()

        self.y_true = []
        self.y_scores = []


    def __init__(self, dfDataX,dfDataY,target_Columns):  # Constructor method
        self.dfDataX = dfDataX
        self.dfDataY = dfDataY
        self.target_Columns = target_Columns

        self.accuracy = metrics.Accuracy()
        self.f1 = metrics.F1()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.conf_matrix = metrics.ConfusionMatrix()
        self.kappa = metrics.CohenKappa()

        self.model11 = StandardScaler() | ARFClassifier(seed=42,n_models=6,drift_detector=None)

        self.y_true = []
        self.y_scores = []

    def setModel(self,model):
        self.model11 = model

    def getModel(self):
        return self.model11

    def resetMetrices(self):
        self.accuracy = metrics.Accuracy()
        self.f1 = metrics.F1()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()

    def processXData(self,dfNewDataX):
      self.processedDFX = dfNewDataX
      if dfNewDataX is not None:
         pDataX= pd.DataFrame(dfNewDataX)
      else:
         pDataX= pd.DataFrame(self.dfDataX)
      self.processedX = pDataX.to_dict(orient='records')

    def processYData(self,dfNewDataY):
      self.processedDFY = dfNewDataY
      if dfNewDataY is not None:
        pDataY= pd.DataFrame(dfNewDataY)
      else:
        pDataY= pd.DataFrame(self.dfDataY)
      self.processedY = pDataY[self.target_Columns].tolist()

    def getLastProcessedData(self):
        return self.processedX,self.processedY;

    def getLastProcessedXData(self):
        return self.processedDFX;

    def getLastProcessedYData(self):
        return self.processedDFY;

    def learn(self):            # Instance method
        print("Learn Value")
        for x, y in zip(self.processedX,self.processedY):
          self.model11.learn_one(x, y)

    def predict(self):  # Another method
        print("Predict Value")
        y_predValue = []
     
        for x in zip(self.processedX):
          y_pred = self.model11.predict_one(x[0])
          y_predValue.append(y_pred)
        return y_predValue

    def getMetrices(self):  # Another method

        for x, y in zip(self.processedX,self.processedY):
          y_predp = self.model11.predict_proba_one(x)
          y_pred = self.model11.predict_one(x)
          prob = y_predp.get(True, 0.0)  # Probability of class True
          self.y_true.append(y)
          self.y_scores.append(prob)
          self.accuracy.update(y, y_pred)
          self.f1.update(y, y_pred)
          self.precision.update(y, y_pred)
          self.recall.update(y, y_pred)
          self.conf_matrix.update(y_true=y, y_pred=y_pred)
          self.kappa.update(y_true=y, y_pred=y_pred)
        print(f"Accuracy:  {self.kappa.get():.4f}")
        print(f"Kappa:  {self.accuracy.get():.4f}")
        print(f"F1 Score:  {self.f1.get():.4f}")
        print(f"Precision: {self.precision.get():.4f}")
        print(f"Recall:    {self.recall.get():.4f}")
        print("📊 Confusion Matrix:")
        print(self.conf_matrix)
        return self.accuracy,self.kappa,self.f1,self.precision,self.recall,self.conf_matrix

    def getThresholdConfidence(self):
      precision = []
      recall = []
      f1Score = []
      threshold = []

      for t in [0.1 * i for i in range(1, 10)]:
        y_pred = [1 if p >= t else 0 for p in  self.y_scores]
        threshold.append(t)
        precision.append(precision_score(self.y_true, y_pred))
        recall.append(recall_score(self.y_true, y_pred))
        f1Score.append(f1_score(self.y_true, y_pred))
        print(f"Threshold: {t:.1f} | Precision: {precision_score(self.y_true, y_pred):.2f} | Recall: {recall_score(self.y_true, y_pred):.2f} | F1Score: {f1_score(self.y_true, y_pred)}")
      return precision,recall,f1Score,threshold

    def getAccuracy(self):
        return self.accuracy.get()

    def getF1(self):
        return self.f1.get()

    def getPrecision(self):
        return self.precision.get()

    def getRecall(self):
        return self.recall.get()

    def getKappa(self):
        return self.kappa.get()

    def getConfMatrix(self):
        return self.conf_matrix


def initilizeModel():
    
    print("Start Initialization")
    resetModel()

    df = pd.read_csv('C:/Users/sshankar/Downloads/AircompressorDataset_Raw.csv')

    global raw_df
    global train_df 
    global model 
    raw_df = df

    X_train1, X_test1, y_train1, y_test1 = train_test_split(df, df['bearings'], test_size=0.3, random_state=42)
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train1, columns=['rpm', 'air_flow', 'noise_db', 'water_outlet_temp', 'water_flow', 'gaccx', 'haccx'])
    y_train_df = pd.DataFrame(y_train1, columns=["bearings"])
    z_train_df = pd.DataFrame(y_train1, columns=["prediction"])
  
  
    # Reset indices to align properly
    X_train_df = X_train_df.reset_index(drop=True)
    y_train_df = y_train_df.reset_index(drop=True)
    z_train_df = z_train_df.reset_index(drop=True)

    # Concatenate along columns
    train_df1 = pd.concat([X_train_df, y_train_df,z_train_df], axis=1)

    # Get current time in desired format
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add new column with same timestamp for all rows
    train_df1['DetectionTimestamp'] = ""

    dataProcessor = dataProcessing(train_df1)
    dataProcessor.addTimestamp(60)

    train_df = copy.deepcopy(dataProcessor.get_data())

    columns_to_drop = ['DetectionTimestamp','Timestamp','acmotor', 'wpump', 'radiator', 'exvalve','outlet_temp','water_inlet_temp','oil_tank_temp','wpump_outlet_press','outlet_pressure_bar','haccz','gaccy','gaccz','oilpump_power','wpump_power','torque','haccy','motor_power','Timestamp']
    dataProcessor.dropColumns(columns_to_drop)

    dat = dataProcessor.get_rawdata()
    column_names = dat.columns.tolist()

    processedX = dataProcessor.getProcessedX()
    processedY = dataProcessor.getProcessedY('bearings')

    X_train, X_test, y_train, y_test = train_test_split(processedX, processedY, test_size=0.2, random_state=42)
    

    incrementalLearer = incrementalLearning(X_train,y_train,'bearings')

    incrementalLearer.processXData(None)
    incrementalLearer.processYData(None)
  
    incrementalLearer.learn()

    incrementalLearer.processXData(X_test)
    incrementalLearer.processYData(y_test)

    predictedValues = incrementalLearer.getMetrices()

    global incrementalLearerGlobal

    incrementalLearerGlobal = copy.deepcopy(incrementalLearer)
    model = copy.deepcopy(incrementalLearerGlobal.getModel)

    addOutputFile(train_df)
    return incrementalLearer

def trainModel():
   incrementalLearerGlobal.learn()
   

def resetModel():

  global incrementalLearerGlobal
  incrementalLearerGlobal = None
  global train_df
  train_df = None
  global raw_df
  raw_df = None
  global model
  model = None
  global XTest
  global YTest
  XTest = None
  YTest = None
   
  # Specify the file path
  file_path = "C:/Users/sshankar/Downloads/PredictiveMaintanance_OutputFile.csv"

  # Check if the file exists before attempting to delete
  if os.path.exists(file_path):
      os.remove(file_path)
      print(f"{file_path} has been deleted.")
  else:
      print(f"{file_path} does not exist.")


def addOutputFile(output_train_df):
  
  # Define the file name
  file_path  = "C:/Users/sshankar/Downloads/PredictiveMaintanance_OutputFile.csv"

  # Ensure the directory exists
  os.makedirs(os.path.dirname(file_path), exist_ok=True)

  # Check if the file exists
  file_exists = os.path.isfile(file_path)

  output_train_df.to_csv(file_path, mode='a', header=not file_exists, index=False)


def SimulateBatch1(timestamp1):

  batchDF = raw_df.sample(frac=0.1)  

  global train_df 
  global incrementalLearerGlobal 
  global model
  global XTest
  global YTest

  dataProcessor = dataProcessing(batchDF)
  dataProcessor.addTimestamp(60)
  columns_to_drop = ['id','Timestamp','acmotor', 'wpump', 'radiator', 'exvalve','outlet_temp','water_inlet_temp','oil_tank_temp','wpump_outlet_press','outlet_pressure_bar','haccz','gaccy','gaccz','oilpump_power','wpump_power','torque','haccy','motor_power','Timestamp']
  dataProcessor.dropColumns(columns_to_drop)

  dat = dataProcessor.get_data()
  column_names = dat.columns.tolist()
  print(column_names)

  processedX = dataProcessor.getProcessedX()
  processedY = dataProcessor.getProcessedY('bearings')

  incrementalLearerGlobal.processXData(processedX)
  incrementalLearerGlobal.processYData(processedY)

  yPredict =incrementalLearerGlobal.predict()

  incrementalLearerGlobal.getMetrices()

  # Convert to DataFrames
  z_train_df = pd.DataFrame(yPredict, columns=["prediction"])

  
  # Reset indices to align properly
  dat = dat.reset_index(drop=True)
  z_train_df = z_train_df.reset_index(drop=True)

  # Concatenate along columns
  train_df1 = pd.concat([dat,z_train_df], axis=1)
  # Get current time in desired format
  current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  # Add new column with same timestamp for all rows
  train_df1['DetectionTimestamp'] = current_time

  dataProcessor = dataProcessing(train_df1)

  dataProcessor.addTimestamp(timestamp1)
  train_df1 = copy.deepcopy(dataProcessor.get_data())
  addOutputFile(train_df1)
  return incrementalLearerGlobal
  

def labelAndLearnBatch():
  
  global incrementalLearerGlobal

  incrementalLearerGlobal.learn()

  return incrementalLearerGlobal

def getProcessedData():


  filtered_df = None
  file_path  = "C:/Users/sshankar/Downloads/PredictiveMaintanance_OutputFile.csv"

  # Check if the file exists
  file_exists = os.path.isfile(file_path)
  
  if file_exists:
    # Select specific columns
    df = pd.read_csv(file_path)
    columns_to_show = ['rpm', 'air_flow', 'noise_db', 'water_outlet_temp',
                    'water_flow', 'gaccx', 'haccx', 'bearings','Timestamp','DetectionTimestamp','prediction']
    filtered_df = df[columns_to_show]
  return file_exists, filtered_df


def getAirFlowForcast(filtered_df):
  filtered_df = filtered_df.head(50)
  airFlowDF = filtered_df[['air_flow','Timestamp']]
  airFlowDF['isForcasted'] = 0
  model = ARIMA(airFlowDF['air_flow'], order=(1,1,0))  # (p,d,q)
  fit = model.fit()

  # Forecast next 5 points
  forecast = fit.forecast(20)
  print(forecast)
  forecastDF = forecast.to_frame(name='air_flow')
  forecastDF['isForcasted'] = 1
  start_time = datetime.now()
  forecastDF['Timestamp'] = [start_time + timedelta(hours=2*i) for i in range(len(forecastDF))]

  airFlowDF = airFlowDF.reset_index(drop=True)
  forecastDF = forecastDF.reset_index(drop=True)
  print(forecastDF)
  # Concatenate along columns
  finalAirflowForcast = pd.concat([forecastDF,airFlowDF], ignore_index=True)
  print(finalAirflowForcast.columns.tolist())

  return finalAirflowForcast


def getNoiceDBForcast(filtered_df):
  filtered_df = filtered_df.head(20)
  noiceDBFlowDF = filtered_df[['noise_db','Timestamp']]
  noiceDBFlowDF['isForcasted'] = 0
  model = ARIMA(noiceDBFlowDF['noise_db'], order=(1,1,0))  # (p,d,q)
  fit = model.fit()

  # Forecast next 5 points
  forecast = fit.forecast(35)
  print(forecast)
  forecastDF = forecast.to_frame(name='noise_db')
  forecastDF['isForcasted'] = 1
  start_time = datetime.now()
  forecastDF['Timestamp'] = [start_time + timedelta(hours=2*i) for i in range(len(forecastDF))]

  noiceDBFlowDF = noiceDBFlowDF.reset_index(drop=True)
  forecastDF = forecastDF.reset_index(drop=True)
  print(forecastDF)
  # Concatenate along columns
  finalNoiceDBForcast = pd.concat([forecastDF,noiceDBFlowDF], ignore_index=True)
  print(finalNoiceDBForcast.columns.tolist())

  return finalNoiceDBForcast


def getPowerConsumptionForcast():

  batchDF = raw_df.sample(frac=0.1)  

  dataProcessor = dataProcessing(batchDF)
  dataProcessor.addTimestamp(60)

  filtered_df = dataProcessor.get_data()

  filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'], errors='coerce')
  filtered_df = filtered_df.sort_values(by='Timestamp')

  filtered_df = filtered_df.head(100)

  powerDF = filtered_df[['motor_power','Timestamp']]

  print(powerDF)

  powerDF['isForcasted'] = 0
  model = ARIMA(powerDF['motor_power'], order=(1,1,0))  # (p,d,q)
  fit = model.fit()

  # Forecast next 5 points
  forecast = fit.forecast(40)
  print(forecast)
  forecastDF = forecast.to_frame(name='motor_power')
  forecastDF['isForcasted'] = 1
  start_time = datetime.now()
  forecastDF['Timestamp'] = [start_time + timedelta(hours=2*i) for i in range(len(forecastDF))]

  powerDF = powerDF.reset_index(drop=True)
  forecastDF = forecastDF.reset_index(drop=True)
  print(forecastDF)
  # Concatenate along columns
  finalPCForcast = pd.concat([forecastDF,powerDF], ignore_index=True)
  print(finalPCForcast.columns.tolist())

  return finalPCForcast


