from nicegui import ui
import pandas as pd
import datetime
import random
import CompressorTraining
from statsmodels.tsa.holtwinters import Holt

resetButton = None
initializeButton = None
SimulateB1Button = None
LNRBatch1 = None
SimulateB2Button = None
LNRBatch2 = None
SimulateB3Button = None
LNRBatch3 = None
SimulateB4Button = None
LNRBatch4 = None
evaluationTable = None
confMatrixTable = None
infoMessageField = None
AnomalyDetailsTable = None
AnomalyDetailsChart = None
col_to_update = None
PowerConsumptionChart = None
AirFlowChart = None
NoiceDBChart = None
MotorPowerChart = None
col3_tab1_update = None




def generate_line_data():
    return {
        'x': list(range(10)),
        'y': [random.randint(0, 100) for _ in range(10)]
    }

with ui.column().classes('w-full gap-0'):
    ui.label('Predictive Maintanance & Anomaly Detection').classes(
    'text-2xl font-bold text-white bg-blue-600 p-2 w-full rounded-t text-center sticky top-0 z-50'
    )

    ui.label('(Single Stage Air Compressors)').classes(
    'text-lg text-gray-100 bg-blue-600 p-2 rounded-b w-full text-center'
    )

def add_cm_filters():
    with ui.row():
        with ui.column():
            with ui.row().classes('items-center gap-x-2'):
                ui.label('Start Date').classes('text-sm text-gray-600')
                ui.input().props('type=datetime-local').classes('w-42 rounded px-1 py-1 border border-gray-200')

        with ui.column():
            with ui.row().classes('items-center gap-x-1'):
                ui.label('End Date').classes('text-sm text-gray-600')
                ui.input().props('type=datetime-local').classes('w-42 rounded px-1 py-1 border border-gray-300')

def tab1_content():
    with ui.row().classes('w-full gap-0'):
        
        global MotorPowerChart
        global col3_tab1_update
        col3_tab1_update = ui.column().classes('w-full items-start gap-0')
        with col3_tab1_update:
            #with ui.card().classes('w-1/2'):
            ui.label('Power Consumption (Historical/Forcasting)').classes('text-xl font-bold m-0 p-0')
            MotorPowerChart = ui.echart({
        'xAxis': {'type': 'category', 'data': []},
        'yAxis': {'type': 'value'},
        'series': [{
            'type': 'line',
            'name': 'RPM',
            'data': []
        }],
        'tooltip': {'trigger': 'axis'}
    }).classes('w-full m-0 p-0')
        
        global AirFlowChart
        global col1_tab1_update
        col1_tab1_update = ui.column().classes('w-full items-start gap-0')
        with col1_tab1_update:
            #with ui.card().classes('w-1/2'):
            ui.label('Air Flow (Historical/Forcasting)').classes('text-xl font-bold m-0 p-0')
            AirFlowChart = ui.echart({
        'xAxis': {'type': 'category', 'data': []},
        'yAxis': {'type': 'value'},
        'series': [{
            'type': 'line',
            'name': 'RPM',
            'data': []
        }],
        'tooltip': {'trigger': 'axis'}
    }).classes('w-full m-0 p-0')
            
        global NoiceDBChart
        global col2_tab1_update
        col2_tab1_update = ui.column().classes('w-full items-start gap-0')
        with col2_tab1_update:
            #with ui.card().classes('w-1/2'):
            ui.label('(Noice(DB) (Historical/Forcasting))').classes('text-xl font-bold m-0 p-0')
            NoiceDBChart = ui.echart({
        'xAxis': {'type': 'category', 'data': []},
        'yAxis': {'type': 'value'},
        'series': [{
            'type': 'line',
            'name': 'RPM',
            'data': []
        }],
        'tooltip': {'trigger': 'axis'}
    }).classes('w-full m-0 p-0')

def tab2_content():
    global AnomalyDetailsChart
    global col_to_update
    with ui.row().classes('gap-x-4 w-full'):
        col_to_update = ui.column().classes('w-full items-start')
        with col_to_update:
            #with ui.card().classes('w-1/2'):
            ui.label('RPM Data - With Anomaly Indicator').classes('text-xl font-bold')
            AnomalyDetailsChart = ui.echart({
        'xAxis': {'type': 'category', 'data': []},
        'yAxis': {'type': 'value'},
        'series': [{
            'type': 'line',
            'name': 'RPM',
            'data': []
        }],
        'tooltip': {'trigger': 'axis'}
    }).classes('w-full')

    with ui.row().classes('gap-x-4 w-full'):  
        with ui.column().classes('w-full items-start'):
            ui.label('Predicted Anomaly Details').classes('text-xl font-bold')
            global AnomalyDetailsTable

            
            columns = [
                {'name': 'DetectionTimestamp', 'label': 'Detection Timestamp', 'field': 'DetectionTimestamp'},
                {'name': 'rpm', 'label': 'RPM', 'field': 'rpm'},
                {'name': 'air_flow', 'label': 'Air Flow', 'field': 'air_flow'},
                {'name': 'noise_db', 'label': 'Noice(db)', 'field': 'noise_db'},
                {'name': 'noise_db', 'label': 'Water Outlet Temp', 'field': 'noise_db'},
                {'name': 'noise_db', 'label': 'Water Flow', 'field': 'noise_db'},
                {'name': 'gaccx', 'label': 'GACCX', 'field': 'gaccx'},
                {'name': 'haccx', 'label': 'HACCX', 'field': 'haccx'},
                {'name': 'Timestamp', 'label': 'Record Timestamp', 'field': 'Timestamp'},
                {'name': 'prediction', 'label': 'Predicted Value', 'field': 'prediction'},
            ]
            rows = []
            AnomalyDetailsTable = ui.table(columns=columns, rows=rows,row_key='rpm').classes('w-full')
            AnomalyDetailsTable.add_slot('body-cell-prediction', '''
    <q-td key="agprediction" :props="props">
        <q-badge :color="props.value == "1" ? 'red' : 'green'">
            {{ props.value }}
        </q-badge>
    </q-td>
''')

def row_class(row):
    return 'bg-green-100' if row['prediction'] == 'ok' else ''

def tab3_content():
    with ui.row().classes('gap-x-4 w-full'):
        global initializeButton
        global resetButton
        initializeButton = ui.button('Initialize & Create Model', on_click=lambda: ui.notify(onButtonClick_Initialize()),color='purple').classes('w-60')
        resetButton = ui.button('Reset All', on_click=lambda: ui.notify(onButtonClick_ResetModel()),color='purple').classes('w-60')
        resetButton.disable()

        # This creates a clean horizontal separator

        ui.separator().classes('my-4 border-t-2 border-gray-500').style('margin-top: 0px; margin-bottom: 0px;')
   
        with ui.row().classes('gap-x-4 w-full'):

            with ui.column().classes():

                with ui.row().classes('gap-x-4'):
                    global LNRBatch1
                    global SimulateB1Button

                    SimulateB1Button = ui.button('Simulate Batch1', on_click=lambda: ui.notify(onButtonClick_SimulateB1()),color='orange').classes('w-40')
                    LNRBatch1 = ui.button('Label & Retrain', on_click=lambda: ui.notify(onButtonClick_LR1()),color='orange').classes('w-40')
                    LNRBatch1.disable()
                    SimulateB1Button.disable()

                with ui.row().classes('gap-x-4'):
                    global LNRBatch2
                    global SimulateB2Button
                    SimulateB2Button = ui.button('Simulate Batch 2', on_click=lambda: ui.notify(onButtonClick_SimulateB2()),color='orange').classes('w-40')
                    LNRBatch2 = ui.button('Label & Retrain', on_click=lambda: ui.notify(onButtonClick_LR2()),color='orange').classes('w-40')
                    LNRBatch2.disable()
                    SimulateB2Button.disable()

                with ui.row().classes('gap-x-4'):
                    global LNRBatch3
                    global SimulateB3Button
                    SimulateB3Button = ui.button('Simulate Batch 3', on_click=lambda: ui.notify(onButtonClick_SimulateB3()),color='orange').classes('w-40')
                    LNRBatch3= ui.button('Label & Retrain', on_click=lambda: ui.notify(onButtonClick_LR3()),color='orange').classes('w-40')
                    LNRBatch3.disable()
                    SimulateB3Button.disable()

                with ui.row().classes('gap-x-4'):
                    global LNRBatch4
                    global SimulateB4Button
                    SimulateB4Button = ui.button('Simulate Batch 4', on_click=lambda: ui.notify(onButtonClick_SimulateB4()),color='orange').classes('w-40')
                    LNRBatch4 = ui.button('Label & Retrain', on_click=lambda: ui.notify(onButtonClick_LR4()),color='orange').classes('w-40')
                    LNRBatch4.disable()
                    SimulateB4Button.disable()

            with ui.column().classes():
                with ui.row().classes('gap-x-4 w-full'):
                    # Wrap it in a styled rectangle box
                    with ui.card().classes('w-full max-w-md mx-auto p-4 bg-gray-100 shadow-md ').style('height: 200px; width: 400px; display: flex; align-items: center;'):
                        global infoMessageField
                        # Add a label inside the card
                        infoMessageField = ui.label('Simulator is Ready for Action').classes('text-red-500 font-bold').style('text-align: left; white-space: pre-line; width: 100%')

        ui.separator().classes('my-4 border-t-2 border-gray-500').style('margin-top: 0px; margin-bottom: 0px;')

        with ui.row().classes('gap-0').style('margin: 0; padding: 0;'):
                ui.label('Evaluation Metrices').classes('w-[33rem] text-lg text-gray-100 bg-blue-600 p-1 rounded-b w-full text-left')
        ui.separator().classes('my-4 border-t-2 border-gray-500').style('margin-top: 0px; margin-bottom: 0px;')   

        with ui.row().classes('gap-0').style('margin: 0; padding: 0;'):
            with ui.column().classes():
                global evaluationTable

                # Define table headers and rows
                ui.label('Evaluation-Scores')
                columns = [
                    {'name': 'col1', 'label': '', 'field': 'name'},
                    {'name': 'col2', 'label': 'Scores', 'field': 'Scores'},
                ]

                rows = [
                    {'name': 'Accuracy', 'Scores': 0},
                    {'name': 'F1', 'Scores': 0},
                    {'name': 'Precision', 'Scores': 0},
                    {'name': 'Recall', 'Scores': 0},
                    {'name': 'Kappa', 'Scores': 0},
                ]

                # Create the table
                evaluationTable = ui.table(columns=columns, rows=rows).classes('w-full').props('dense bordered')

            with ui.column().classes('flex-1'):
                global confMatrixTable
                # Define table headers and rows
                ui.label('Confusion Matrix')
                columns = [
                    {'name': 'col1', 'label': '', 'field': 'Label1'},
                    {'name': 'col2', 'label': 'positive', 'field': 'positive'},
                    {'name': 'col3', 'label': 'negative', 'field': 'negative'},
                ]

                rows = [
                    {'Label1': 'positive', 'positive': 0,'negative':0},
                    {'Label1': 'negative', 'positive': 0,'negative':0},
                ]

                # Create the table
                confMatrixTable = ui.table(columns=columns, rows=rows).classes('w-full').props('dense bordered')
        

def table_Update(incrementalLearer):
    global evaluationTable
    global confMatrixTable

    new_data1 = [
                {'name': 'Accuracy', 'Scores': (round(incrementalLearer.getAccuracy(), 2))*100},
                {'name': 'F1', 'Scores':(round(incrementalLearer.getF1(), 2))*100},
                {'name': 'Precision', 'Scores': (round(incrementalLearer.getPrecision(), 2))*100},
                {'name': 'Recall', 'Scores': (round(incrementalLearer.getRecall(), 2))*100},
                {'name': 'Kappa', 'Scores': (round(incrementalLearer.getKappa(), 2))*100},
                ]


    evaluationTable.rows = new_data1  # Update the table's data
    evaluationTable.update()         # Refresh the U
    
    new_data = [
                {'Label1': 'positive', 'positive': incrementalLearer.getConfMatrix()[1][1],'negative':incrementalLearer.getConfMatrix()[0][1]},
                {'Label1': 'negative', 'positive': incrementalLearer.getConfMatrix()[1][0],'negative':incrementalLearer.getConfMatrix()[0][0] },
                ]


    confMatrixTable.rows = new_data  # Update the table's data
    confMatrixTable.update()  

def table_AnomalyDetailsUpdate(dfUpdate):
    global AnomalyDetailsTable
    filtered_df = dfUpdate[dfUpdate['DetectionTimestamp'].notnull()]
    df_sorted = filtered_df.sort_values(by='Timestamp',ascending=False)
    table_data = df_sorted.to_dict(orient='records')

    AnomalyDetailsTable.rows = table_data  # Update the table's data
    AnomalyDetailsTable.update()         # Refresh the U
    AnomalyDetailsTable.add_slot('body-cell-prediction', '''
    <q-td key="prediction" :props="props">
        <q-badge :color="props.value == 1 ? 'red' : 'green'">
            {{ props.value }}
        </q-badge>
    </q-td>
''')

def table_reset():
    new_data1 = [
                {'name': 'Accuracy', 'Scores': (0)},
                {'name': 'F1', 'Scores':(0)},
                {'name': 'Precision', 'Scores': (0)},
                {'name': 'Recall', 'Scores': (0)},
                {'name': 'Kappa', 'Scores': (0)},
                ]


    evaluationTable.rows = new_data1  # Update the table's data
    evaluationTable.update()         # Refresh the U
    
    new_data = [
                {'Label1': 'positive', 'positive': 0,'negative':0},
                {'Label1': 'negative', 'positive': 0,'negative':0 },
                ]


    confMatrixTable.rows = new_data  # Update the table's data
    confMatrixTable.update()  

# Button to trigger the update

def onButtonClick_Initialize():
    print('Initialize Model')
    incrementalLearer = CompressorTraining.initilizeModel()
    global initializeButton
    global resetButton
    global infoMessageField
    resetButton.enable()
    initializeButton.disable()
    SimulateB1Button.enable()

    table_Update(incrementalLearer)
    infoMessageField.set_text('Model is created and Initilized successfully.\n\nHistorical Data is Processed and Trained Successfully. We will call this data as Historical Batch.\n\n Waiting for Next Batch!!')

def onButtonClick_ResetModel():
    print('Reset Model')
    global initializeButton
    global resetButton
    CompressorTraining.resetModel()
    resetButton.disable()
    initializeButton.enable()

    table_reset()

    global infoMessageField
    infoMessageField.set_text('Model Reset Completed Successfully. Please Initialize and Create Model')

def onButtonClick_LR1():
    print('onButtonClick_LR1')
    global LNRBatch1
    global SimulateB2Button
    LNRBatch1.disable()
    SimulateB2Button.enable()

    incrementalLearer = CompressorTraining.labelAndLearnBatch()
    table_Update(incrementalLearer)

    global infoMessageField
    infoMessageField.set_text('Batch 1 is Labeled & Trained Successfully.\n\nWaiting for Next Batch!!')

def onButtonClick_LR2():
    print('onButtonClick_LR2')
    global LNRBatch2
    global SimulateB3Button
    LNRBatch2.disable()
    SimulateB3Button.enable()

    incrementalLearer = CompressorTraining.labelAndLearnBatch()
    table_Update(incrementalLearer)

    global infoMessageField
    infoMessageField.set_text('Batch 2 is Labeled & Trained Successfully.\n\nWaiting for Next Batch!!')


def onButtonClick_LR3():
    print('onButtonClick_LR3')
    global LNRBatch3
    global SimulateB4Button
    LNRBatch3.disable()
    SimulateB4Button.enable()
    
    incrementalLearer = CompressorTraining.labelAndLearnBatch()
    table_Update(incrementalLearer)

    global infoMessageField
    infoMessageField.set_text('Batch 3 is Labeled & Trained Successfully.\n\nWaiting for Next Batch!!')


def onButtonClick_LR4():
    print('onButtonClick_LR4')
    global LNRBatch4
    LNRBatch4.disable()
    CompressorTraining.labelAndLearnBatch()

    incrementalLearer = CompressorTraining.labelAndLearnBatch()
    table_Update(incrementalLearer)
    
    global infoMessageField
    infoMessageField.set_text('Batch 4 is Labeled & Trained Successfully.\n\nDemo Completed Successfully, Please use Reset Button and start again!!')


def onButtonClick_SimulateB1():
    print('onButtonClick_SimulateB1')
    global LNRBatch1
    global SimulateB1Button
    LNRBatch1.enable()
    SimulateB1Button.disable()

    incrementalLearer = CompressorTraining.SimulateBatch1(30)
    table_Update(incrementalLearer)
    file_exists, filtered_df = CompressorTraining.getProcessedData()
    if file_exists:
        table_AnomalyDetailsUpdate(filtered_df)
        update_AnomalyChartCol(filtered_df)
        airflowDF = CompressorTraining.getAirFlowForcast(filtered_df)
        update_AirflowChartCol(airflowDF)
        noiceDBDF = CompressorTraining.getNoiceDBForcast(filtered_df)
        update_NoiceDBChartCol(noiceDBDF)
        powerConspDF = CompressorTraining.getPowerConsumptionForcast()
        update_PowerChartCol(powerConspDF)
    global infoMessageField
    infoMessageField.set_text('Batch 1 Received and Evaluated Successfully.Please check below given Evaluation Scores.')


def onButtonClick_SimulateB2():
    print('onButtonClick_SimulateB2')
    global LNRBatch2
    global SimulateB2Button
    LNRBatch2.enable()
    SimulateB2Button.disable()

    incrementalLearer = CompressorTraining.SimulateBatch1(30)
    table_Update(incrementalLearer)
    file_exists, filtered_df = CompressorTraining.getProcessedData()
    if file_exists:
        table_AnomalyDetailsUpdate(filtered_df)
        update_AnomalyChartCol(filtered_df)
        airflowDF = CompressorTraining.getAirFlowForcast(filtered_df)
        update_AirflowChartCol(airflowDF)
        noiceDBDF = CompressorTraining.getNoiceDBForcast(filtered_df)
        update_NoiceDBChartCol(noiceDBDF)
        powerConspDF = CompressorTraining.getPowerConsumptionForcast()
        update_PowerChartCol(powerConspDF)

    global infoMessageField
    infoMessageField.set_text('Batch 2 Received and Evaluated Successfully.Please check below given Evaluation Scores.')

def onButtonClick_SimulateB3():
    print('onButtonClick_SimulateB3')
    global LNRBatch3
    global SimulateB3Button
    LNRBatch3.enable()
    SimulateB3Button.disable()

    incrementalLearer = CompressorTraining.SimulateBatch1(30)
    table_Update(incrementalLearer)
    file_exists, filtered_df = CompressorTraining.getProcessedData()
    if file_exists:
        table_AnomalyDetailsUpdate(filtered_df)
        update_AnomalyChartCol(filtered_df)
        airflowDF = CompressorTraining.getAirFlowForcast(filtered_df)
        update_AirflowChartCol(airflowDF)
        noiceDBDF = CompressorTraining.getNoiceDBForcast(filtered_df)
        update_NoiceDBChartCol(noiceDBDF)
        powerConspDF = CompressorTraining.getPowerConsumptionForcast()
        update_PowerChartCol(powerConspDF)

    global infoMessageField
    infoMessageField.set_text('Batch 3 Received and Evaluated Successfully.Please check below given Evaluation Scores.')

def onButtonClick_SimulateB4():
    print('onButtonClick_SimulateB4')
    global LNRBatch4
    global SimulateB4Button
    LNRBatch4.enable()
    SimulateB4Button.disable()

    incrementalLearer = CompressorTraining.SimulateBatch1(30)
    table_Update(incrementalLearer)
    table_Update(incrementalLearer)
    file_exists, filtered_df = CompressorTraining.getProcessedData()
    if file_exists:
        table_AnomalyDetailsUpdate(filtered_df)
        update_AnomalyChartCol(filtered_df)
        airflowDF = CompressorTraining.getAirFlowForcast(filtered_df)
        update_AirflowChartCol(airflowDF)
        noiceDBDF = CompressorTraining.getNoiceDBForcast(filtered_df)
        update_NoiceDBChartCol(noiceDBDF)
        powerConspDF = CompressorTraining.getPowerConsumptionForcast()
        update_PowerChartCol(powerConspDF)

    global infoMessageField
    infoMessageField.set_text('Batch 4 Received and Evaluated Successfully.Please check below given Evaluation Scores.')


def update_AnomalyChartCol(filtered_df1):
    col_to_update.clear()
    with col_to_update:
        ui.label('RPM Data - With Anomaly Indicator').classes('text-xl font-bold')
        filtered_df = filtered_df1[filtered_df1['DetectionTimestamp'].notnull()]
        df_sorted = filtered_df.sort_values(by='Timestamp',ascending=True)


        #prepare data for chart
        df_sorted['Timestamp'] = pd.to_datetime(df_sorted['Timestamp'], errors='coerce')
        x_values = df_sorted['Timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        y_values = df_sorted['rpm'].tolist()

        # Create styled data points
        styled_data = [
            {
                'value': rpm,
                'itemStyle': {'color': 'red' if pred == 1 else 'grey'}
            }
            for rpm, pred in zip(df_sorted['rpm'], df_sorted['prediction'])
        ]
        AnomalyDetailsChart = ui.echart({
    'xAxis': {'type': 'category', 'data': x_values},
    'yAxis': {'type': 'value'},
    'series': [{
        'type': 'line',
        'name': 'RPM',
        'data': styled_data
    }],
    'tooltip': {'trigger': 'axis'}
}).classes('w-full')
        

def update_AirflowChartCol(filtered_df1):
    col1_tab1_update.clear()
    with col1_tab1_update:
        ui.label('Air Flow (Historical/Forcasting)').classes('text-xl font-bold')



        #prepare data for chart
        filtered_df1['Timestamp'] = pd.to_datetime(filtered_df1['Timestamp'], errors='coerce')
        filtered_df1 = filtered_df1.sort_values(by='Timestamp')
        x_values = filtered_df1['Timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        y_values = filtered_df1['air_flow'].tolist()

        # Create styled data points
        styled_data = [
            {
                'value': air_flow,
                'itemStyle': {'color': 'blue' if pred == 1 else 'grey'}
            }
            for air_flow, pred in zip(filtered_df1['air_flow'], filtered_df1['isForcasted'])
        ]
        AirFlowChart = ui.echart({
    'xAxis': {'type': 'category', 'data': x_values},
    'yAxis': {'type': 'value'},
    'series': [{
        'type': 'line',
        'name': 'Air Flow',
        'data': styled_data
    }],
    'tooltip': {'trigger': 'axis'}
}).classes('w-full')
        

def update_PowerChartCol(filtered_df1):
    col3_tab1_update.clear()
    with col3_tab1_update:
        ui.label('Power Consumption (Historical/Forcasting)').classes('text-xl font-bold')



        #prepare data for chart
        filtered_df1['Timestamp'] = pd.to_datetime(filtered_df1['Timestamp'], errors='coerce')
        filtered_df1 = filtered_df1.sort_values(by='Timestamp')
        x_values = filtered_df1['Timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        y_values = filtered_df1['motor_power'].tolist()

        # Create styled data points
        styled_data = [
            {
                'value': motor_power,
                'itemStyle': {'color': 'blue' if pred == 1 else 'grey'}
            }
            for motor_power, pred in zip(filtered_df1['motor_power'], filtered_df1['isForcasted'])
        ]
        MotorPowerChart = ui.echart({
    'xAxis': {'type': 'category', 'data': x_values},
    'yAxis': {'type': 'value'},
    'series': [{
        'type': 'line',
        'name': 'Motor Power',
        'data': styled_data
    }],
    'tooltip': {'trigger': 'axis'}
}).classes('w-full')

def update_NoiceDBChartCol(filtered_df1):
    col2_tab1_update.clear()
    with col2_tab1_update:
        ui.label('Noice DB (Historical/Forcasting)').classes('text-xl font-bold')



        #prepare data for chart
        filtered_df1['Timestamp'] = pd.to_datetime(filtered_df1['Timestamp'], errors='coerce')
        filtered_df1 = filtered_df1.sort_values(by='Timestamp')
        x_values = filtered_df1['Timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        y_values = filtered_df1['noise_db'].tolist()

        # Create styled data points
        styled_data = [
            {
                'value': noise_db,
                'itemStyle': {'color': 'blue' if pred == 1 else 'grey'}
            }
            for noise_db, pred in zip(filtered_df1['noise_db'], filtered_df1['isForcasted'])
        ]
        NoiceDBChart = ui.echart({
    'xAxis': {'type': 'category', 'data': x_values},
    'yAxis': {'type': 'value'},
    'series': [{
        'type': 'line',
        'name': 'Air Flow',
        'data': styled_data
    }],
    'tooltip': {'trigger': 'axis'}
}).classes('w-full')

with ui.tabs().classes('justify-start flex gap-x-6 rounded-[8px] px-2 py-2 bg-blue-500 text-white ') as tabs:
    tab1 = ui.tab('📊 Condition Monitoring').classes('min-w-max bg-blue-500 text-white text-center border-r border-black')
    tab2 = ui.tab('📈 Anomaly Dection').classes('min-w-max bg-blue-500 text-white text-center py-1  border-r border-black')
    tab3 = ui.tab('⚙️ Simulator').classes('min-w-max bg-blue-500 text-white text-center py-1')

with ui.tab_panels(tabs, value=tab1):
    with ui.tab_panel(tab1):
        tab1_content()
    with ui.tab_panel(tab2):
        tab2_content()
    with ui.tab_panel(tab3):
        tab3_content()

ui.run()