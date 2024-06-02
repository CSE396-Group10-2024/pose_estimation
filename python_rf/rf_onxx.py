from sklearn import datasets
import pandas as pd
import numpy as np
import socket
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import json
import os
import threading
import queue
# Load and preprocess data
import signal
import sys
move_repetitions={
    "left_arm":0,
    "right_arm":0,
    "zombie":0,
    "t_pose":0,
    "left_knee":0,
    "right_knee":0,
    "left_head":0,
    "right_head":0,
    "squat":0,
    "walk":0
}
global_client_pos = None
global_client_dscn = None
global_log_queue=None
def signal_handler(sig,_a):
    print("Signal received, shutting down python process...")
    global_log_queue.stop_logging()
    print(move_repetitions)
    if global_client_pos:
        global_client_pos.close()

    if global_client_dscn:
        global_client_dscn.close()

    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals
def find_directory(directory_name):
    # Get the current working directory
    current_directory = os.getcwd()
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(current_directory):
        if directory_name in dirs:
            return os.path.join(root, directory_name)
    
    return None

# Example usage
directory_name = 'onnx_csv'  # Replace with the name of the directory you are looking for
path = find_directory(directory_name)

dfs=[]
all_files = glob.glob(path + "/*.csv")
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            filepath = os.path.join(root, file)
            df = pd.read_csv(filepath)
            df = df.dropna(how='all')
            # Drop the 'index' column if it exists
            if 'index' in df.columns:
                df = df.drop(columns=['index'])
            # Drop columns that include 'angle' in their name
         

            dfs.append(df)
features = pd.concat(dfs, ignore_index=True)
print(features.columns)
class_counts = features['target_class'].value_counts()
print(class_counts)
#   {"left_arm_body",9,11,0},
#     {"right_arm_body",10,12,0},
    
#     {"right_elbow_right_head",8,4,0},
#     {"left_elbow_left_head",7,3,0}

def find_columns_to_drop(df):
    columns_to_drop = [col for col in df.columns if (
        ('z' in col and len(col) in [2, 3]) or 
        ('x' in col and len(col) in [2, 3])or
        ('y' in col and len(col) in [2, 3]) or
        ('visibility' in col) or 
        ('presence' in col) or 
        ('localized' in col) or
        ('_loc' in col) or
        ('Unnamed'in col)

    )]
    return columns_to_drop

def find_columns_to_keep(df):
    columns_to_keep = [col for col in df.columns if 'angle' in col or col == 'target_class']
    return columns_to_keep

# Get the columns that meet the criteria
columns_to_drop = find_columns_to_drop(features)

# Drop the identified columns
features = features.drop(columns=columns_to_drop)
columns_to_keep=find_columns_to_keep(features)
features = features[columns_to_keep]

print(features.columns)
duplicate_rows = features.duplicated()
num_duplicates = duplicate_rows.sum()
print(f'Number of duplicate rows: {num_duplicates}')
duplicates = features[duplicate_rows]
print('Duplicate rows:')
print(duplicates)
features = features.drop_duplicates()

duplicate_rows = features.duplicated()
num_duplicates = duplicate_rows.sum()
print(f'Number of duplicate rows: {num_duplicates}')

X = features.drop(['target_class'], axis=1)
y = features['target_class']
cols = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=0)
cols = X_train.columns

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=cols)
X_test = pd.DataFrame(X_test, columns=cols)

# Replace SVC with RandomForestClassifier
clf = RandomForestClassifier(bootstrap= False, max_depth= 40, min_samples_leaf= 2, min_samples_split= 9, n_estimators= 87)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

IP_POSE = "127.0.0.1"
PORT_POSE = 1234
ADDR_POSE = (IP_POSE, PORT_POSE)
SIZE = 1024
FORMAT = "utf-8"

IP_DSCN = "127.0.0.1"
PORT_DSCN = 1235
ADDR_DSCN = (IP_DSCN, PORT_DSCN)
SIZE = 1024


LEFT_ARM_TRESHOLD = 0.75
RIGHT_ARM_TRESHOLD = 0.75
ZOMBIE_TRESHOLD = 0.85
T_POSE_TRESHOLD = 0.75
LEFT_KNEE_TRESHOLD = 0.75
RIGHT_KNEE_TRESHOLD = 0.75
LEFT_HEAD_TRESHOLD = 0.75
RIGHT_HEAD_TRESHOLD = 0.75
SQUAT_TRESHOLD = 0.75
WALK_TRESHOLD = 0.75

moves_tresholds={
    "left_arm":LEFT_ARM_TRESHOLD,
    "right_arm":RIGHT_ARM_TRESHOLD,
    "zombie":ZOMBIE_TRESHOLD,
    "t_pose":T_POSE_TRESHOLD,
    "left_knee":LEFT_KNEE_TRESHOLD,
    "right_knee":RIGHT_KNEE_TRESHOLD,
    "left_head":LEFT_HEAD_TRESHOLD,
    "right_head":RIGHT_HEAD_TRESHOLD,
    "squat":SQUAT_TRESHOLD,
    "walk":WALK_TRESHOLD

}
LEFT_ARM_TIME = 3
RIGHT_ARM_TIME = 3
ZOMBIE_TIME = 2
T_POSE_TIME = 3
LEFT_KNEE_TIME = 3
RIGHT_KNEE_TIME = 3
LEFT_HEAD_TIME = 3
RIGHT_HEAD_TIME = 3
SQUAT_TIME = 3
WALK_TIME = 3

move_times={
    "left_arm":LEFT_ARM_TIME,
    "right_arm":RIGHT_ARM_TIME,
    "zombie":ZOMBIE_TIME,
    "t_pose":T_POSE_TIME,
    "left_knee":LEFT_KNEE_TIME,
    "right_knee":RIGHT_KNEE_TIME,
    "left_head":LEFT_HEAD_TIME,
    "right_head":RIGHT_HEAD_TIME,
    "squat":SQUAT_TIME,
    "walk":WALK_TIME
}

NEW_MOVE_TRESHOLD=3
class MyRandomForest():
    def __init__(self):
        self.mode = "predict"
        self.new_file = True
        self.server_dscn = None
        self.conn = None
        self.addr = None
        self.client_pos = None
        self.client_dscn=None
        self.pose_miss_counter = 0
        self.dscn_miss_counter = 0
        self.data_dic = None
        self.log_queue = queue.Queue()

        self.logging_thread = threading.Thread(target=self.log_to_file, daemon=True)
        self.logging_thread.start()
        self.main()

    def log_to_file(self):
        with open('log.txt', 'a') as log_file:
            while True:
                message = self.log_queue.get()
                if message == "EXIT":
                    break
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                log_file.write(f"{timestamp} - {message}\n")
                log_file.flush()

    def log_message(self, message):
        self.log_queue.put(message)

    def connect_dscn(self):
        while True:
            try:
                self.client_dscn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_dscn.connect(ADDR_DSCN)
                test_message = "Connection established from client."
                self.client_dscn.sendall(test_message.encode(FORMAT))  # Encode with FORMAT
                self.log_message("Connected by pose")
                break
            except Exception as e:
                time.sleep(2)
                self.log_message(f"Error in pose: {e}")

    def connect_pose(self):
        while True:
            try:
                self.client_pos = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_pos.connect(ADDR_POSE)
                test_message = "Connection established from client."
                self.client_pos.sendall(test_message.encode(FORMAT))  # Encode with FORMAT
                self.log_message("Connected by pose")
                break
            except Exception as e:
                time.sleep(2)
                self.log_message(f"Error in pose: {e}")
        

    def main(self):
        self.connect_pose()

        # self.connect_dscn()
        global global_client_pos, global_client_dscn
        global_client_pos = self.client_pos
        global_client_dscn = self.client_dscn
        
        current_class = None
        current_probability = None
        current_time = None
        current_class_count=0
        new_move={'name':None,'time':None}
        current_is_done=False
        miss_move_time=None
        
        while True:
            time.sleep(0.1)
            try:
                data_str = self.client_pos.recv(SIZE).decode(FORMAT)
                if len(data_str) == 0:
                    self.data_dic = None
                else:
                    try:
                        self.data_dic = json.loads(data_str)
                    except:
                        if data_str =="EXIT":
                            print(move_repetitions)
                            break
            except Exception as e:
                print(move_repetitions)
                self.log_message(move_repetitions)
                print(f"Error in get: {e}")
                continue

            if self.data_dic is None:
                continue

            if self.mode == "predict":
                # print(self.data_dic)
                new_entry_df = pd.DataFrame(self.data_dic, index=[0])
                # print(cols)
                # print(new_entry_df.columns)
               
                new_entry_df = scaler.transform(new_entry_df)
                probabilities = clf.predict_proba(new_entry_df)

                # Get the top 3 predictions and their probabilities
                top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
                top_3_predictions = clf.classes_[top_3_indices]
                top_3_probabilities = probabilities[0][top_3_indices]

                try:

                   
                    if(top_3_probabilities[0]>moves_tresholds[top_3_predictions[0]]):

                        if current_class==None :
                            current_class=top_3_predictions[0]
                            current_probability=top_3_probabilities[0]
                            current_time=time.time()
                            current_is_done=False
                            print(f"Prediction: Class = {current_class}, Probability = {current_probability:.4f} is started")
                            print(f"repeatition{move_repetitions[current_class]}")
                            self.log_message(f"Prediction: Class = {current_class}, Probability = {current_probability:.4f} is started")
                            if not current_class!="walk":
                                self.client_dscn.sendall(f"{current_class} {current_probability:.4f} started 0".encode(FORMAT))  

                        else:
                        #prediction mmove same current move and ,current move is done 
                            if(  not current_is_done and current_class == top_3_predictions[0] and time.time()-current_time>move_times[current_class] ):#
                                miss_move_time=None
                                print(f"Prediction: Class = {current_class} is done")
                                self.log_message(f"Prediction: Class = {current_class} is done")
                                current_is_done=True
                                move_repetitions[current_class]+=1
                                print(f"repeatition{move_repetitions[current_class]}")

                                print(f"Repetitions: {move_repetitions}")
                                if not current_class!="walk":
                                    self.client_dscn.sendall(f"{current_class} {current_probability:.4f} done {move_repetitions[current_class]}".encode(FORMAT))  
                                else:
                                    move_repetitions[current_class]=0

                            elif (current_class == top_3_predictions[0] and time.time()-current_time<move_times[current_class]):
                                miss_move_time=None
                                print(f"Prediction: Class = {top_3_predictions[0]} is continued")
                                self.log_message(f"Prediction: Class = {top_3_predictions[0]} is continued")
                                print(f"repeatition{move_repetitions[current_class]}")

                                self.client_dscn.sendall(f"{current_class} {current_probability:.4f} continue {move_repetitions[current_class]}".encode(FORMAT))  
                                if not current_class!="walk":
                                    self.client_dscn.sendall(f"{current_class} {current_probability:.4f} done {move_repetitions[current_class]}".encode(FORMAT))  
                                
                            elif(current_is_done and current_class == top_3_predictions[0]  and time.time()-current_time>move_times[current_class] ):
                                miss_move_time=None
                                print(f"{current_class}repetiton done, sıradaki harekete geç")
                                print(f"repeatition{move_repetitions[current_class]}")
                                #prediction move same current move and current move is not done
                                self.log_message(f"{current_class}repetiton done, sıradaki harekete geç")
                                self.client_dscn.sendall(f"{current_class} {current_probability:.4f} move_next {move_repetitions[current_class]}".encode(FORMAT))  
                                if not current_class!="walk":
                                    self.client_dscn.sendall(f"{current_class} {current_probability:.4f} done {move_repetitions[current_class]}".encode(FORMAT))  
                                


                            if ( top_3_predictions[0]!=current_class):
                                miss_move_time=None
                                if (new_move["name"]==top_3_predictions[0] and new_move["time"]>move_times[new_move["name"]]/2 ):
                                    current_class=new_move["name"]
                                    current_time=new_move["time"]
                                    current_is_done=False
                                    print(f"hareket değişti {current_class} {top_3_probabilities[0]}")
                                    self.log_message(f"hareket değişti {current_class} {top_3_probabilities[0]}")
                                    print(f"repeatition{move_repetitions[current_class]}")
                                    if not current_class!="walk":
                                        self.client_dscn.sendall(f"{current_class} {current_probability:.4f} done {move_repetitions[current_class]}".encode(FORMAT))  
                                
                                   

                         
                                    #aynsı değilse eşitle
                                elif new_move["name"]!=top_3_probabilities[0]:
                                    new_move["name"]=top_3_predictions[0]
                                    new_move["time"]=time.time()
                    else:
                        if(miss_move_time==None):
                            miss_move_time=time.time()
                        elif(miss_move_time-time.time()>NEW_MOVE_TRESHOLD/6):
                            current_class=None
                            current_probability=None
                            current_time=None
                            current_is_done=False
                            

                        print(f"Prediction {top_3_predictions[0]}  is not valid" )  
                        self.log_message(f"Prediction {top_3_predictions[0]}  is not valid")   
                        self.client_dscn.sendall(f"{top_3_predictions[0]} {top_3_probabilities[0]:.4f} not_valid 0".encode(FORMAT))  
                                

                            #prediction move is different from current move
#                            elif(current_class != top_3_predictions[0] and moves_tresholds[top_3_predictions[0]]>top_3_probabilities[0]):##new move



                                

                                #prediction move is different from 
                               

                    # Send top 3 predictions and their probabilities
                    # message = json.dumps({
                    #     "predictions": top_3_predictions.tolist(),
                    #     "probabilities": top_3_probabilities.tolist()
                    # })
                    # self.conn.sendall(message.encode(FORMAT))
                except Exception as e:
                    print()

            self.client_pos.sendall(data_str.encode(FORMAT))
                # new_entry_df = scaler.transform(new_entry_df)
                # prediction = clf.predict(new_entry_df)
                # confidence = clf.predict_proba(new_entry_df)
                # try:
                #     print(f"send {prediction}, confidence: {confidence}")
                #     prediction = prediction[0]
                #     confidence = confidence.max()  # get the maximum probability as confidence score
                #     print(f"send {prediction}, confidence: {confidence}")
                #     # self.conn.send(f"{prediction},{confidence}".encode(FORMAT))
                # except Exception as e:
                #     print(f"Error sending prediction: {e}")
                #     self.dscn_miss_counter += 1
    def stop_logging(self):
        self.log_queue.put("EXIT")
        self.logging_thread.join()

MyRandomForest1 = MyRandomForest()
global_log_queue=MyRandomForest1
