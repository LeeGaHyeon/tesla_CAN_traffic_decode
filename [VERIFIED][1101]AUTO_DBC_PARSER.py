

#################################################
# Written date: 2023-02-05
# Update date: 2023-11-02
# Description: 지정된 DBC 파일을 읽어, CAN 메시지와 매칭 후, 실제 값을 구히는 코드
# Status: IF you wanna Calculate Big-endian, Add to code.
#################################################

import re #Using Regular Expression
import pandas as pd # For data processing
import copy
from tqdm import tqdm # Print progress of parsing
import os
from multiprocessing import Pool
import numpy as np
import multiprocessing
import time
import glob


from numba import jit

# GLOBAL Var

base_locate = os.getcwd()
base_CAN_DATA_dir = base_locate+"\CAN"
list_woong = ['choimingi', 'choimingi_auto', 'houjonguk' , 'houjonguk_auto', 'jeongyubin', 'jeongyubin_auto', 'leegahyeon', 'leegahyeon_auto']
# list_woong = ['leegahyeon']
DBC_FILE_PATH = './Model3CAN.dbc'


file_lt = glob.glob('CAN/*/*.csv')
FILE_PATH_LIST = file_lt
RESULT_FILE_PATH_LIST = file_lt

#DBC_ID_DATA = []
#DBC_DLC_DATA = []
#DBC_DATA = []

### SEARCH FOR NEED TO DECODING FILE
for folder_name in list_woong :
        folder_path = os.path.join(base_CAN_DATA_dir, folder_name)
        # A, B, C 폴더 순회
        # for sub_folder_name in ['A', 'B', 'C']:
        #     sub_folder_path = os.path.join(folder_path, sub_folder_name)

        # 폴더 내의 모든 CSV 파일을 읽어오기
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):

                file_path = os.path.join(folder_path, file_name)
                result_return_path =  os.path.join(folder_path, "[Decode]"+file_name)
                FILE_PATH_LIST.append(file_path)
                RESULT_FILE_PATH_LIST.append(result_return_path)



def load_dbc_file(dbc_file_path):
    DBC_ID_DATA = []
    DBC_DLC_DATA = []
    DBC_CM_DATA = []
    DBC_CM = []
    with open(dbc_file_path, 'r') as dbc_file:
         dbc_data = dbc_file.read().split("\n\n")
         DBC_CM = [x for x in dbc_data if x[0:4] == "CM_ "]
         DBC_CM = [row for text in DBC_CM for row in text.split("\n")]
         DBC_CM = [text for text in DBC_CM if text[0:7] == "CM_ SG_"]

         for data in DBC_CM:
             # lst = data.split(" ")
             comment = data[data.find("\"") + 1:-2]
             data = data[:data.find("\"") - 1]
             data = data.split(" ")
             data = data[2:]
             # print(data)
             comment = comment.replace(" ", "_")
             data.append(comment)
             DBC_CM_DATA.append(data)

         dbc_data = dbc_data[4:-2]
         dbc_data[0] = (dbc_data[0])[1:]
         dbc_data = [x for x in dbc_data if x[0:4] == "BO_ "]

    for text in dbc_data:
        dbc_message_data_list = text.split(' ')
        DBC_ID_DATA.append(dbc_message_data_list[1])
        DBC_DLC_DATA.append(dbc_message_data_list[3])

    return dbc_data,DBC_ID_DATA,DBC_DLC_DATA,DBC_CM_DATA


# 전역 변수 설정
DBC_DATA,DBC_ID_DATA,DBC_DLC_DATA,DBC_CM_DATA = load_dbc_file(DBC_FILE_PATH)


# STEP3. 받아온 시그널을 일정한 FORM 에 맞게 해석
# parameter :: required CAN ID for Signal Full text
# return :: 중요 정보로 구성된 리스트

def signal_normalization(text):
    # list to major Information
    # [SIGNAL NAME , [bit_start, bit-length, endian], [scale, offset], [min, max], Unit]
    # EXAMPLE : ['EngineSpeed', ['24', '16', '1'], ['0.125', '0'], ['0', '8031.875'], '"rpm"', 4968]
    singal_data_text = text.strip().split()
    #print(singal_data_text)
    mi_signal_data = singal_data_text[1:-2]
    #print(mi_signal_data)
    unit_name = singal_data_text[-2]

    signal_name =""
    for i in range(0,len(mi_signal_data)):
        if mi_signal_data[i] != ':':
            signal_name += mi_signal_data[i]
        else :
            mi_signal_data = mi_signal_data[i+1:]
            break;


    mi_signal_data = [re.findall(r'-?\d*\.?\d+',x) for x in mi_signal_data]
    mi_signal_data = [x for x in mi_signal_data if x ]

    mi_signal_data.insert(0,signal_name)
    mi_signal_data.append(unit_name)

    return mi_signal_data

# STEP 2-1 . CAN DATA를 Bit list로 변환 하는 함수
# parameter : CAN DATA eg,. FF FF FF 68 13 FF FF FF
# RETURN :  ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '0', '1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',                '1', '1', '1', '1', '1', '1', '1', '1']

def intel_convert_bit_CAN(can_data):
    can_data = [format((int(x,16)),'b').zfill(8) for x in can_data.split(" ")]
    can_data = [x[::-1] for x in can_data]
    can_data = str(can_data).replace("0b","").replace("['","").replace("\'","")
    bit_can_data = re.findall(r'\d', str(can_data))

    return bit_can_data

# pysical value 를 계산함
# return : return_lst[canID,SIGNAL NAME , physical_value , unit

def calculate_pysical(mi_data , raw_can_data , can_id ,can_timestamp , can_dlc , comment):

    essential_val = ""
    rval_lst = []
    hex_rval = []
    return_lst = []
    stop_flag = 0
    # bit_start 부터 length 까지 실제 사용되는 데이터 부분을 저장함 : essential_val
    # eg,. ['0', '1', '1', '0', '1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1', '1']

    if int(mi_data[1][0]) + int(mi_data[1][1]) >= 65 :
        return None

    #raw_can_data.reverse() # reverse 가 필요 v0.01

    if mi_data[1][2] == '0' :
        return None

    temp_essential_val = [ raw_can_data[x] for x in range(int(mi_data[1][0])  , int(mi_data[1][0]) + int(mi_data[1][1]) )]


    # ['0', '1', '1', '0', '1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1', '1'] 의 형태를 01101000,0010011 형태로 변환

    start_bit = int(mi_data[1][0])

    for x in range(len(temp_essential_val) ):

        if start_bit+x !=0 and (start_bit+x) % 8 == 0:
            essential_val = essential_val+","+temp_essential_val[x]
        else:
            essential_val = essential_val+temp_essential_val[x]

    essential_val = "".join(reversed(essential_val))


    # 01101000,0010011  로 변환 ['01101000', '0010011']
# version v0.1
    if ',' in essential_val:
        rval_lst = essential_val.split(",")
    else :
        rval_lst = essential_val.split()

    # ['01101000', '0010011'] 을 ['0x68', '0x13'] 로 변환 (hex)
    for z in range(0,len(rval_lst)):
        if len(rval_lst[z]) == 0:
            del rval_lst[z]


    if len(rval_lst) >= 2: # If Data Bytes more then 2 Bytes
        hex_rval = [ hex(int("".join(x),2)) for x in rval_lst ]
    else :
        hex_rval.append(hex(int("".join(rval_lst),2)))

    # 주요 데이터 정보 파일에 append
    # EXAMPLE : ['EngineSpeed', ['24', '16', '1'], ['0.125', '0'], ['0', '8031.875'], '"rpm"', ['0x68', '0x13']]
    mi_data.append(hex_rval)

    # if SIGNAL DATA IS little-endian , reverse() 사용해서 데이터를 뒤집어줌
    if mi_data[1][2] == '1' :
        if len(mi_data[5]) >= 2:
            mi_data[5].reverse()
        else :
            pass
    else:
        pass

    # ['EngineSpeed', ['24', '16', '1'], ['0.125', '0'], ['0', '8031.875'], '"rpm"', 4968]
    hex_data = mi_data[5]
    hex_data ="0x"+("".join(hex_data).replace("0x",""))
    mi_data[5] = int(hex_data,16)

    #calc _ pysical_value , and , Min-Max range in value check
    pys_val = float(mi_data[2][1]) + float(mi_data[2][0]) * float(mi_data[5])

    if float(mi_data[3][1]) == 0 and float(mi_data[3][0]) == 0:
        pass
    elif pys_val >= float(mi_data[3][0]) and float(mi_data[3][1]) >= pys_val:
        pass
    else :
        #print("ERROR : Do Not calculate Physical value ")
        stop_flag = 1;


    # CAN ID , SIGNAL_NAME , dlc , timestamp , physical Value , Unit
    if (stop_flag == 0):
        return_lst.append(can_id)
        return_lst.append(mi_data[0]) # SIGNAL_NAME
        return_lst.append(can_dlc)
        return_lst.append(can_timestamp)
        return_lst.append(pys_val) # physical Value
        return_lst.append(mi_data[4]) # Unit
        return_lst.append(comment)

        #print(return_lst)
        return return_lst
    else :
        return None



def run(can_data, dbc_data , DBC_ID_DATA ,DBC_DLC_DATA , DBC_CM_DATA):
    stop_flag = 0
    return_full_lst = []
    ref_signal_comment_lst = []


    # STEP1. Find DBC message that match with CAN message ID and length and Split()
    # STEP1. CAN ID와 DLC가 맞는 DBC 메시지를 찾아서, DBC 파일에서 분리함.

    ref_dbc_signal_data_list = None  # Message information to earn in STEP1. / STEP1.에서 얻을 메시지 정보를 저장

    for dbc_idx in range(0,len(DBC_ID_DATA)):

        if int(DBC_ID_DATA[dbc_idx]) == int(can_data['id'], 16) and int(DBC_DLC_DATA[dbc_idx]) == int(can_data['dlc']):
        # if int(DBC_ID_DATA[dbc_idx]) == int(can_data['canID'], 16) and int(DBC_DLC_DATA[dbc_idx]) == int(can_data['DLC']):

            ref_dbc_signal_data_list = dbc_data[dbc_idx].split("\n")  # If correct ID and Length, Store that DBC message. / CAN ID와 DLC가 일치할 경우, DBC 메시지를 저장


    ref_signal_comment_lst = [cml_row for cml_row in DBC_CM_DATA if int(cml_row[0]) == int(can_data['id'], 16)]
    # ref_signal_comment_lst = [cml_row for cml_row in DBC_CM_DATA if int(cml_row[0]) == int(can_data['canID'], 16)]
    #print(ref_signal_comment_lst)
    # STEP1-1. If there is no DBC messages, exit this function.
    # STEP1-1. 만약애 일치하는 DBC 메시지가 없을 경우, 이 함수를 종료함.
    if ref_dbc_signal_data_list is None:
        return None

    # STEP2. DBC SIGNAL FILE
    # STEP2. CAN ID로 부터 추출할 수 있는 DBC SIGNAL 추출,
    # 추출할 수 있는 SIGNAL 만큼 반복

    #STEP2. CAN DATA 를 Bit로 변환함
    bit_can_data = intel_convert_bit_CAN(can_data['data'])



    for j in range(1,len(ref_dbc_signal_data_list)):
        ref_comment = ""
        return_lst = []
        mi_signal_data = signal_normalization(ref_dbc_signal_data_list[j])
        for signal_comment in ref_signal_comment_lst:
            if signal_comment[1] == mi_signal_data[0]:
                ref_comment = signal_comment[2]

# 여기다가 mi_signal_data[0][0] 와 비교
        return_lst = calculate_pysical(mi_signal_data , copy.deepcopy(bit_can_data) , can_data['id'], can_data['Timestamp'] , can_data['dlc'], ref_comment)
        # return_lst = calculate_pysical(mi_signal_data, copy.deepcopy(bit_can_data), can_data['canID'], can_data['Timestamp'], can_data['DLC'], ref_comment)

        if return_lst is None:
            pass
        else :
            return_full_lst.append(return_lst)


    return return_full_lst



def multipool(can_data_split):
    return_main_full_lst = []
    for i in tqdm(range(0, len(can_data_split) ), leave=True):
        # Find a bit, Calculate physical values.
        # 비트를 찾아, 실제 값을 계산함.

        return_full_lst = run(can_data_split.iloc[i], DBC_DATA , DBC_ID_DATA ,DBC_DLC_DATA , DBC_CM_DATA)
        #print(return_full_lst)
        if return_full_lst is None:
            continue

        else :
            pass

            return_main_full_lst.append(return_full_lst)
            #df = pd.DataFrame(data = return_full_lst ,columns=["canID" , "Signal", "DLC", "Timestamp", "Physical_value", "Unit"])
            #result_df = pd.concat([result_df,df], ignore_index=True)

    return return_main_full_lst



def main():
    print("Load DBC DATA .. SUCCESS ")

    print("You have {} CPU".format(multiprocessing.cpu_count()))
    num_cores = int(input("How many cpu would you like to use? (for multi-processing) : " ))

    if len(FILE_PATH_LIST) == len(RESULT_FILE_PATH_LIST):
        for file_idx in range(0, len(FILE_PATH_LIST)):
            # Store parsed information. / 파싱된 정보를 저장함.
            # 반복문 시간
            result_df = pd.DataFrame(columns=["canID", "Signal", "DLC", "Timestamp", "Physical_value", "Unit" , "Comment"])
            can_data = pd.read_csv(FILE_PATH_LIST[file_idx])


            print("Load CAN DATA FILE .. SUCCESS ", FILE_PATH_LIST[file_idx])
            p = Pool(num_cores)

            can_data_split = np.array_split(can_data, num_cores)

            #result_df = pd.concat(p.map(multipool, can_data_split), ignore_index=True)
            full_lst = p.map(multipool, can_data_split)
            p.close()
            p.join()
            full_lst = [z for x in full_lst for y in x for z in y]

            print("Creating decode FILE .. ")
            result_df = pd.DataFrame(data=full_lst,columns=["canID", "Signal", "DLC", "Timestamp", "Physical_value", "Unit", "Comment"])

            print(RESULT_FILE_PATH_LIST[file_idx])
            result_df.to_csv(RESULT_FILE_PATH_LIST[file_idx] , index=False)
            print("Creating decode SUCCESS .. ")
            print("Save DBCtoCAN FILE ... SUCCESS " , RESULT_FILE_PATH_LIST[file_idx])

if __name__ == "__main__":
    main()


