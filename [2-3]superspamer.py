import keyboard 
import os 
import logging 
import csv 
import asyncio 
import pickle 
import re 
import numpy as np 
import cv2 
import aiohttp 
import discord 
import subprocess 
import random 
import json 
import sys 
from typing import Optional ,Tuple 
from collections import Counter 
from concurrent .futures import ThreadPoolExecutor 
#import mouse
import pyautogui

DISCORD_BOT_TOKEN =input ("請輸入您的 Discord Bot Token (若已設置可跳過): ")or "預設"
try :
    TARGET_CHANNEL_ID =int (input ("請輸入要監聽的頻道 ID: "))
except :
    TARGET_CHANNEL_ID =0 

TARGET_USER_ID =716390085896962058 
INDEX_FILE ="db_features.pkl"
POKE_LIST_FILE ="pokelist.csv"


SPAM_WORDS =["spamer.py","testing"]
SPAM_INTERVAL =2.0 
MAX_WORKERS =4 


executor =ThreadPoolExecutor (max_workers =MAX_WORKERS )
GLOBAL_INDEX_DATA =None 
POKEMON_NAME_MAP ={}
spam_control_event =asyncio .Event ()

logging .basicConfig (level =logging .INFO )
logger =logging .getLogger ("discord_poke_monitor")

intents =discord .Intents .default ()
intents .message_content =True 
client =discord .Client (intents =intents )



def extract_number (text ):
    if not text :return None 
    text =str (text )
    match =re .search (r'(\d+)',text )
    if match :return int (match .group (1 ))
    return None 

def load_pokemon_mapping (file_path ):
    mapping ={}
    print (f"正在讀取並解析 {file_path }...")
    encodings =['utf-8-sig','utf-8','cp950','gbk']
    lines =[]
    file_content =False 
    for enc in encodings :
        try :
            with open (file_path ,'r',encoding =enc )as f :
                lines =f .readlines ()
            file_content =True 
            break 
        except UnicodeDecodeError :
            continue 
    if not file_content :
        print (f"錯誤：無法讀取 {file_path }")
        return {}
    for line in lines :
        parts =line .strip ().split (',')
        if len (parts )<3 :continue 
        clean_id =extract_number (parts [0 ])
        english_name =parts [2 ]
        if clean_id is not None and english_name :
            if clean_id not in mapping :
                mapping [clean_id ]=english_name .strip ()
    return mapping 

def identify_image_worker (data :bytes )->Optional [Tuple [str ,int ]]:
    """OpenCV 特徵比對 (CPU 密集型，需在 Executor 中執行)"""
    if GLOBAL_INDEX_DATA is None :return None 
    try :
        nparr =np .frombuffer (data ,np .uint8 )
        target_img =cv2 .imdecode (nparr ,cv2 .IMREAD_GRAYSCALE )
        if target_img is None :return None 

        h ,w =target_img .shape 
        if max (h ,w )>640 :
            scale =640 /max (h ,w )
            target_img =cv2 .resize (target_img ,None ,fx =scale ,fy =scale )

        orb =cv2 .ORB_create (nfeatures =500 )
        kp_query ,des_query =orb .detectAndCompute (target_img ,None )
        if des_query is None :return None 

        index_params =dict (algorithm =6 ,table_number =6 ,key_size =12 ,multi_probe_level =1 )
        flann =cv2 .FlannBasedMatcher (index_params ,dict (checks =50 ))
        flann .add ([GLOBAL_INDEX_DATA ["descriptors"]])
        flann .train ()

        matches =flann .knnMatch (des_query ,k =2 )
        votes =[]
        for m_tuple in matches :
            if len (m_tuple )==2 :
                m ,n =m_tuple 
                if m .distance <0.75 *n .distance :
                    idx =m .trainIdx 
                    if idx <len (GLOBAL_INDEX_DATA ["indices"]):
                        votes .append (GLOBAL_INDEX_DATA ["indices"][idx ])
        if not votes :return None 
        best_img_id ,count =Counter (votes ).most_common (1 )[0 ]
        if count >=8 :
            return (GLOBAL_INDEX_DATA ["filenames"][best_img_id ],count )
        return None 
    except Exception as e :
        logger .error (f"辨識錯誤: {e }")
        return None 



def create_wdchange_script ():
    """建立維持視窗焦點的腳本"""
    data ="""
import pygetwindow
import time
import keyboard
keyword = "Discord"
print(f"Waiting for {keyword}...")
while True:
    wins = pygetwindow.getWindowsWithTitle(keyword)
    if wins:
        target = wins[0]
        break
    time.sleep(1)
print(f"Locked on {target.title}")
try:
    while True:
        aw = pygetwindow.getActiveWindow()
        if aw and keyword not in aw.title:
            try:
                if not target.isMinimized: target.activate()
                else: target.restore(); target.activate()
            except: pass
        if keyboard.is_pressed("esc"): break
        time.sleep(0.1)
except KeyboardInterrupt: pass
"""
    with open ("wdchange.py","w",encoding ="utf-8")as f :
        f .write (data )

def load_spam_settings ():
    """讀取 Spammer 設定"""
    global SPAM_WORDS ,SPAM_INTERVAL 
    try :
        with open ('words.txt','r',encoding ='utf-8')as f :
            content =f .read ()
            if ','in content :
                SPAM_WORDS =[s .strip ()for s in content .split (',')if s .strip ()]
            else :
                SPAM_WORDS =content .split ()
    except :
        print ("找不到 words.txt，使用預設字詞。")

    try :
        SPAM_INTERVAL =float (input ("請輸入發話間隔 (秒, 預設 2.0): ")or 2.0 )
    except :
        SPAM_INTERVAL =2.0 



async def spam_task ():
    """
    這是 Spammer 的主要迴圈，它與 Bot 並行運作。
    """
    await client .wait_until_ready ()
    print (">>> Spammer 模組已啟動，準備開始...")


    create_wdchange_script ()
    try :
        subprocess .Popen (["python","wdchange.py"])
        print (">>> 視窗鎖定腳本已執行 (wdchange.py)")
    except Exception as e :
        print (f"無法啟動 wdchange.py: {e }")


    print (">>> Spammer 將在 5 秒後開始運作...")
    await asyncio .sleep (5 )
    spam_control_event .set ()

    while not client .is_closed ():
        try :


            if not spam_control_event .is_set ():
                print (">>> [Spammer] 暫停中，等待圖片辨識完成...")
                await spam_control_event .wait ()
                print (">>> [Spammer] 恢復運作！")


            word =random .choice (SPAM_WORDS )
            keyboard .write (word )
            keyboard .press_and_release ("enter")


            sleep_time =SPAM_INTERVAL +(random .randint (-100 ,100 )/1000 )
            print (f"[Spammer] 已發送: {word }, 等待 {sleep_time :.2f}s")


            await asyncio .sleep (max (0.1 ,sleep_time ))


            if keyboard .is_pressed ("esc"):
                print (">>> 偵測到 ESC，停止程式。")
                await client .close ()
                break 

        except Exception as e :
            print (f"Spammer 發生錯誤: {e }")
            await asyncio .sleep (1 )



@client .event 
async def on_ready ():
    global GLOBAL_INDEX_DATA ,POKEMON_NAME_MAP 
    print (f"\n登入身分: {client .user }")


    if not os .path .exists (INDEX_FILE ):
        print (f"錯誤: 找不到 {INDEX_FILE }");await client .close ();return 

    print ("正在載入影像特徵庫...")
    with open (INDEX_FILE ,"rb")as f :
        raw =pickle .load (f )

    all_des ,all_idx ,all_fnames =[],[],[]
    for i ,(fn ,des )in enumerate (raw .items ()):
        if des is not None :
            all_fnames .append (fn );all_des .extend (des );all_idx .extend ([i ]*len (des ))

    GLOBAL_INDEX_DATA ={
    "descriptors":np .array (all_des ,dtype =np .uint8 ),
    "indices":all_idx ,
    "filenames":all_fnames 
    }


    if not os .path .exists (POKE_LIST_FILE ):
        print (f"錯誤: 找不到 {POKE_LIST_FILE }");await client .close ();return 
    POKEMON_NAME_MAP =load_pokemon_mapping (POKE_LIST_FILE )

    print (f"\n=== 系統就緒 ===")
    print (f"正在監聽頻道 ID: {TARGET_CHANNEL_ID }")


    client .loop .create_task (spam_task ())

@client .event 
async def on_message (message ):

    if message .author ==client .user :return 
    if message .channel .id !=TARGET_CHANNEL_ID :return 
    if message .author .id !=TARGET_USER_ID :return 


    image_url =None 
    if message .attachments :
        for att in message .attachments :
            if any (att .filename .lower ().endswith (ext )for ext in ['.png','.jpg','.jpeg','.webp']):
                image_url =att .url ;break 
    if not image_url and message .embeds :
        for embed in message .embeds :
            if embed .image :image_url =embed .image .url ;break 
            if embed .thumbnail :image_url =embed .thumbnail .url ;break 

    if image_url :
        print (f"偵測到圖片! 暫停 Spammer 並開始辨識... (URL: {image_url [:30 ]}...)")


        spam_control_event .clear ()

        try :
            async with aiohttp .ClientSession ()as session :
                async with session .get (image_url )as resp :
                    if resp .status ==200 :
                        img_bytes =await resp .read ()
                        loop =asyncio .get_event_loop ()


                        result =await loop .run_in_executor (executor ,identify_image_worker ,img_bytes )

                        if result :
                            matched_filename ,score =result 
                            poke_id =extract_number (matched_filename )
                            english_name =POKEMON_NAME_MAP .get (poke_id ,"Unknown Name")

                            print (f"🔥 辨識成功！")
                            print (f"   - 原始檔名: {matched_filename }")
                            print (f"   - 英文名稱: {english_name }")
                            print ("-"*30 )
                            print (english_name )
                            pyautogui.click(x=10, y=200, button='left')
                            keyboard.write(f'@Pokétwo#8236 c {english_name}')
                            keyboard.press_and_release('enter')
                            pyautogui.click(x=1500, y=200, button='left')
                        else :
                            print ("❌ 辨識失敗：特徵不足或無匹配對象。")
        except Exception as e :
            print (f"處理圖片時發生錯誤: {e }")
        finally :


            print ("辨識流程結束，恢復 Spammer。")
            spam_control_event .set ()

if __name__ =="__main__":
    if not DISCORD_BOT_TOKEN :
        print ("錯誤：未輸入 Token")
    else :

        load_spam_settings ()
        client .run (DISCORD_BOT_TOKEN )
