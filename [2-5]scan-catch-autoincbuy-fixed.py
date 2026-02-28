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
from typing import Optional ,Tuple 
from collections import Counter 
from concurrent .futures import ThreadPoolExecutor 
import datetime
if not os.path.exists("failed"):
    os.makedirs("failed")

faildown=True
DISCORD_BOT_TOKEN =input ("TOKEN: ")or "預設TOKEN"
try :
    TARGET_CHANNEL_ID =int (input ("頻道ID: "))
except :
    TARGET_CHANNEL_ID =int("預設ID")
TARGET_USER_ID =716390085896962058 
INDEX_FILE =r"db_features.pkl"
POKE_LIST_FILE =r"pokelist.csv"

MAX_WORKERS =4 
executor =ThreadPoolExecutor (max_workers =MAX_WORKERS )

GLOBAL_INDEX_DATA =None 
POKEMON_NAME_MAP ={}

logging .basicConfig (level =logging .INFO )
logger =logging .getLogger ("discord_poke_monitor")

intents =discord .Intents .default ()
intents .message_content =True 
client =discord .Client (intents =intents )

def extract_number (text ):
    if not text :
        return None 
    text =str (text )
    match =re .search (r'(\d+)',text )
    if match :
        return int (match .group (1 ))
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
        print (f"錯誤：無法讀取 {file_path }，請檢查編碼。")
        return {}

    for line in lines :
        parts =line .strip ().split (',')
        if len (parts )<3 :
            continue 

        raw_id =parts [0 ]
        english_name =parts [2 ]

        clean_id =extract_number (raw_id )

        if clean_id is not None and english_name :
            if clean_id not in mapping :
                mapping [clean_id ]=english_name .strip ()

    print (f"已載入 {len (mapping )} 筆寶可夢名稱資料。")
    return mapping 

def identify_image_worker (data :bytes )->Optional [Tuple [str ,int ]]:
    if GLOBAL_INDEX_DATA is None :
        return None 
    try :
        nparr =np .frombuffer (data ,np .uint8 )
        target_img =cv2 .imdecode (nparr ,cv2 .IMREAD_GRAYSCALE )
        if target_img is None :
            return None 

        h ,w =target_img .shape 
        if max (h ,w )>640 :
            scale =640 /max (h ,w )
            target_img =cv2 .resize (target_img ,None ,fx =scale ,fy =scale )

        orb =cv2 .ORB_create (nfeatures =500 )
        kp_query ,des_query =orb .detectAndCompute (target_img ,None )
        if des_query is None :
            return None 

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

        if not votes :
            return None 

        best_img_id ,count =Counter (votes ).most_common (1 )[0 ]

        if count >=8 :
            return (GLOBAL_INDEX_DATA ["filenames"][best_img_id ],count )
        return None 
    except Exception as e :
        logger .error (f"辨識錯誤: {e }")
        return None 

@client .event 
async def on_ready ():
    global GLOBAL_INDEX_DATA 
    global POKEMON_NAME_MAP 

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
    print (f"等待目標使用者 ID: {TARGET_USER_ID } 發送圖片...\n")

@client .event 
async def on_message (message ):
    done=False
    if message .embeds :
        print ("偵測到嵌入訊息，正在檢查內容...")
        for embed in message .embeds :
            check_text =""

            if embed .footer and embed .footer .text :
                check_text +=embed .footer .text
            if embed .description :
                check_text +=embed .description

            for field in embed .fields :
                check_text +=f" {field .value }"

            if "Spawns Remaining: 0"in check_text :
                print ("偵測到 'Spawns Remaining: 0'")
                done =True
                if done :
                    print ("正在輸入購買命令...")
                    keyboard .write ('<@716390085896962058> inc buy 30minutes 30seconds -y')
                    keyboard .press_and_release ('enter')
                    await asyncio .sleep (0.2 )
                    #return
    if message .content.lower()=="ir":

        keyboard .write ('<@716390085896962058> inc resume')
        keyboard .press_and_release ("enter")
        await asyncio .sleep (0 )
        return 
    if  message .content .lower()=="ip"or"whoa"in message.content.lower():

        keyboard .write ('<@716390085896962058> inc pause')
        keyboard .press_and_release ("enter")
        await asyncio .sleep (0 )
        return 
    if message .author ==client .user :
        return 
    if message .channel .id !=TARGET_CHANNEL_ID :
        return 

    if message .author .id !=TARGET_USER_ID :
        return 

    #done =False 

    if re .search (r'https?://\S+',message .content ):

        keyboard .write ('@Pokétwo#8236 inc pause')
        keyboard .press_and_release ('enter')
        await asyncio .sleep (0.2 )
        return 
    """
    if message .embeds :
        print ("偵測到嵌入訊息，正在檢查內容...")
        for embed in message .embeds :
            check_text =""

            if embed .footer and embed .footer .text :
                check_text +=embed .footer .text 
            if embed .description :
                check_text +=embed .description 

            for field in embed .fields :
                check_text +=f" {field .value }"

            if "Spawns Remaining: 0"in check_text :
                print ("偵測到 'Spawns Remaining: 0'")
                done =True 
                break 
    """
    image_url =None 

    if message .attachments :
        for att in message .attachments :
            if any (att .filename .lower ().endswith (ext )for ext in ['.png','.jpg','.jpeg','.webp']):
                image_url =att .url 
                break 

    if not image_url and message .embeds :
        for embed in message .embeds :
            if embed .image :
                image_url =embed .image .url 
                break 
            if embed .thumbnail :
                image_url =embed .thumbnail .url 
                break 

    if image_url :
        print (f"偵測到圖片，正在辨識... (URL: {image_url [:30 ]}...)")

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

                        print (f"   - 原始檔名: {matched_filename }")
                        print (f"   - 寶可夢 ID: {poke_id }")
                        print (f"   - 英文名稱: {english_name }")
                        print ("-"*30 )
                        keyboard .write (f'@Pokétwo#8236 c {english_name }')
                        keyboard .press_and_release ('enter')
                        await asyncio .sleep (0.5 )
                        """
                        if done :
                            print ("正在輸入購買命令...")
                            keyboard .write ('<@716390085896962058> inc buy 30minutes 30seconds -y')
                            keyboard .press_and_release ('enter')
                            await asyncio .sleep (0.2 )
                        """
                    else :
                        print ("辨識失敗：特徵不足或無匹配對象。")
                        if faildown:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = os.path.join("failed", f"failed_{timestamp}.png")
                            with open(save_path, "wb") as f:
                                f.write(img_bytes)
                            print(f"失敗的圖片已儲存至: {save_path}")

if __name__ =="__main__":
    if not DISCORD_BOT_TOKEN :
        print ("錯誤：未輸入 Token")
    else :
        client .run (DISCORD_BOT_TOKEN )
