import subprocess 
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


if not os .path .exists ("failed"):
    os .makedirs ("failed")

faildown =True 
DISCORD_BOT_TOKEN =input("TOKEN: ")or"預設" #改成你的
try :
    TARGET_CHANNEL_ID =int (input("頻道ID: ")) 
except :
    TARGET_CHANNEL_ID =int ("預設")#改成你的
TARGET_USER_ID =716390085896962058 
INDEX_FILE =r"db_features.pkl"
POKE_LIST_FILE =r"pokelist.csv"

MAX_WORKERS =4 
executor =ThreadPoolExecutor (max_workers =MAX_WORKERS )
GLOBAL_DATA =None 
POKEMON_NAME_MAP ={}

logging .basicConfig (level =logging .INFO )
logger =logging .getLogger ("discord_poke_monitor")

intents =discord .Intents .default ()
intents .message_content =True 
client =discord .Client (intents =intents )

def extract_number (text ):
    if not text :return None 
    match =re .search (r'(\d+)',str (text ))
    return int (match .group (1 ))if match else None 

def load_pokemon_mapping (file_path ):
    mapping ={}
    encodings =['utf-8-sig','utf-8','cp950','gbk']
    for enc in encodings :
        try :
            with open (file_path ,'r',encoding =enc )as f :
                reader =csv .reader (f )
                for row in reader :
                    if len (row )>=3 :
                        clean_id =extract_number (row [0 ])
                        if clean_id :mapping [clean_id ]=row [2 ].strip ()
            return mapping 
        except :continue 
    return {}

def identify_process (target_img ,mode ="ORB"):
    """
    核心辨識邏輯：支援 ORB 與 SIFT
    """
    if mode =="ORB":
        detector =cv2 .ORB_create (nfeatures =500 )
        index_params =dict (algorithm =6 ,table_number =6 ,key_size =12 ,multi_probe_level =1 )
        search_params =dict (checks =50 )
        ratio =0.75 
        min_votes =8 
    else :
        detector =cv2 .SIFT_create ()
        index_params =dict (algorithm =1 ,trees =5 )
        search_params =dict (checks =50 )
        ratio =0.8 
        min_votes =5 

    kp_query ,des_query =detector .detectAndCompute (target_img ,None )
    if des_query is None :return None 

    flann =cv2 .FlannBasedMatcher (index_params ,search_params )
    train_des =GLOBAL_DATA [mode .lower ()]["descriptors"]
    flann .add ([train_des ])
    flann .train ()

    matches =flann .knnMatch (des_query ,k =2 )
    votes =[]
    for m_tuple in matches :
        if len (m_tuple )==2 :
            m ,n =m_tuple 
            if m .distance <ratio *n .distance :
                idx =m .trainIdx 
                if idx <len (GLOBAL_DATA [mode .lower ()]["indices"]):
                    votes .append (GLOBAL_DATA [mode .lower ()]["indices"][idx ])

    if not votes :return None 
    best_img_id ,count =Counter (votes ).most_common (1 )[0 ]

    if count >=min_votes :
        return (GLOBAL_DATA [mode .lower ()]["filenames"][best_img_id ],count )
    return None 

def dual_identify_worker (data :bytes )->Optional [Tuple [str ,int ,str ]]:
    try :
        nparr =np .frombuffer (data ,np .uint8 )
        target_img =cv2 .imdecode (nparr ,cv2 .IMREAD_GRAYSCALE )
        if target_img is None :return None 


        h ,w =target_img .shape 
        if max (h ,w )>640 :
            scale =640 /max (h ,w )
            target_img =cv2 .resize (target_img ,None ,fx =scale ,fy =scale )


        res =identify_process (target_img ,mode ="ORB")
        if res :
            return (res [0 ],res [1 ],"ORB")


        logger .info ("ORB 辨識失敗，啟動 SIFT 備援路徑...")
        res =identify_process (target_img ,mode ="SIFT")
        if res :
            return (res [0 ],res [1 ],"SIFT")

        return None 
    except Exception as e :
        logger .error (f"辨識錯誤: {e }")
        return None 

@client .event 
async def on_ready ():
    global GLOBAL_DATA ,POKEMON_NAME_MAP 
    print (f"\n登入身分: {client .user }")

    if not os .path .exists (INDEX_FILE ):
        print (f"錯誤: 找不到 {INDEX_FILE }");await client .close ();return 

    with open (INDEX_FILE ,"rb")as f :
        raw_data =pickle .load (f )


    GLOBAL_DATA ={"orb":{},"sift":{}}
    for mode in ["orb","sift"]:
        all_des ,all_idx ,all_fnames =[],[],[]
        dtype =np .uint8 if mode =="orb"else np .float32 
        for i ,(fn ,des )in enumerate (raw_data [mode ].items ()):
            all_fnames .append (fn )
            all_des .extend (des )
            all_idx .extend ([i ]*len (des ))
        GLOBAL_DATA [mode ]={
        "descriptors":np .array (all_des ,dtype =dtype ),
        "indices":all_idx ,
        "filenames":all_fnames 
        }

    POKEMON_NAME_MAP =load_pokemon_mapping (POKE_LIST_FILE )
    print (f"系統就緒，已載入雙重特徵庫。")

@client .event 
async def on_message (message ):
    if message .author .id ==874910942490677270 :
        return 
    done =False 
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
                    await asyncio .sleep (5)
                    print ("正在輸入購買命令...")
                    text ="<@716390085896962058> inc buy 30minutes 30seconds -y"
                    subprocess .run (['clip'],input =text .strip (),encoding ='utf-16',check =True )

                    keyboard .press_and_release ('ctrl+v')

                    keyboard .press_and_release ('enter')
                    await asyncio .sleep (0.2 )

    if message .content .lower ()=="ir":
        text ="<@716390085896962058> inc resume"
        subprocess .run (['clip'],input =text .strip (),encoding ='utf-16',check =True )

        keyboard .press_and_release ('ctrl+v')

        keyboard .press_and_release ("enter")
        await asyncio .sleep (0 )
        return 
    if message .content .lower ()=="ip":#or "whoa"in message .content .lower ()
        text ="<@716390085896962058> inc pause"
        subprocess .run (['clip'],input =text .strip (),encoding ='utf-16',check =True )

        keyboard .press_and_release ('ctrl+v')

        keyboard .press_and_release ("enter")
        await asyncio .sleep (0 )
        return 
    if message .author ==client .user :
        return 
    if message .channel .id !=TARGET_CHANNEL_ID :
        return 

    if message .author .id !=TARGET_USER_ID :
        return 



    if re .search (r'https?://\S+',message .content ):
        os .system (f'echo ^<@716390085896962058^> inc pause | clip')
        keyboard .press_and_release ('ctrl+v')

        keyboard .press_and_release ('enter')
        await asyncio .sleep (0.2 )
        return 
    """
    if message.author == client.user: return 
    if message.channel.id != TARGET_CHANNEL_ID: return 
    if message.author.id != TARGET_USER_ID: return 
    """
    image_url =None 
    if message.attachments:
        for att in message.attachments:
            if any(att.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                image_url = att.url
                break 

    if not image_url and message.embeds:
        for embed in message.embeds:
            if embed.image:
                image_url = embed.image.url
            elif embed.thumbnail:
                image_url = embed.thumbnail.url
            if image_url: break

    if image_url :
        async with aiohttp .ClientSession ()as session :
            async with session .get (image_url )as resp :
                if resp .status ==200 :
                    img_bytes =await resp .read ()
                    loop =asyncio .get_event_loop ()
                    result =await loop .run_in_executor (executor ,dual_identify_worker ,img_bytes )

                    if result :
                        matched_filename ,score ,engine =result 
                        poke_id =extract_number (matched_filename )
                        english_name =POKEMON_NAME_MAP .get (poke_id ,"Unknown Name")

                        print (f"   - 辨識引擎: {engine }")
                        print (f"   - 英文名稱: {english_name } (Score: {score })")
                        keyboard .write (f'@Pokétwo#8236 c {english_name }')
                        keyboard .press_and_release ('enter')
                    else :
                        
                        if faildown :
                            save_path =os .path .join ("failed",f"failed_{datetime .datetime .now ().strftime ('%Y%m%d_%H%M%S')}.png")
                            with open (save_path ,"wb")as f :f .write (img_bytes )

if __name__ =="__main__":
    client .run (DISCORD_BOT_TOKEN )





