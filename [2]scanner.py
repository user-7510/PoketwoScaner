import os 
import logging 
import csv 
import urllib .parse 
import asyncio 
import pickle 
import numpy as np 
import cv2 
import aiohttp 
import discord 
from typing import Optional ,List ,Dict ,Tuple 
from collections import Counter 
from concurrent .futures import ThreadPoolExecutor 


DISCORD_BOT_TOKEN =input("TOKEN：") #自行填入
TARGET_GUILD_ID =int (input ("伺服器ID："))
TARGET_USER_ID =716390085896962058 
OUTPUT_CSV ="match_results.csv"
INDEX_FILE ="db_features.pkl"


MAX_CONCURRENT_CHANNELS =15 
MAX_WORKERS =4 


GLOBAL_INDEX_DATA =None 
executor =ThreadPoolExecutor (max_workers =MAX_WORKERS )
semaphore =asyncio .Semaphore (MAX_CONCURRENT_CHANNELS )

logging .basicConfig (level =logging .INFO )
logger =logging .getLogger ("discord_img_matcher")

intents =discord .Intents .default ()
intents .message_content =True 
intents .guilds =True 
client =discord .Client (intents =intents )



def identify_image_worker (data :bytes )->Optional [Tuple [str ,int ]]:
    """這部分會在 ThreadPoolExecutor 中執行，不阻塞主程式"""
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
        logger .error (f"辨識過程發生錯誤: {e }")
        return None 



async def process_single_channel (channel ,session ,csv_writer ,user_id ):
    """單一頻道的掃描任務"""
    async with semaphore :
        try :
            perms =channel .permissions_for (channel .guild .me )
            if not perms .read_message_history :return 


            target_msg =None 
            async for msg in channel .history (limit =100 ):
                if msg .author .id ==user_id :
                    target_msg =msg 
                    break 

            if not target_msg :return 


            urls =[]
            for att in target_msg .attachments :
                if any (att .filename .lower ().endswith (ext )for ext in [".png",".jpg",".jpeg",".webp"]):
                    urls .append ({"url":att .url ,"name":att .filename })

            for emb in target_msg .embeds :
                if emb .image :urls .append ({"url":emb .image .url ,"name":"embed_img.png"})

            if not urls :return 


            for item in urls :
                async with session .get (item ["url"])as resp :
                    if resp .status !=200 :continue 
                    img_bytes =await resp .read ()


                loop =asyncio .get_event_loop ()
                result =await loop .run_in_executor (executor ,identify_image_worker ,img_bytes )

                if result :
                    matched_file ,count =result 
                    print (f"✅ [成功] 頻道: {channel .name .ljust (15 )} | 匹配: {matched_file } ({count } 票)")
                    csv_writer .writerow ([channel .name ,target_msg .id ,target_msg .author ,matched_file ,count ,item ['url']])
                else :
                    print (f"❌ [失敗] 頻道: {channel .name .ljust (15 )} | 無匹配結果")

        except Exception as e :

            pass 



@client .event 
async def on_ready ():
    global GLOBAL_INDEX_DATA 
    print (f"機器人已連線: {client .user }")


    if not os .path .exists (INDEX_FILE ):
        print (f"錯誤: 找不到 {INDEX_FILE }");await client .close ();return 

    print (f"正在優化索引資料...")
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


    guild =client .get_guild (TARGET_GUILD_ID )
    if not guild :
        print ("找不到指定伺服器");await client .close ();return 


    f =open (OUTPUT_CSV ,'a',newline ='',encoding ='utf-8-sig')
    writer =csv .writer (f )
    if os .stat (OUTPUT_CSV ).st_size ==0 :
        writer .writerow (['頻道','訊息ID','發送者','匹配檔案','票數','URL'])


    print (f"開始並行掃描 {guild .name }，請稍候...\n")
    async with aiohttp .ClientSession ()as session :
        channels =[c for c in guild .channels if isinstance (c ,discord .TextChannel )]


        tasks =[process_single_channel (ch ,session ,writer ,TARGET_USER_ID )for ch in channels ]


        await asyncio .gather (*tasks )

    f .close ()
    print ("\n[完成] 所有頻道已判讀完畢，結果存於 CSV。")
    await client .close ()

if __name__ =="__main__":
    client .run (DISCORD_BOT_TOKEN )
