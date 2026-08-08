#!/usr/bin/env python3
import re 
import os 
import requests 

def main ():
    file_path ='failed.txt'


    if not os .path .exists (file_path ):
        print (f"找不到檔案 {file_path }，請確定它與本程式放在同一個資料夾。")
        return 


    with open (file_path ,'r',encoding ='utf-8')as f :
        content =f .read ()




    pattern =r'(https://cdn\.discordapp\.com/[^\s:]+?(?:hm=[a-f0-9]{64}|size=\d+))'
    urls =re .findall (pattern ,content )


    output_dir ='downloaded_failed_attachments'
    if not os .path .exists (output_dir ):
        os .makedirs (output_dir )

    print (f"解析完畢！共找到 {len (urls )} 個有效的下載連結，準備開始下載...\n")


    success_count =0 
    for i ,url in enumerate (urls ):
        try :

            response =requests .get (url ,timeout =10 )
            response .raise_for_status ()


            filename =url .split ('/')[-1 ].split ('?')[0 ]
            if not filename :
                filename =f"attachment_{i }.png"


            name ,ext =os .path .splitext (filename )
            save_name =f"{name }_{i +1 :03d}{ext }"
            save_path =os .path .join (output_dir ,save_name )


            with open (save_path ,'wb')as img_file :
                img_file .write (response .content )

            print (f"[{i +1 }/{len (urls )}] 成功下載 -> {save_name }")
            success_count +=1 

        except requests .exceptions .RequestException as e :
            print (f"[{i +1 }/{len (urls )}] 下載失敗: {url }\n   └─ 錯誤原因: {e }")
        except Exception as e :
            print (f"[{i +1 }/{len (urls )}] 發生未預期的錯誤: {e }")

    print (f"\n下載作業結束！成功下載了 {success_count } 個檔案。")

if __name__ =='__main__':
    main ()

