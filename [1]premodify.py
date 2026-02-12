import cv2 
import os 
import pickle 
import numpy as np 

def resize_image (image ,max_size =640 ):
    h ,w =image .shape [:2 ]
    if max (h ,w )<=max_size :
        return image 
    scale =max_size /max (h ,w )
    return cv2 .resize (image ,(int (w *scale ),int (h *scale )),interpolation =cv2 .INTER_AREA )

def build_index (database_dir ,output_file ="db_features.pkl"):
    if not os .path .exists (database_dir ):
        print ("錯誤：找不到資料夾")
        return 


    orb =cv2 .ORB_create (nfeatures =500 )

    database_features ={}
    files =[f for f in os .listdir (database_dir )if f .lower ().endswith (('.png','.jpg','.jpeg'))]

    print (f"正在建立 {len (files )} 張圖片的索引，請稍候...")

    count =0 
    for filename in files :
        path =os .path .join (database_dir ,filename )


        img =cv2 .imread (path ,0 )
        if img is None :continue 


        img =resize_image (img )


        kp ,des =orb .detectAndCompute (img ,None )


        if des is not None and len (des )>=2 :
            database_features [filename ]=des 
            count +=1 

        if count %100 ==0 :
            print (f"已處理 {count } 張...")


    with open (output_file ,"wb")as f :
        pickle .dump (database_features ,f )

    print (f"--------------------------------")
    print (f"索引建立完成！已儲存至 {output_file }")
    print (f"有效圖片數: {count }")

if __name__ =="__main__":
    build_index ("data/images")
