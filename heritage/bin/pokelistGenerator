#!/usr/bin/env python3
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

def build_combined_index (database_dir ,output_file ="db_features.pkl"):
    if not os .path .exists (database_dir ):
        print ("錯誤：找不到資料夾")
        return 

    orb =cv2 .ORB_create (nfeatures =500 )
    sift =cv2 .SIFT_create ()


    database_features ={
    "orb":{},
    "sift":{}
    }

    files =[f for f in os .listdir (database_dir )if f .lower ().endswith (('.png','.jpg','.jpeg'))]
    print (f"正在建立 {len (files )} 張圖片的雙重索引 (ORB + SIFT)...")

    count =0 
    for filename in files :
        path =os .path .join (database_dir ,filename )
        img =cv2 .imread (path ,0 )
        if img is None :continue 

        img =resize_image (img )


        kp_orb ,des_orb =orb .detectAndCompute (img ,None )
        if des_orb is not None :
            database_features ["orb"][filename ]=des_orb 


        kp_sift ,des_sift =sift .detectAndCompute (img ,None )
        if des_sift is not None :
            database_features ["sift"][filename ]=des_sift 

        count +=1 
        if count %100 ==0 :
            print (f"已處理 {count } 張...")

    with open (output_file ,"wb")as f :
        pickle .dump (database_features ,f )

    print (f"--------------------------------")
    print (f"雙重索引建立完成！有效圖片數: {count }")

if __name__ =="__main__":
    build_index_path ="."
    build_combined_index (build_index_path )
