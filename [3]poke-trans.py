import pandas as pd 
import re 

def extract_number (text ):
    """
    從字串中提取第一個連續的數字序列並轉換為整數。
    例如: "No. 001" -> 1, "ID: 25 (Pikachu)" -> 25, "#0001" -> 1
    """
    if pd .isna (text ):
        return None 


    text =str (text )


    match =re .search (r'(\d+)',text )
    if match :
        return int (match .group (1 ))
    return None 

def main ():

    file_a_path ='pokelist.csv'
    file_b_path ='match_results.csv'
    output_path ='output_result.csv'



    target_col ='D'


    try :


        df_a =pd .read_csv (file_a_path ,dtype =str )
        df_b =pd .read_csv (file_b_path ,dtype =str )

        print ("資料讀取成功...")



        df_a ['clean_id']=df_a ['編號'].apply (extract_number )



        if target_col not in df_b .columns :
            print (f"錯誤：在表 B 中找不到欄位名稱 '{target_col }'。請檢查 CSV 表頭。")
            return 

        df_b ['lookup_id']=df_b [target_col ].apply (extract_number )



        result =pd .merge (
        df_b ,
        df_a [['clean_id','中文','英文']],
        left_on ='lookup_id',
        right_on ='clean_id',
        how ='left'
        )



        result .drop (columns =['lookup_id','clean_id'],inplace =True )


        result .to_csv (output_path ,index =False ,encoding ='utf-8-sig')
        print (f"處理完成！結果已儲存為：{output_path }")


        print ("\n--- 預覽結果 ---")
        print (result .head ())

    except FileNotFoundError as e :
        print (f"找不到檔案: {e }")
    except Exception as e :
        print (f"發生錯誤: {e }")

if __name__ =="__main__":
    main ()
