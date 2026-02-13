import pandas as pd 
import re 

def extract_number (text ):
    """
    從字串中提取第一個連續的數字序列並轉換為整數。
    """
    if pd .isna (text ):
        return None 
    text =str (text )
    match =re .search (r'(\d+)',text )
    if match :
        return int (match .group (1 ))
    return None 

def read_csv_safe (file_path ):
    """
    嘗試使用不同的編碼讀取 CSV，解決 Windows 中文亂碼問題
    """
    encodings =['cp950','utf-8','utf-8-sig','gbk']

    for enc in encodings :
        try :

            return pd .read_csv (file_path ,dtype =str ,encoding =enc )
        except UnicodeDecodeError :
            continue 
        except Exception as e :
            print (f"讀取 {file_path } 時發生其他錯誤: {e }")
            return None 

    print (f"錯誤：無法識別檔案 {file_path } 的編碼格式。")
    return None 

def main ():

    file_a_path ='pokelist.csv'
    file_b_path ='match_results.csv'
    output_path ='output_result.csv'


    target_col_index =4 



    print ("正在讀取檔案...")
    df_a =read_csv_safe (file_a_path )
    df_b =read_csv_safe (file_b_path )

    if df_a is None or df_b is None :
        return 

    try :



        if '編號'in df_a .columns :
            df_a ['clean_id']=df_a ['編號'].apply (extract_number )
        else :

            df_a ['clean_id']=df_a .iloc [:,0 ].apply (extract_number )



        actual_col_idx =target_col_index -1 


        if len (df_b .columns )<=actual_col_idx :
            print (f"錯誤：表 B 只有 {len (df_b .columns )} 欄，無法讀取第 {target_col_index } 欄。")
            return 

        print (f"正在處理表 B 的第 {target_col_index } 欄 (標題名稱為: '{df_b .columns [actual_col_idx ]}')...")


        df_b ['lookup_id']=df_b .iloc [:,actual_col_idx ].apply (extract_number )



        cols_to_merge =['clean_id','中文','英文']

        existing_cols =[c for c in cols_to_merge if c in df_a .columns or c =='clean_id']

        result =pd .merge (
        df_b ,
        df_a [existing_cols ],
        left_on ='lookup_id',
        right_on ='clean_id',
        how ='left'
        )


        if 'clean_id'in result .columns :
            result .drop (columns =['clean_id'],inplace =True )
        if 'lookup_id'in result .columns :
            result .drop (columns =['lookup_id'],inplace =True )


        result .to_csv (output_path ,index =False ,encoding ='utf-8-sig')
        print (f"成功！結果已儲存為：{output_path }")


        print (result .head (3 ))

    except Exception as e :
        import traceback 
        traceback .print_exc ()
        print (f"處理過程中發生錯誤: {e }")

if __name__ =="__main__":
    main ()
