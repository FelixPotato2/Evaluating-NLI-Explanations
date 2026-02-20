import pandas as pd

"""below you can change the df to whichever csv file you want to get insight into. 
This script will print the first 10 rows of that dataset in a somewhat readable way. You could also choose to print certain given rownumbers instead."""



#These prints below are some alternative ways to access certain things in pandas dataframes. 
#print(df)
#print(df.iloc[0])
#print("columns:")
#print(df.columns)
#print(df.columns[0])
#print(df["pairID"])

def print_example(df, ID=None, rownum=None, ignore_highlights=False):
    """
    Print all column values for a selected row in the DataFrame.

    param: df (pandas.DataFrame): DataFrame containing e-SNLI data.
    param: ID (str or None): pairID of the row to print.
    param: rownum (int or None): Row index to print.
    param: ignore_highlights (bool): If True, skip highlight-related columns.
    returns: None
    """
    print("----------------------------------------------------")
    if ID is not None: 
        row = df.loc[df["pairID"] == ID]
        if row.empty:
            print("ID not found.")
            return
        row = row.iloc[0]  
    if rownum is not None:
        row = df.iloc[rownum]
    for col in df.columns:
        if "Unnamed" in col: 
            continue
        if ignore_highlights:
            if "Highlighted" in col or "marked" in col: 
                continue
        #else: 
        print(f"{col}:")
        print(row[col])
        #print()

## Uncomment to choose the df
#df_glob = pd.read_csv("entailment_probs_or.csv")
#df = pd.read_csv("processed_esnli_EA.csv")
#df = pd.read_csv("esnli_dev.csv")


## Uncomment to generate 10 examples
# print(f"amount of rows:{df_glob.shape[0]}")        
# for i in range(0,10):
#     print(f"----------------EXAMPLE {i} ----------------\n")
#     print_example(df_glob, rownum=i)

#print_example(None, 4)
