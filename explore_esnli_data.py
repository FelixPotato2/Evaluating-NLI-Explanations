import pandas as pd

"""below you can change the df to whichever csv file you want to get insight into. 
This script will print the first 10 rows of that dataset in a somewhat readable way. You could also choose to print certain given rownumbers instead."""

df = pd.read_csv("entailment_probs_or.csv")
#df = pd.read_csv("processed_esnli_EA.csv")
#df = pd.read_csv("esnli_dev.csv")

#These prints below are some alternative ways to access certain things in pandas dataframes. 
#print(df)
#print(df.iloc[0])
#print("columns:")
#print(df.columns)
#print(df.columns[0])
#print(df["pairID"])

def print_example(ID = None, rownum = None):
    
    if ID is not None: 
        row = df.loc[df["pairID"] == ID]
        if row.empty:
            print("ID not found.")
            return
        row = row.iloc[0]  
    if rownum is not None:
        row = df.iloc[rownum]
    for col in df.columns:
        #if "Highlighted" in col: 
            #continue
        #else: 
        print(f"{col}:")
        print(row[col])
        print()

print(f"amount of rows:{df.shape[0]}")        
for i in range(0,10):
    print(f"----------------EXAMPLE {i} ----------------\n")
    print_example(rownum=i)

#print_example(None, 4)
