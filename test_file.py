import re
import json
import ast

# text = "This is some text, illustrate the idea."
# print(re.split(r"\W+(?:some|illustrate)", text))
# print("version 2")

# print(re.split(r"(?:some|illustrate)", text))



# print( "3")
# text = "cheese is a kind of dairy"
# print(re.split(r"(?:type of|form of| kind of)", text))


# list1 = [1,2,3,4,5]
# list2 = [1,2,3]
# listtest1 = list1[-4:]
# listtest2 = list2[-4:]

# print(listtest1, listtest2)

s = "['commandos', 'gun']"
#a = s.split(',')
#b = [w.strip(" '][") for w in a]
#a = list(map(str.strip(" ']["), s.split(',')))  
#a = json.loads(s)
a = ast.literal_eval(s) 
print(a)