import re

text = "This is some text, illustrate the idea."
print(re.split(r"\W+(?:some|illustrate)", text))
print("version 2")

print(re.split(r"(?:some|illustrate)", text))



print( "3")
text = "cheese is a kind of dairy"
print(re.split(r"(?:type of|form of| kind of)", text))