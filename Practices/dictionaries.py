# tt = {
#     'fname' : "Mohammed",
#     'lname' : "Alghamdi"
# }
#
# print(f"Only keys : {tt.keys()}")
# print(f"Only values : {tt.values()} ")
# print(f"Values and keys : {tt.items()} ")
#
# print(f"Chick if Mohammed Exists in the dictionary {'Mohammed' in tt.values() }")
#

# pi_string = '3.141592653589793'
# pi_string_list = list(pi_string)
#
# result = {}
# for x in pi_string:
#     if x.isnumeric():
#         if x not in result:
#             result[x] = 0
#         result[x] += 1
#
#
# print(result)
#
# code1 = '1234' # should return True
# code2 = '8281' # should return False
#
#
# code1_list = list(code1)
# code2_list = list(code2)
#
# for x in code2_list:
#     print(f"{x} == {code1.count(x)} ")
#     if code2_list.count(x) > 1:
#         print(f"this is repeded number {x}")



text = "This is Super super cool"

text_list = text.lower().split()
result = {}
tem_list = []
for x in text_list:
    result[text_list.count(x)] = []
for x in text_list:
    if x not in result[text_list.count(x)]:
        result[text_list.count(x)].append(x)
print(result)





