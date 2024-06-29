# -*- coding: utf-8 -*-

import random
import pickle as pkl
with open('/content/DIACRITICS_LIST.pickle', 'rb') as file:
    DIACRITICS_LIST = pkl.load(file)
DIACRITICS_LIST = DIACRITICS_LIST + [" َّ", "ٌّ", " ٍّ", "ًّ", "ُّ", "ِّ"]

with open('/content/ARABIC_LETTERS_LIST.pickle', 'rb') as file:
    ARABIC_LETTERS_LIST = pkl.load(file)

def remove_diacritics(data_raw):
  ''' Returns undiacritized text'''
  return data_raw.translate(str.maketrans('', '', ''.join(DIACRITICS_LIST)))

def replace_char_at_index(input_string, index, replacement):
    # Convert the string to a list of characters
    char_list = list(input_string)

    # Check if the index is within the bounds of the list
    if 0 <= index < len(char_list):
        # Replace the character at the specified index
        char_list[index] = replacement

    # Convert the list back to a string
    modified_string = ''.join(char_list)
    return modified_string

#replace the Shaddah and remove the related diacritics of it.
def replace_first_Shaddah(input_string):

  modified_string=input_string
  if modified_string[0]in ARABIC_LETTERS_LIST:
    if(modified_string[1]=='ّ'):
     print("it is start with shaddh ",modified_string )
     #chose any thing from the list expect for shaddah
     rand_char=random.choice([diacritic for diacritic in DIACRITICS_LIST if diacritic != 'ّ'])
     print("rand_char ", rand_char)
     modified_string = replace_char_at_index(modified_string,1,rand_char)
     if(modified_string[2] in DIACRITICS_LIST):
      print("it has another diac")
      modified_string = replace_char_at_index(modified_string,2,'')

  print("replace_first_Shaddah ", modified_string)
  return modified_string

def remove_random_diac_from_almad(input_string):
#No random diacritics for the almad character

    modified_string=input_string
    #print("word ", modified_string)
    #print("len ",len(modified_string) )

    index = 0
    while index < len(modified_string)-1:

         if modified_string[index] == "ا" : #not first char and there is a character before it
           if modified_string[index+1] in DIACRITICS_LIST:
            #print("yes", index+1,modified_string[index+1])
            modified_string=replace_char_at_index(modified_string, index+1,"") #remove the next char which is the Diacritic
            #print("yes2",modified_string)
           if  index!=0:
            if modified_string[index-1] in DIACRITICS_LIST:
             modified_string=replace_char_at_index(modified_string, index-1,"َ") # fatha can be before alef
         index += 1

    print("remove_random_diac_from_almad ", modified_string)
    return modified_string

#replace the Shaddah and remove the related diacritics of it.
def change_diacritic_for_alef_with_hamza(input_string):
    modified_string=input_string
    #print("word ", modified_string)

    for char_index in  range(len(modified_string)-2):

      #print("index ", char_index)
      #print(modified_string[char_index])
      if modified_string[char_index]=='أ':

       #if the character alrady have shadda, remove it
       if modified_string[char_index+1]=='ّ':
          modified_string = replace_char_at_index(modified_string,char_index+1,"")

       else:
        if modified_string[char_index+1] in DIACRITICS_LIST:
         if modified_string[char_index+1] != 'َ' or modified_string[char_index+1] != 'ُ':
          rand_char=random.choice([diacritic for diacritic in ['ُ','َ']])
          modified_string = replace_char_at_index(modified_string,char_index+1,rand_char)

      #check hamza under alph
    for char_index in  range(len(modified_string)-1):
      if modified_string[char_index]=='إ':
       if modified_string[char_index+1] in DIACRITICS_LIST:
        if modified_string[char_index+1] != 'ّ':
         modified_string = replace_char_at_index(modified_string,char_index+1,"")
         #print("Deleted")

       else:
        if modified_string[char_index+1] in DIACRITICS_LIST:
         if modified_string[char_index+1] != 'ِ':

          rand_char="ِ"
          modified_string = replace_char_at_index(modified_string,char_index+1,'ِ')
          #print( modified_string[char_index+1])

    print("change_diacritic_for_alef_with_hamza ", modified_string)
    return modified_string

def change_random_diacritic_before_feminie_taa(input_string):
    #no random diacritic for the letter before the feminie taa

    modified_string=input_string
    for index, char in enumerate(input_string):
      if char in ["ة"]:
        if  modified_string[index-1] in DIACRITICS_LIST: #check if the previous is character or diacritic
          modified_string=replace_char_at_index(modified_string, index-1,"َ") #replcae diac with fathaa
        else:
         modified_string=modified_string[:index] + 'َ' + modified_string[index:] #add fatahaa to the previous character

    return modified_string

def remove_diacritics_from_alef_with_maddah(input_string):
    # no random diacritics for alef with maddah
    modified_string=input_string
    for index, char in enumerate(input_string):
        if char == "آ":
          if modified_string[index+1] in DIACRITICS_LIST:
            modified_string=replace_char_at_index(modified_string, index+1,"")

    return modified_string

def remove_diacritics_from_alef_with_alef_maqsura(input_string):
    # no random diacritics for alef with maddah
    modified_string=input_string
    index=0
    while index < len(input_string)-1:
        if modified_string[index] == "ى":
          if input_string[index+1] in DIACRITICS_LIST:
            modified_string=replace_char_at_index(modified_string, index+1,"")
        index+=1


    return modified_string

def remove_end_word_diacritics(input_string):

    modified_string=input_string
    for index, char in enumerate(input_string):
        if modified_string[len(modified_string)-1] in DIACRITICS_LIST:
            modified_string=replace_char_at_index(modified_string, (len(modified_string)-1),"")

    print("remove_end_word_diacritics ", modified_string)
    return modified_string

def remove_diacs_from_non_letters(input_string):
    modified_string=input_string
    index=0
    while index <(len(input_string)-2):
       if modified_string[index] not in ARABIC_LETTERS_LIST and modified_string[index] not in DIACRITICS_LIST:
          if modified_string[index+1] in DIACRITICS_LIST:
            modified_string=replace_char_at_index(modified_string, index+1,"")
       index+=1
    print("remove_diacs_from_non_letters ", modified_string)
    return modified_string

def apply_sukun_rules(input_string):

    #no two consecutive sukuns

    modified_string=input_string
    index=1
    while index < (len(modified_string)-2):

      if modified_string[index]=='ْ' and  modified_string[index+2]=='ْ' :

        modified_string=replace_char_at_index(modified_string,index+2,"")
      index+=1

    return modified_string

def solve_contiguous_diacs(input_string):

    #no two consecutive sukuns

    modified_string=input_string
    index=0
    while index < (len(modified_string)-1):
      if modified_string[index] in DIACRITICS_LIST and  modified_string[index+1] in DIACRITICS_LIST and modified_string[index] !='ّ' :
        modified_string=replace_char_at_index(modified_string,index+1,"")
      index+=1

    return modified_string

def apply_shaddh_followed_by_diac(input_string):
    modified_string = input_string
    index = 0
    while index < len(modified_string) - 1:
        if modified_string[index] == 'ّ' and modified_string[index + 1] in ARABIC_LETTERS_LIST:
            #print("modified_string[index] ", modified_string[index])
            #print("modified_string[index+1] ", modified_string[index+1])
            rand_char = random.choice(["َ"])
            modified_string = modified_string[:index + 1] + rand_char + modified_string[index + 1:]
            index += 2  # Move to the character after the newly added diacritic
        else:
            index += 1  # Move to the next character

    # Handle the last character separately
    if modified_string[-1] == 'ّ':
        rand_char = random.choice(["َ"])#(["َ", "ُ" , "ِ"])
        modified_string = modified_string + rand_char  # Add diacritic to the last character

    return modified_string

def apply_tanween_rules(input_string):
    """
    tanween can only occur at the end of diacritics
    """
    tanween_marks = ["ً","ٌ","ٍ","ًّ","ُّ","ٍّ"]
    modified_string=input_string
    modified_string=solve_contiguous_diacs(modified_string)
    index=0
    while index < len(modified_string)-1:

        if modified_string[index] in tanween_marks :
            print("iii")
            #print(modified_string[index])
            rand_char=random.choice([diacritic for diacritic in ["َ","ُ","ِ"]]) #chose an random rather than tanween
            modified_string=replace_char_at_index(modified_string,index,rand_char)

        index += 1
    print("apply_tanween_rules ", modified_string)
    return modified_string

def check_all_conditions_bool(input_string):

  # 1. check first character Shaddah
  for char_index in  range(len(input_string)):
   #print(input_string)
   #print("len ",(len(input_string)) )
   if(len(input_string)>1):
    if(input_string[1]=='ّ'):
     input_string=replace_first_Shaddah(input_string)


  # 2. check Alef Diacritics
  for char_index in  range(len(input_string)-1):
      if input_string[char_index]=='أ':
       if input_string[char_index+1] != 'َ' or input_string[char_index+1] != 'ُ':
        input_string= change_diacritic_for_alef_with_hamza(input_string)
      if input_string[char_index]=='إ':
       if input_string[char_index+1] != 'ِ':
         input_string=change_diacritic_for_alef_with_hamza(input_string)

  # 3. check madd characters
  #for index in range (len(input_string)-2):
  ind=0
  while ind < len(input_string)-1:
     if(input_string[ind] in ARABIC_LETTERS_LIST):
       if input_string[ind] in ["ا"] :
          if input_string[ind+1] in DIACRITICS_LIST:
           input_string= remove_random_diac_from_almad(input_string)
     ind+=1


  for index in range (len(input_string)-2):
        if  input_string[index] in ["و", "ي"]:
          if input_string[index+1] in DIACRITICS_LIST:
            input_string= remove_random_diac_from_almad(input_string)

  # 4. change_random_diacritic_before_feminie_taa marboota
  for index, char in enumerate(input_string):
      if char in ["ة"]:
        if  input_string[index-1] in DIACRITICS_LIST:
          input_string= change_random_diacritic_before_feminie_taa(input_string)

  # 5. check the diacritics_from_al (ال)
  index=0
  while(index < (len(input_string)-3)):
      if(input_string[index]=='ا' and input_string[index+1] in DIACRITICS_LIST and  input_string[index+2]=='ل'): # اَل
        input_string= remove_diacritics_from_al(input_string)
      if(input_string[index]=='ا' and  input_string[index+1]=='ل'and input_string[index+2] in DIACRITICS_LIST ): # َال
        input_string= remove_diacritics_from_al(input_string)
      if(input_string[index]=='ا' and input_string[index+1] in DIACRITICS_LIST and  input_string[index+2]=='ل'and input_string[index+3] in DIACRITICS_LIST ): # َاَل
        input_string= remove_diacritics_from_al(input_string)
      index+=1

  # 6. remove_diacritics_from_alef_with_maddah
  for index, char in enumerate(input_string):
        if char == "آ":
          if input_string[index+1] in DIACRITICS_LIST:
            input_string= remove_diacritics_from_alef_with_maddah(input_string)

   # 7. remove_diacritics_from_alef_with_maddah
  ind_alef=0
  while ind_alef < len(input_string)-1:
        if char == "ى":
          if input_string[ind_alef+1] in DIACRITICS_LIST:
            input_string= remove_diacritics_from_alef_with_alef_maqsura(input_string)
        ind_alef+=1


  # 8. remove_diacs_from_non_letters
  for index in range (len(input_string)-1):
       if input_string[index] not in ARABIC_LETTERS_LIST and input_string[index] not in DIACRITICS_LIST:
          if input_string[index+1] in DIACRITICS_LIST:
           input_string=remove_diacs_from_non_letters(input_string)

  # 9. apply_sukun_rules (no 2 sukun can follow each other)
  inde_sukun=0
  while inde_sukun< (len(input_string)-2):
      if input_string[inde_sukun]=='ْ' and  input_string[inde_sukun+2]=='ْ' :
        input_string=apply_sukun_rules(input_string)
      inde_sukun+=1

  # 10. apply_tanween_rules
  tanween_marks = ['ً',  'ٍ', 'ٌ','ٌّ',' ٍّ', 'ًّ']
  index_tanw=0
  while index_tanw < len(input_string)-1:
        if input_string[index_tanw] in tanween_marks:
         input_string=apply_tanween_rules(input_string)
        index_tanw+=1

  # 11.check the contiguous_diacs
  index_Cdi=0
  while index_Cdi < (len(input_string)-1):
      if input_string[index_Cdi] in DIACRITICS_LIST and  input_string[index_Cdi+1] in DIACRITICS_LIST :
        input_string=solve_contiguous_diacs(input_string)
      index_Cdi+=1


  # 12.apply_shaddh_followed_by_diac
  index_shaddah = 0
  while index_shaddah < len(input_string) - 1:
        if input_string[index_shaddah] == 'ّ' and input_string[index_shaddah + 1] in ARABIC_LETTERS_LIST:
          input_string=apply_shaddh_followed_by_diac(input_string)
        if input_string[-1] == 'ّ':
          input_string=apply_shaddh_followed_by_diac(input_string)
        index_shaddah+=1


  return input_string

def assign_random_dic (input_string):
  rand_diac=[]
  output_string=''
  for index in range(len(input_string)):
    if input_string[index] in ARABIC_LETTERS_LIST:
     rand_char=random.choice([diacritic for diacritic in DIACRITICS_LIST])
     output_string += input_string[index] + rand_char

  return output_string

def apply_sukun_rules(input_string):

    #no two consecutive sukuns

    modified_string=input_string
    index=0
    while index < (len(modified_string)-2):

      if modified_string[index]=='ْ' and  modified_string[index+2]=='ْ' :
        rand_char=random.choice([diacritic for diacritic in ["َ", "ُ", "ِ"]])
        modified_string=replace_char_at_index(modified_string,index,rand_char)
      index+=1

    return modified_string

def solve_consecutive_shaddahs(input_string):

    #no two consecutive sukuns

    modified_string=input_string
    index=0
    while index < (len(modified_string)-4):

      if modified_string[index]=="ّ" and  modified_string[index+3]=="ّ" :
        rand_char=random.choice([diacritic for diacritic in ["َ", "ُ", "ِ"]])
        modified_string=replace_char_at_index(modified_string,index,rand_char)
        modified_string=replace_char_at_index(modified_string,index+1,"")
      index+=1

    return modified_string
