def autoCatchLinux(Name):
    from random import randint
    from keyboard import press_and_release as key
    from pyperclip import copy, paste
    from time import sleep
    key('enter')
    sleep(0.2)
    copy(f"@Pokétwo#8236 c {Name}")
    sleep(randint(5,20)/10)
    key("ctrl+v")#paste()
    sleep(0.2)
    key('enter')
