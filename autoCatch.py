def autoCatchLinux(Name):
    from random import randint
    from keyboard import press_and_release as key
    from pyperclip import copy, paste
    from time import sleep
    key('ctrl+a')
    sleep(0.2)
    copy(f"@Pokétwo#8236 c {Name}")
    sleep(randint(5,20)/10)
    key("ctrl+v")#paste()
    sleep(0.2)
    key('enter')

def autoPauseLinux(content):
    if "Whoa there."in content:
        key('ctrl+a')
        sleep(0.2)
        copy('@Pokétwo#8236 inc p')
        sleep(0.2)
        key('ctrl+v')
        key('enter')
def autoResumeLinux(content):
    if "Spawns Remaining: 0."in content:
        key('ctrl+a')
        sleep(0.2)
        copy('@Pokétwo#8236 inc buy 30minute 30second -y')
        sleep(0.2)
        key('ctrl+c')
        key('enter')

