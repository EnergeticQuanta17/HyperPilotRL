import pyautogui
import time

# Delay to give time to switch to the window where you want to send the key presses
time.sleep(5)

# Send the sequence of keys 50 times
for i in range(5):
    pyautogui.press('space')
    pyautogui.press('-')
    pyautogui.press('space')
    pyautogui.press('delete')
    pyautogui.press('delete')
    pyautogui.press('down')
    pyautogui.press('down')
    pyautogui.press('end')