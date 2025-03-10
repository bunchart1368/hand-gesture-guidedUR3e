import keyboard

count = 0

while True:
    try:
        if keyboard.is_pressed("ctrl"):
            print(f"Ctrl key pressed {count} times")
            count += 1
    except:
        break
