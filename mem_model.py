import pymem
import pymem.process
import struct
import time
import mss
import numpy as np
import cv2
from digit_match import detect_digit
import pyvjoy
import random
from collections import defaultdict
import ctypes
from ctypes import wintypes
import os
import pickle
import math


# Windows API constants
PAGE_READWRITE = 0x04
MEM_COMMIT = 0x1000

# Struct to hold memory info
class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress",       wintypes.LPVOID),
        ("AllocationBase",    wintypes.LPVOID),
        ("AllocationProtect", wintypes.DWORD),
        ("RegionSize",        ctypes.c_size_t),
        ("State",             wintypes.DWORD),
        ("Protect",           wintypes.DWORD),
        ("Type",              wintypes.DWORD),
    ]

# Function to enumerate regions
def looks_like_valid_data(data):
    # Simple sanity check: not all zero or all 0xFF
    if all(b == 0x00 for b in data):
        return False
    if all(b == 0xFF for b in data):
        return False
    return True

def find_dolphin_emulated_ram(pm, known_gamecube_addrs=None):
    if known_gamecube_addrs is None:
        # Some example known GameCube RAM addresses to test
        known_gamecube_addrs = [
            0x8120CC18,  # lap_progress2 or something you know
            0x81442FE8,  # speed or similar
            0x8120CC40,  # lap_time or something else
        ]
    
    gc_base = 0x80000000
    mbi = MEMORY_BASIC_INFORMATION()
    address = 0

    while address < 0x7FFFFFFFFFFF:
        res = ctypes.windll.kernel32.VirtualQueryEx(
            pm.process_handle,
            ctypes.c_void_p(address),
            ctypes.byref(mbi),
            ctypes.sizeof(mbi)
        )
        if not res:
            break

        if (
            mbi.State == MEM_COMMIT
            and mbi.Protect == PAGE_READWRITE
            and mbi.RegionSize >= 0x02000000
        ):
            for gc_addr in known_gamecube_addrs:
                offset = gc_addr - gc_base
                try:
                    test_addr = mbi.BaseAddress + offset
                    data = pm.read_bytes(test_addr, 4)
                    if looks_like_valid_data(data):
                        print(f"[INFO] Found Dolphin RAM region at {hex(mbi.BaseAddress)} (passed test addr {hex(test_addr)})")
                        return mbi.BaseAddress
                except pymem.exception.MemoryReadError:
                    continue

        address += mbi.RegionSize

    raise RuntimeError("Could not find Dolphin emulated RAM region.")

# Usage:
pm = pymem.Pymem('Dolphin.exe')
BASE_ADDRESS = find_dolphin_emulated_ram(pm)
print(f"Detected Dolphin base address: {hex(BASE_ADDRESS)}")

a = 1
b = 2
x = 3
y = 4
start_s = 5
l = 6
r = 7
z = 8
up_d = 9
down_d = 10
left_d = 11
right_d = 12

j = pyvjoy.VJoyDevice(1)
def left():
    j.set_axis(pyvjoy.HID_USAGE_X,0x1)
def center_x():
    j.set_axis(pyvjoy.HID_USAGE_X,0x4000)
def right():
    j.set_axis(pyvjoy.HID_USAGE_X,0x8000)
def center_y():
    j.set_axis(pyvjoy.HID_USAGE_Y,0x4000)
def down():
    j.set_axis(pyvjoy.HID_USAGE_Y,0x8000)
def up():
    j.set_axis(pyvjoy.HID_USAGE_Y,0x1)
def set_btn(btn):
    j.set_button(btn, 1)
def rel_btn(btn):
    j.set_button(btn,0)


#list of things we can do
actions = [lambda:set_btn(a), lambda:rel_btn(a), lambda:set_btn(b), lambda:rel_btn(b), 
        lambda:set_btn(l), lambda:rel_btn(l), 
        lambda:set_btn(r), lambda:rel_btn(r), lambda:left(), lambda:right(), lambda:center_x()]


NUM_STATES = 100    # 0 to 99 speeds
NUM_ACTIONS = len(actions)

#comment out everything down to the else if you want to restart from scratch
Q_file = "q_table.pkl"

def zero_action_array():
    return np.zeros(NUM_ACTIONS)

if os.path.exists(Q_file):
    print("Loading existing Q-table...")
    with open(Q_file, "rb") as f:
        Q = pickle.load(f)
else:
    print("No saved Q-table found. Starting fresh.")
    Q = defaultdict(zero_action_array)

# Hyperparameters
alpha = 0.1
gamma = 0.95
epsilon = 0.1

gc_base = 0x80000000  # Dolphin's virtual memory base for GameCube
pm = pymem.Pymem('Dolphin.exe')
BASE_ADDRESS = find_dolphin_emulated_ram(pm)

def get_real_address(addr):
    offset = addr - gc_base
    return BASE_ADDRESS + offset

def read_float(addr):
    real_addr = get_real_address(addr)
    data = pm.read_bytes(real_addr, 4)
    big = struct.unpack('>f', data)[0]  # assuming big-endian float
    return big  # or whichever you confirm is correct
    '''
    # Always unpack only ONCE
    value = struct.unpack('>f', data)[0]
    return value'''

def read_byte(addr):
    offset = addr - 0x80000000  # GameCube base address
    real_addr = BASE_ADDRESS + offset
    return pm.read_bytes(real_addr, 1)[0]

speed = 0x81442FE8
accel = 0x81442FEC
lap_time = 0x8120CC40
lap_progress = 0x8120CC14
lap_progress2 = 0x8120CC18
terrain = 0x814434BB
lap_num = 0x811ED903
face_x = 0x814CB944
face_y = 0x814CB948
x_pos = 0x81449BB0
y_pos = 0x8120CF40

'''reward ideas to consider
reward = 0

# Reward for moving fast
reward += speed * 0.1

# Reward for positive acceleration
reward += max(accel, 0) * 0.05

# Reward for lap progress delta
reward += (lap_progress2 - prev_lap_progress) * 5

# Penalize time spent on non-road
if terrain != 0:
    reward -= 2

# Additional penalty if speed is too low
if speed < 5:
    reward -= 1
'''
prev_lap_progress = 0
prev_lap_progress2 = 0

def get_lap_progress():
    return safe_read_float(lap_progress, clamp_min=0.0, clamp_max=1.0)

def get_lap_progress2():
    return safe_read_float(lap_progress2, clamp_min=-1.0, clamp_max=3.0)
'''
def get_lap_progress_delta():
    global prev_lap_progress2, prev_lap_progress
    lap = read_byte(lap_num)
    lp = get_lap_progress()
    lp2 = get_lap_progress2()

    # If lap_progress2 is negative, the kart is going backward through the finish line
    if lp2 < lap:
        # Reset progress or treat delta as zero to avoid weird spikes
        prev_lap_progress = lp
        prev_lap_progress2 = lp2
        return lp2-lap-1

    # Normal delta calculation
    delta = lp - prev_lap_progress

        # Handle big jumps (possible lap reset)
    if delta < -0.5:
        prev_lap_progress = lp
        prev_lap_progress2 = lp2
        return  1 - prev_lap_progress + lp
    # If delta is negative (kart going backward), ignore it
    if delta < 0:
        prev_lap_progress = lp
        prev_lap_progress2 = lp2

    return delta
'''

def get_facing_angle():
    y_dir = read_float(face_y)
    x_dir = read_float(face_x)
    while(y_dir != y_dir):
        y_dir = read_float(face_y)
    while(x_dir != x_dir):
        x_dir = read_float(face_x)
    angle_rad = math.atan2(y_dir, x_dir)
    angle_deg = math.degrees(angle_rad)
    angle_deg = angle_deg % 360  # normalize to 0–360
    return angle_deg


def get_reward():
    global prev_lap_progress2
    reward = 0
    # Reward for moving fast
    # Reward for positive acceleration
    #reward += max(read_float(accel), 0) * 0.01
    # Reward for lap progress delta
    lap_val = read_byte(lap_num)
    if(lap_val < 0 or lap_val > 3):
        lap_val = 0
    current_lap_progress2 = safe_read_float(lap_progress2, clamp_min = -1, clamp_max = 3)
    if(current_lap_progress2 - prev_lap_progress - lap_val > .1 or current_lap_progress2 - prev_lap_progress - lap_val < 0):
        reward -= 50
    delta = current_lap_progress2 - prev_lap_progress - lap_val
    #delta = get_lap_progress_delta(prev_lap_progress)
    # Reward forward progress in the race
    if(current_lap_progress2 < lap_val):
        reward -= 5
    else:
        if(delta > 0):
            reward += read_float(speed) * 5
    reward += delta * 50000
    prev_lap_progress2 = current_lap_progress2

    # Penalize time spent on non-road
    if read_byte(terrain) != 1:
        reward -= 5
    # Additional penalty if speed is too low
    if read_float(speed) < 5:
        reward -= 2
    return reward

def start():
    for i in range(1, 12):
        rel_btn(i)
    center_x()
    center_y()
    set_btn(start_s)
    set_btn(x)
    set_btn(b)
    time.sleep(1)
    rel_btn(start_s)
    rel_btn(x)
    rel_btn(b)
    time.sleep(2)
    j.set_axis(pyvjoy.HID_USAGE_X,0x4000)
    j.set_axis(pyvjoy.HID_USAGE_Y,0x4000)
    #start
    j.set_button(5, 1)
    time.sleep(0.1)
    j.set_button(5, 0)
    time.sleep(1)
    #a
    j.set_button(1, 1)
    time.sleep(0.1)
    j.set_button(1, 0)
    time.sleep(1)
    #a
    j.set_button(1, 1)
    time.sleep(0.1)
    j.set_button(1, 0)
    time.sleep(1)
    #down
    j.set_axis(pyvjoy.HID_USAGE_Y,0x8000)
    time.sleep(0.1)
    j.set_axis(pyvjoy.HID_USAGE_Y,0x4000)
    #a 5 times
    for i in range(0,10):
        time.sleep(0.1)
        j.set_button(1, 1)
        time.sleep(0.1)
        j.set_button(1, 0)
        time.sleep(1)
    time.sleep(1)
    #higher cooldown A presses
    for i in range(0,3):
        time.sleep(0.2)
        j.set_button(1, 1)
        time.sleep(0.1)
        j.set_button(1, 0)
        time.sleep(1)
    time.sleep(4)

def apply_action(action_idx):
    # Map index to vJoy button/stick command
    actions[action_idx]()

def step(action_idx):
    apply_action(action_idx)
    time.sleep(0.1)
    new_speed = read_float(speed)
    reward = get_reward()
    return new_speed, reward

def safe_read_float(addr, clamp_min=0.0, clamp_max=None, default=0.0):
    raw = read_float(addr)
    # Handle NaN
    if raw != raw:
        return default
    # Handle Inf
    if raw == float('inf') or raw == float('-inf'):
        return default
    # Clamp
    if clamp_max is not None:
        raw = min(raw, clamp_max)
    if clamp_min is not None:
        raw = max(raw, clamp_min)
    return raw

def main_loop():
    global epsilon, prev_lap_progress, prev_lap_progress2, BASE_ADDRESS, last_good_progress
    reward = 0
    state = None
    action = None
    episode = 0
    max_steps = 4000
    lap_val = 0
    while True:
        last_good_progress = 0.0
        step_count = 0
        total_episode_reward = 0
        prev_lap_progress = 0
        prev_lap_progress2 = 0.0
        start()
        BASE_ADDRESS = find_dolphin_emulated_ram(pm)
        # Send input action here, e.g., via vJoy
        # e.g., send_joystick_input(...)
        #there will be a million while loops here.
        #they can probably be simplified, but basically, something called
        #memory tear happens where we read while its writing memory, so we want to 
        #ensure that we dont get a bad value
        while(read_byte(lap_num) != 3 and step_count < max_steps):
            #ensure not reading on a memory tear
            #read everything first (we might get lucky)
            speed_value = read_float(speed)
            x_val = read_float(x_pos)
            y_val = read_float(y_pos)
            lap_val = read_byte(lap_num)
            lap_progress_value = read_float(lap_progress2)
            terrain_id = read_byte(terrain)
            facing_angle = get_facing_angle()
            while(speed_value < -1000 or speed_value > 1000):
                speed_value = read_float(speed)
            speed_bucket = int(speed_value // 20)        # e.g., 0–10
            while(x_val < -100000 or x_val > 100000):
                x_val = read_float(x_pos)
            x_bucket = int(x_val // 1000)
            while(y_val < -100000 or y_val > 100000):
                y_val = read_float(y_pos)
            y_bucket = int(y_val // 1000)
            if(lap_val == 63):
                lap_val = 0
            while lap_val not in (0,1,2,3):
                lap_val = read_byte(lap_num)
            while(lap_progress_value < -1 or lap_progress_value > 3):
                lap_progress_value = read_float(lap_progress2)
            #safe since we just did the sanity loops
            lap_progress_relative = lap_progress_value - lap_val
            # You can bucket it simply:
            lap_progress_bucket = int(lap_progress_relative * 10)
            if(terrain_id > 5 or terrain_id < 0):
                #if bad, we will classify as not the right terrain
                terrain_id = 2
            #dealt with in get_facing_angle
            facing_angle = get_facing_angle()
            angle_bucket = int(facing_angle / 45)
            state = (speed_bucket, lap_progress_bucket, terrain_id, angle_bucket, x_bucket, y_bucket)
            # ε-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = np.argmax(Q[state])
            new_state, reward = step(action)
            total_episode_reward += reward
            new_lap_val = read_byte(lap_num)
            new_lap_progress_value = read_float(lap_progress2)
            speed_value = read_float(speed)
            next_terrain_id = read_byte(terrain)
            if(new_lap_val == 63):
                new_lap_val = 0
            while(new_lap_val not in (0, 1, 2, 3)):
                new_lap_val = read_byte(lap_num)
            while(new_lap_progress_value < -1 or new_lap_progress_value > 3):
                new_lap_progress_value = read_float(lap_progress2)
            new_lap_progress_relative = new_lap_progress_value - new_lap_val
            # Compute progress delta relative to last known "good" progress
            progress_delta = new_lap_progress_relative - last_good_progress

            # Decide reward based on the delta
            if abs(progress_delta) > 0.3:
                reward = -50.0  # Big penalty for "teleporting" or going backward too far
            elif progress_delta > 0.0:
                reward += progress_delta * 5000.0  # Reward forward progress
                last_good_progress = new_lap_progress_relative  # Update last_good_progress
            else:
                reward += progress_delta * 1000.0  # Mild penalty for small backward moves

            reward -= 0.01  # Living penalty to encourage movement
            new_lap_progress_bucket = int(new_lap_progress_relative * 10)
            while(speed_value < -1000 or speed_value > 1000):
                speed_value = read_float(speed)
            next_speed_bucket = int(speed_value // 20)
            if(next_terrain_id > 5 or next_terrain_id < 0):
                next_terrain_id = 2
            #dealt with in get_facing_angle
            next_facing_angle = get_facing_angle()
            next_angle_bucket = int(next_facing_angle / 45)
            new_x = read_float(x_pos)
            while(new_x < -100000 or new_x > 100000):
                new_x = read_float(x_pos)
            new_x_bucket = int(new_x // 1000)
            new_y = read_float(y_pos)
            while(new_y < -100000 or new_y > 100000):
                new_y = read_float(y_pos)
            new_y_bucket = int(new_y // 1000)
            new_state = (next_speed_bucket, new_lap_progress_bucket, next_terrain_id, next_angle_bucket, new_x_bucket, new_y_bucket)
            prev_lap_progress = new_lap_progress_relative
            best_next_q = np.max(Q[new_state])
            td_target = reward + gamma * best_next_q
            td_error = td_target - Q[state][action]
            
            Q[state][action] += alpha * td_error

            state = new_state
            #prev_lap_progress = read_float(lap_progress2)
            if step_count % 100 == 0:
                print(f"Episode Step {step_count}, speed={state}, total reward={total_episode_reward}")
            if step_count % 10 == 0:
                with open("q_table.pkl", "wb") as f:
                    pickle.dump(Q, f)
            step_count += 1
        epsilon = max(0.01, epsilon * 0.99)
        episode += 1


#when we want to do the starting sequence first
#time.sleep(10)
if __name__ == "__main__":
    main_loop()