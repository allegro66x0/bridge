import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from hardware.coin_sorter import CoinSorter

# CONFIG
TEST_PORT = 'COM11' # Change to your actual port

def main():
    print(f"Connecting to Coin Sorter on {TEST_PORT}...")
    sorter = CoinSorter(TEST_PORT)
    
    if not sorter.connect():
        print("Failed to connect.")
        return

    print("Connected!")
    print("1. Testing Feeder (M1) at speed 200")
    sorter.set_feeder_speed(200)
    time.sleep(2)
    sorter.set_feeder_speed(0)
    
    print("2. Testing Conveyors (M2, M3) at speed 180")
    sorter.set_conveyor_speed(180)
    time.sleep(2)
    sorter.set_conveyor_speed(0)
    
    print("3. Testing Start Sequence (All Motors)")
    sorter.start_all()
    time.sleep(3)
    
    print("4. Stop All")
    sorter.stop_all()
    
    sorter.close()
    print("Test Complete.")

if __name__ == "__main__":
    main()
