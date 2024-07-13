import time
import os

def monitor_logs(log_file='./logs/log_file.log'):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    return lines[-10:]  # Display the last 10 log entries

def monitor_tensorboard():
    os.system('tensorboard --logdir=logs')

def main():
    while True:
        print("Last 10 log entries:")
        for line in monitor_logs():
            print(line.strip())
        print("\nStarting TensorBoard...")
        monitor_tensorboard()
        time.sleep(60)  # Monitor every 60 seconds

if __name__ == '__main__':
    main()
