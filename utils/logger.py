import datetime

class SimpleLogger:
    def __init__(self):
        self.log_file = f"logs/vehicle_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log(self, message):
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        # Ghi vào file (tùy chọn)
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_message + '\n')
        except:
            pass  # Không quan trọng nếu không ghi được file 