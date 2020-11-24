from collections import defaultdict
from threading import Thread
from pathlib import Path
from queue import Queue
import subprocess
import ezgmail 
import logging 
import time
import csv
import os
import re

class tasks_items:
    def __init__(self,task,task_type,sender,section,timestamp,completion_timestamp):
        self.task = task
        self.task_type = task_type
        self.sender = sender
        self.section =  section
        self.timestamp = timestamp
        self.completion_timestamp = completion_timestamp

def init_enviroment():
    # Create a new folder for each runtime of the program to store data
    instance_time = str(time.strftime("%a-%d-%b-%Y-%H-%M-%S")) # Mon-1-Jan-2020-00-00-00
    instance_folder = r'\Instance-' + instance_time 
    instance_path = r'C:\YOLOv5\yolov5\Email Automation\Instances' + instance_folder
    os.mkdir(instance_path)
    # Initialize logging
    logFile = instance_path + r'\runtime_log.txt'
    logging.basicConfig(filename=logFile, filemode='w+', level=logging.DEBUG)
    
def init_ezgmail():
    print("Initialing email connection...")
    # Path to the gmail account API JSON credentials file
    os.chdir(r'C:\YOLOv5\yolov5\Email Automation\Credentials')
    # Initialize the mail client with JSON credentials
    ezgmail.init()
    logging.info("Succesfully initializing email connection.")
    print("Succesfully initializing email connection." + "\n")

def check_inbox():
    # get a list of all the email threads of unread emails
    logging.info("\n" + "Checking for new unread emails at: " + str(time.strftime("%H:%M:%S")) #23:04:44
    print("Checking for new unread emails at: " + str(time.strftime("%H:%M:%S")) #23:04:44
    unreadThreads = ezgmail.unread()
    numUnreads =  len(unreadThreads)
    print("Number of Unread Emails: " + str(numUnreads))
    return(unreadThreads)

def send_email(recipient, subject, body):
    logging.info("Atempting to send email.\n" + "Recipient: " + recipient + "\n" + "Subject: " + subject + "\n" + "Body: " + body)
    print("Attempting to send and email to:" + recipient)
    ezgmail.send(recipient, subject, body)
    logging.info("Email Sent")
    print("Email Sent")

def main():
    # Initialize the program files and email connection
    init_enviroment()
    init_ezgmail()

    # Opens up the csv of all tasks
    task_path = r'C:\YOLOv5\yolov5\Email Automation\task-history.csv'
    tasks = []

    # Read in the previous tasks in case of a crash
    with open(task_path,'r') as csv_taskHistory:
        reader = csv.reader(csv_taskHistory)
        task = list(reader)


if __name__ == '__main__':
    main()

# Debugging messages allowed
# logging.debug("debug message")
# logging.info("info message")
# logging.warning("warning message")
# logging.error("error message")
# logging.critical("critical message")

task_history = open(task_path,'w') # 'w+' denotes the file is opened as writeable and create one if it doesn't exist
write_tasks = csv.writer(task_history)
write_tasks.writerows(tasks)
task_history.flush()
# Columns in CSV file
#fields = ['task','task_type','sender','section','timestamp','completion_timestamp']

# Set the starting task number
current_task = len(tasks)
print(F"Starting Task Number: {current_task}")

while True:

    queue = Queue()
    # get a list of all the email threads of unread emails
    unreadThreads = check_inbox()

    for unread in unreadThreads:
        message = unread.messages
        sender = re.findall(r'\<(.*?)\>', str(message[0].sender))[0] # Return the email address enclosed in <>
        subject = message[0].subject
        body = message[0].body
        timestamp = message[0].timestamp

        # Check to make sure the email came from Four Front Design
        if re.finditer(r'@fourfrontdesign', str(sender)) is not None:
            # Check subject to determine it is an Autocropping task
            if re.match('autocropping', str(subject), re.IGNORECASE):
                print(F"New Unread Autocropping Taskfrom: {sender} Recieved at {timestamp}")
                logging.info(F"New Unread Autocropping Task from: {sender} Recieved at {timestamp}")
                section = str(re.findall(r'\A([^\n]+)', str(body))[0]) # Return the first line up to the first \n (newline) character
                section = str(section.strip()) # remove the chariage return from end of line
                print(F"Section to autocrop: {section}")
                logging.info(F"Section to autocrop: {section}")
                print(F"Assigned task number: {current_task}")
                logging.info(F"Assigned task number: {current_task}")
                new_task = [current_task, 'Autocropping', sender, section, timestamp]
                queue.append(new_task)
                if queue is not None:
                    # Task has been placed in Queue
                    send_email(sender, F"Task Placed In Queue!", F"Your request to process: \n {section} \n Task has been placed in the queue. \n You will be emailed when it completes.")
                current_task += 1
                unread.markAsRead()
        else:
            continue

    completed = []
    for item in queue:
        #send_email(item[2], F"Task Has Begun Processing!", F"Section: \n {item[3]} \n Has begun processing. \n You will be emailed when it completes.")
        print(F"Autocropping: {item[3]}")
        logging.info(F"Autocropping: {item[3]}")
        command = r'C:\YOLOv5\yolov5\Scripts\python.exe C:\YOLOv5\yolov5\detect_dev.py  --source ' + '"' + item[3] + '"'
        os.system(command)
        send_email(item[2], F"Task Has Completed Succesfully!", F"Section: \n {item[3]} \n Has completed succesfully. \n If you have additional items in the queue you will be emailed separately.")
        completion_time = str(time.strftime("%Y-%m-%d %H:%M:%S")) # 2020-11-02 23:04:44
        item.append(completion_time)
        completed.append([str(item[0]), str(item[1]), str(item[2]), str(item[3]), str(item[4]), str(item[5])])

    write_tasks.writerows(completed)
    task_history.flush()

    time.sleep(120) # run every # minutes