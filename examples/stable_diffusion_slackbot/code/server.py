from flask import Flask, request 
from slack_sdk import WebClient
from oxen import RemoteRepo
from threading import Thread
from slack_sdk.errors import SlackApiError
import dotenv
import os
import requests
import codecs
import shortuuid
import time
import json
from unidecode import unidecode
import hashlib

# Configure things
dotenv.load_dotenv()
client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])
app = Flask(__name__)

# Configure Oxen repo
repo = RemoteRepo("ba/slackbot-oxen")
repo.checkout("dev")

# Constants
VOTES = {
    "+1": "Approve",
    "-1": "Disapprove",
}
IMAGE_DIR = "images"
DF_PATH = "annotations/train.csv"
MODEL_VERSION = "0-1"
HASH_LENGTH = 12
USER_HASH_LENGTH = 6

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def hash_file_id(file_id):
    """Hash slack file id into a file name"""
    return hashlib.sha256(file_id.encode('utf-8')).hexdigest()[0:HASH_LENGTH]

def commit_df_to_oxen(row):
    start_commit = time.time()
    try:
        repo.add_df_row(DF_PATH, row)
        print("time to add row", time.time() - start_commit)
        repo.commit(f"Remote commit - {row['rater']} voting on image {row['path'].split('/')[-1]}")
    except Exception as e:
        print('Error adding df row to Oxen', e)    
    print(f"Time to commit: {time.time() - start_commit}")

def commit_image_to_oxen(filepath):
    start_time_image = time.time()
    try:
        repo.add(filepath, IMAGE_DIR)
        repo.commit(f"Adding image image {filepath.split('/')[-1]}")
    except Exception as e:
        print('Error adding image to Oxen', e)
    print(f"Time to commit image: {time.time() - start_time_image}")

def download_image(url, image_id):
    headers = {"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"}
    img_data = requests.get(url, headers=headers).content
    file_hash = hash_file_id(image_id)
    file_path = f'{IMAGE_DIR}/{file_hash}.png'

    with open(file_path, 'wb') as handler:
        handler.write(img_data)
    return file_path

def fetch_message(conversation_id, message_ts):
    try:
        result = client.conversations_history(
            channel=conversation_id,
            inclusive=True,
            oldest = message_ts,
            limit=1
        )
    except SlackApiError as e:
        print(f"Error: {e}")
    return result

def fetch_file(file_id):
    try:
        file = client.files_info(
            file=file_id,
            count=1
        )
    except SlackApiError as e:
        print(f"Error: {e}")
    return file
    

def is_valid_reaction(reaction_data):
    if reaction_data['event']['reaction'] not in list(VOTES.keys()):
        print("Invalid reaction, skipping")
    return "Success"

def is_valid_reaction_message(message_data):
    if len(message_data["messages"]) < 1:
        print("No message found, aborting")
        return False
    message = message_data["messages"][0]
    if message["user"] != os.environ["SLACK_BOT_USER_ID"]:
        print("Message not authored by bot user, aborting")
        return False
    # Check if the message has files 
    if "files" not in message:
        print("No files found, aborting")
        return False
    return True

def is_valid_file_upload(file_data: str) -> bool:
    if file_data['event']['user_id'] != os.environ["SLACK_BOT_USER_ID"]:
        print("File upload not by bot user, aborting")
        return False
    return True


def handle_reaction(reaction_data, message_data):
    start_reaction = time.time()
    file_id = message_data["files"][0]["id"]
    file_hash = hash_file_id(file_id)
    prompt = unidecode(message_data["files"][0]["title"])
    oxen_row = {
        "prompt": prompt,
        "path": f"{IMAGE_DIR}/{file_hash}.png",
        "rating": VOTES[reaction_data['event']['reaction']],
        "rater": hashlib.sha256(reaction_data['event']['user'].encode("utf-8")).hexdigest()[0:USER_HASH_LENGTH],
        "model_version": MODEL_VERSION,
    }
    print("Time elapsed before upload df call", time.time() - start_reaction)
    commit_df_to_oxen(oxen_row)
    print("Total execution time for reaction: ", time.time() - start_reaction)

def handle_file_upload(file_id):
    file = fetch_file(file_id)
    image_url = file["file"]["url_private_download"]
    filepath = download_image(image_url, file_id)
    # Commit to oxen
    commit_image_to_oxen(filepath)
        

@app.route("/")
def hello():
    return ""

# Post route to accept incoming data 
@app.route("/post", methods=["POST"])
def post():
    start = time.time()
    data = request.get_json()
    if data['event']['type'] == 'file_created':
        if not is_valid_file_upload(data):
            return "Skipping"
        
        thr = Thread(target=handle_file_upload, args=[data['event']['file_id']])
        thr.start()
        
        print("Total execution time for file upload: ", time.time() - start)
        return "Success"

    if data['event']['type'] == 'reaction_added':
        if not is_valid_reaction(data):
            return "Skipping"
        
        conversation_id = data['event']['item']['channel']
        message_ts = data['event']['item']['ts']
        
        message_data = fetch_message(conversation_id, message_ts)
        
        if not is_valid_reaction_message(message_data):
            return "Skipping"
        
        message = message_data["messages"][0]

        thr = Thread(target=handle_reaction, args=[data, message])
        thr.start() 
        
        return "Success"

    print("total execution time", time.time() - start)
    return "Success"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)


