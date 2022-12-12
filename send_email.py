#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import base64
import logging as lg
import os as os
import pickle

# 3rd Party Packages
from google_auth_oauthlib.flow import InstalledAppFlow
import googleapiclient.discovery
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os import path

# My packages / Any header files
from time import sleep

"""
    Example script of how to send an email. Code inspiration is from 
    https://scriptreference.com/sending-emails-via-gmail-with-python/
"""

"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 30/11/2022 23:06
    Filename    : send_email
    IDE         : PyCharm
"""


def create_pickle_file():
    # Specify permissions to send and read/write messages
    # Find more information at:
    # https://developers.google.com/gmail/api/auth/scopes
    SCOPES = ['https://www.googleapis.com/auth/gmail.send',
              'https://www.googleapis.com/auth/gmail.modify']

    # Get the user's home directory
    home_dir = os.path.expanduser('~')
    print(f"The pickle file is located at: {home_dir}")

    # By default, we assume that 'credentials.json' is in the Downloads folder. Change accordingly.
    json_path = os.path.join(home_dir, 'Downloads', 'credentials.json')

    # Next we indicate to the API how we will be generating our credentials
    flow = InstalledAppFlow.from_client_secrets_file(json_path, SCOPES)

    # This step will generate the pickle file
    # The file gmail.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    creds = flow.run_local_server(port=0)

    # We are going to store the credentials in the user's home directory
    pickle_path = os.path.join(home_dir, 'gmail.pickle')
    with open(pickle_path, 'wb') as token:
        pickle.dump(creds, token)


def logging_setup():
    # Initialisation of basic logging information. 
    lg.basicConfig(filename='logfile.log',
                   filemode='w',
                   level=lg.DEBUG,
                   format='%(asctime)s - %(message)s',
                   datefmt='%d/%m/%Y %I:%M:%S %p',
                   force=True)


def send_email(sender_email, receiver_email, custom_msg=''):
    # Get the path to the pickle file
    home_dir = '/Users/cameronmceleney/PycharmProjects/SpinchainsAnalysis/Additional Files'
    pickle_path = os.path.join(home_dir, 'gmail.pickle')

    # Load our pickled credentials
    creds = pickle.load(open(pickle_path, 'rb'))

    # Build the service
    service = googleapiclient.discovery.build('gmail', 'v1', credentials=creds)

    # Create a message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Simulated Completed'
    msg['From'] = f'{sender_email}'
    msg['To'] = f'{receiver_email}'

    # Add message contents
    if custom_msg:
        msgPlain = custom_msg
    else:
        msgPlain = 'This is my first email!'
    # msgHtml = 'This is my first email!!'
    msg.attach(MIMEText(msgPlain, 'plain'))
    # msg.attach(MIMEText(msgHtml, 'html'))

    # Encode
    raw = base64.urlsafe_b64encode(msg.as_bytes())
    raw = raw.decode()
    body = {'raw': raw}

    message1 = body
    message = (
        service.users().messages().send(
            userId="me", body=message1).execute())
    print('Message Id: %s' % message['id'])


def main():
    path_to_file = './'
    file_name = 'simulation_metadata.txt'
    filename_and_path = f'{path_to_file}{file_name}'

    should_send_email = True

    senders_address = '2235645.desktop@gmail.com'
    receivers_address = ['cameron.mceleney@gmail.com']

    if path.exists(filename_and_path):
        # Metadata file exists and can be sent to the user along with a notification.
        with open('simulation_metadata.txt') as sim_metadata_file:
            sim_details = sim_metadata_file.read()
            print(sim_details)

            if should_send_email:
                for email_address in receivers_address:
                    send_email(senders_address, email_address, custom_msg=sim_details)

            os.remove(filename_and_path)
    else:
        # Fallback if the metadata file cannot be found; user is still notified of completed simulation.
        if should_send_email:
            inform_user = 'Your simulation is complete, but no metadata could be found'

            for email_address in receivers_address:
                send_email(senders_address, email_address, custom_msg=inform_user)

        print('no file')
        pass

    exit(0)


if __name__ == '__main__':
    # logging_setup()

    main()
