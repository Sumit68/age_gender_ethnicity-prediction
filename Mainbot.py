import logging
import os
import cv2
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from prediction import *  # calling model func
from facedata import *

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)



def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi send an image to classify!')


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def photo(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
    update.message.reply_text(
        'Okay now wait a few seconds!!!'
    )
    #img = cv2.imread('user_photo.jpg')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = facedetection()
    img1 = cv2.imread('face.jpg')
    print(img1.shape)
    age = age_prediction(img1)
    gender = gender_prediction(img1)
    ethnicity = ethnicity_prediction(img1)
    update.message.reply_text("Predicted age: "+str(age)+"\n "+"Predicted Gender: "+str(gender)+'\n '+"Predicted Ethnicity: "+str(ethnicity)+'\n')


def main():
    
    TOKEN = "" # place your token here
    updater = Updater(TOKEN, use_context=True)
    

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, photo))

    
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
