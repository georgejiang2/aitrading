import discord
from discord.ext import tasks
import subprocess
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Bot setup
intents = discord.Intents.default()
bot = discord.Client(intents=intents)

# Channel ID where you want to post results
CHANNEL_ID = 1350579022182223933  # Replace with your actual channel ID

# Path to your script
SCRIPT_PATH = "option.py"  # Replace with actual path

# Path to your image (if your script generates an image, this would be that path)
IMAGE_PATH = "images/test.png"  # Replace with actual image path

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    run_script_periodically.start()

@tasks.loop(minutes=2)
async def run_script_periodically():
    try:
        # Run your script and capture output
        result = subprocess.check_output(['python', SCRIPT_PATH], text=True)
        
        # Get the channel to post to
        channel = bot.get_channel(CHANNEL_ID)
        
        if channel:
            # Create a file object from your image path
            file = discord.File(IMAGE_PATH)
            
            # Send the image with an optional message
            await channel.send(content=f"{result[:1000]}", file=file)
        else:
            print(f"Error: Couldn't find channel with ID {CHANNEL_ID}")
    
    except Exception as e:
        print(f"Error running script: {e}")
        
        # Optionally send error message to Discord
        channel = bot.get_channel(CHANNEL_ID)
        if channel:
            await channel.send(f"Error running script: {e}")

# Run the bot
bot_token = os.getenv("DISCORD_TOKEN")
bot.run(bot_token)