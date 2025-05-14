"""Test a random policy on the OpenAI Gym Hopper environment.

    
    TASK 1: Play around with this code to get familiar with the
            Hopper environment.

            For example:
                - What is the state space in the Hopper environment? Is it discrete or continuous?
                - What is the action space in the Hopper environment? Is it discrete or continuous?
                - What is the mass value of each link of the Hopper environment, in the source and target variants respectively?
                - what happens if you don't reset the environment even after the episode is over?
                - When exactly is the episode over?
                - What is an action here?
"""
import pdb
import gym
from env.custom_hopper import *
import numpy as np
from IPython.display import HTML
from IPython import display
import PIL.Image
from base64 import b64encode

def create_video_frames(frames, filename='animation.gif', _return=True):
    """
    Save frames as an animated GIF or return HTML animation.
    """
    # Save as GIF
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )
    
    if _return:
        # Create HTML animation for notebook display
        with open(filename, 'rb') as f:
            video_file = f.read()
        video_url = data_uri = f'data:image/gif;base64,{b64encode(video_file).decode()}'
        return HTML(f'<img src="{video_url}" />')

def main():
	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('State space:', env.observation_space) # state-space
	print('Action space:', env.action_space) # action-space
	print('Dynamics parameters:', env.get_parameters()) # masses of each link of the Hopper

	n_episodes = 2  # Reduced number of episodes for visualization
	frames = []  # Store frames for animation

	for episode in range(n_episodes):
		done = False
		state = env.reset()	# Reset environment to initial state

		while not done:  # Until the episode is over
			action = env.action_space.sample()	# Sample random action
			state, reward, done, info = env.step(action)	# Step the simulator to the next timestep

			# Render and capture frame
			frame = env.render(mode='rgb_array')
			# Convert to PIL Image
			frame = PIL.Image.fromarray(frame)
			frames.append(frame)

	# Create and display animation
	return create_video_frames(frames)

if __name__ == '__main__':
	video = main()
	if 'google.colab' in str(get_ipython()):
		display.display(video)