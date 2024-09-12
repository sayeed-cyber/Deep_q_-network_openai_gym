import cv2
import numpy as np
import gym
import keras
from algodqn import DeepQLearning

def my_loss_fn(y_true, y_pred):
    from tensorflow import gather_nd
    from tensorflow.keras.losses import MeanSquaredError
    
    s1, s2 = y_true.shape
    indices = np.zeros(shape=(s1, s2))
    indices[:, 0] = np.arange(s1)
    indices[:, 1] = DeepQLearning.actionsAppend  # Ensure this is set correctly

    mse_loss = MeanSquaredError()
    loss = mse_loss(gather_nd(y_true, indices=indices.astype(int)), gather_nd(y_pred, indices=indices.astype(int)))
    return loss

loaded_model = keras.models.load_model(
    "trained_model.h5",
    custom_objects={'my_loss_fn': my_loss_fn}
)

env = gym.make("CartPole-v1", render_mode="rgb_array")

frame_width = 600  
frame_height = 400 
video_writer = cv2.VideoWriter('cartpole_simulation.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

cv2.namedWindow('CartPole Simulation', cv2.WINDOW_NORMAL)

currentState = env.reset()

if isinstance(currentState, tuple):
    currentState = currentState[0] 

currentState = np.array(currentState)

terminalState = False
sumObtainedRewards = 0
max_steps = 999
steps = 0
render_frequency = 1 

while not terminalState and steps < max_steps:
    Qvalues = loaded_model.predict(currentState.reshape(1, 4))
    # Select the action that gives the max Q-value
    action = np.random.choice(np.where(Qvalues[0, :] == np.max(Qvalues[0, :]))[0])
    # Apply the action
    step_result = env.step(action)  # Expect a tuple returned
    next_state = step_result[0]
    reward = step_result[1]
    done = step_result[2]
    
    # Sum the rewards
    sumObtainedRewards += reward

    # Render the environment
    if steps % render_frequency == 0:
        frame = env.render()
        if frame is not None and not frame.size == 0:
            # Resize the frame to match the video writer dimensions
            frame_resized = cv2.resize(frame, (frame_width, frame_height))
            # Write the frame to the video file
            video_writer.write(frame_resized)
            
            # Display the frame in a window
            cv2.imshow('CartPole Simulation', frame_resized)
            
            # Check for 'q' key press to exit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Warning: Received an invalid frame from the environment.")
    
    # Update the state and terminal flag
    currentState = next_state
    terminalState = done
    steps += 1

# Release the video writer, close the environment and destroy the window
video_writer.release()
env.close()
cv2.destroyAllWindows()

print(f"Total obtained rewards: {sumObtainedRewards}")